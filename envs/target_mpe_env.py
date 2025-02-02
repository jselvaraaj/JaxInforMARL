from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from config.mappo_config import CommunicationType
from .default_env_config import (
    AGENT_COLOR,
    CONTACT_FORCE,
    CONTACT_MARGIN,
    CONTINUOUS_ACT,
    DAMPING,
    DISCRETE_ACT,
    DT,
    MAX_STEPS,
    OBS_COLOR,
)
from .multiagent_env import (
    AgentLabel,
    MultiAgentAction,
    MultiAgentEnv,
    MultiAgentState,
    PRNGKey,
    default,
    entity_labels_to_indices,
)
from .schema import (
    RGB,
    AgentIndexAxis,
    CoordinateAxisIndexAxis,
    EntityIndex,
    EntityIndexAxis,
    GraphsTupleWithAgentIndex,
    Info,
    MultiAgentDone,
    MultiAgentGraph,
    MultiAgentObservation,
    MultiAgentReward,
)
from .spaces import Box, Discrete


class MPEState(NamedTuple):
    """Basic MPE State"""
    dones: Bool[Array, AgentIndexAxis]
    step: int
    entity_positions: Float[Array, f"{EntityIndexAxis} {CoordinateAxisIndexAxis}"]
    entity_velocities: Float[Array, f"{EntityIndexAxis} {CoordinateAxisIndexAxis}"]
    did_agent_die_this_time_step: Float[Array, f"{AgentIndexAxis}"]
    agent_communication_message: Float[Array, f"{AgentIndexAxis} ..."] | None
    agent_visibility_radius: Float[Array, f"{AgentIndexAxis}"]


class LinSpaceConfig(NamedTuple):
    lin_range: tuple[int, int]
    lin_step: float


class TargetMPEEnvironment(MultiAgentEnv):
    """
    Discrete Actions  - [do nothing, left, right, down, up] where the 0-indexed value, correspond to action value.
    Continuous Actions - [x, y, z, w, a, b] where each continuous value corresponds to
                        the magnitude of the discrete actions.

    """

    def __init__(
            self,
            num_agents=3,
            action_type=DISCRETE_ACT,
            agent_labels: None | list[AgentLabel] = None,
            action_spaces: dict[AgentLabel, Discrete | Box] = None,
            observation_spaces: dict[AgentLabel, Discrete | Box] = None,
            color: RGB = None,
            # communication_message_dim: int = 0,
            position_dim: int = 2,
            max_steps: int = MAX_STEPS,
            dt: float = DT,
            collision_reward_coefficient=-5,
            agent_visibility_radius=None,
            agent_max_speed: int = 1,
            entity_acceleration=1,
            entities_initial_coord_radius=None,
            one_time_death_reward=2,
            agent_communication_type=None,
            agent_control_noise_std=0,
            add_self_edges_to_nodes=False,
            distance_to_goal_reward_coefficient=5,
            add_target_goal_to_nodes=True,
            heterogeneous_agents=False,
    ):
        super().__init__(
            num_agents=num_agents,
            max_steps=max_steps,
            action_spaces=action_spaces,
            observation_spaces=observation_spaces,
            agent_labels=agent_labels,
        )

        if entities_initial_coord_radius is None:
            entities_initial_coord_radius = (1,)
        if agent_visibility_radius is None:
            agent_visibility_radius = (0.5,)

        self.num_landmarks = num_agents
        self.num_entities = self.num_agents + self.num_landmarks
        self.agent_indices = jnp.arange(self.num_agents)
        self.entity_indices = jnp.arange(self.num_entities)
        self.landmark_indices = jnp.arange(self.num_agents, self.num_entities)

        self.agent_communication_type = agent_communication_type

        self.heterogeneous_agents = heterogeneous_agents

        if heterogeneous_agents:
            self.agent_entity_type = self.entity_indices[: self.num_agents]
            self.landmark_entity_type = self.entity_indices[self.num_agents:]
        else:
            self.agent_entity_type = jnp.zeros(self.num_agents)
            self.landmark_entity_type = jnp.ones(self.num_landmarks)

        # Assumption agent_i corresponds to landmark_i
        self.landmark_labels = [f"landmark_{i}" for i in range(self.num_landmarks)]
        self.landmark_labels_to_index = entity_labels_to_indices(
            self.landmark_labels, start=self.num_agents
        )

        self.add_self_edges_to_nodes = add_self_edges_to_nodes

        self.add_target_goal_to_nodes = add_target_goal_to_nodes

        assert action_type in [DISCRETE_ACT, CONTINUOUS_ACT], "Invalid action type"
        if action_type == DISCRETE_ACT:
            self.action_spaces = default(
                self.action_spaces, {i: Discrete(5) for i in self.agent_labels}
            )
        else:
            self.action_spaces = default(
                self.action_spaces, {i: Box(-1, 1, (2,)) for i in self.agent_labels}
            )

        self.action_to_control_input = (
            self._discrete_action_to_control_input
            if action_type == DISCRETE_ACT
            else self._continuous_action_to_control_input
        )

        self.one_time_death_reward = jnp.full((self.num_agents,), one_time_death_reward)
        self.distance_to_goal_reward_coefficient = distance_to_goal_reward_coefficient

        self.observation_spaces = default(
            self.observation_spaces,
            {_id: Box(-jnp.inf, jnp.inf, (6,)) for _id in self.agent_labels},
        )
        self.entities_initial_coord_radius = jnp.asarray(entities_initial_coord_radius)

        assert (
                color is None or len(color) == num_agents + self.num_landmarks
        ), "color must have length num_agents + num_landmarks. Note num_landmark = num_agents"
        self.color = default(
            color, [AGENT_COLOR] * self.num_agents + [OBS_COLOR] * self.num_landmarks
        )

        self.agent_visibility_radius = jnp.asarray(agent_visibility_radius)

        assert (
                collision_reward_coefficient <= 0.0
        ), "collision_reward must be less than 0"
        self.collision_reward_coefficient = collision_reward_coefficient

        # self.communication_message_dim = communication_message_dim
        self.position_dim = position_dim

        # Environment specific parameters
        self.dt = dt
        self.max_steps = max_steps
        self.entity_radius = jnp.concatenate(
            [jnp.full(self.num_agents, 0.15), jnp.full(self.num_landmarks, 0.2)]
        )
        self.is_moveable = jnp.concatenate(
            [
                jnp.full(self.num_agents, True),
                jnp.full(self.num_landmarks, False),
            ]
        )
        self.can_entity_collide = jnp.concatenate(
            [
                jnp.full(self.num_agents, True),
                jnp.full(self.num_landmarks, False),
            ]
        )
        self.entity_mass = jnp.full(self.num_entities, 1.0)
        self.entity_acceleration = jnp.full(self.num_agents, entity_acceleration)

        self.entity_max_speed = jnp.concatenate(
            [
                jnp.full(self.num_agents, agent_max_speed),
                jnp.full(self.num_landmarks, 0.0),
            ]
        )
        self.agent_control_noise = jnp.full(self.num_agents, agent_control_noise_std)
        # self.communication_noise = self.velocity_noise = jnp.concatenate(
        #     [
        #         jnp.full(self.num_agents, 0),
        #         jnp.full(self.num_agents, 0),
        #     ]
        # )
        self.damping = DAMPING
        self.contact_force = CONTACT_FORCE
        self.contact_margin = CONTACT_MARGIN

    # noinspection DuplicatedCode
    @partial(jax.vmap, in_axes=[None, 0, 0])
    def _discrete_action_to_control_input(
            self,
            agent_index: Int[Array, f"{AgentIndexAxis}"],
            action: Int[Array, f"{AgentIndexAxis}"],
    ) -> Float[Array, f"{AgentIndexAxis} {CoordinateAxisIndexAxis}"]:
        u = jnp.zeros((self.position_dim,))
        x_axis = 0
        y_axis = 1
        action_to_coordinate_axis = jax.lax.select(action <= 2, x_axis, y_axis)
        increase_position = 1.0
        decrease_position = -1.0
        u_val = jax.lax.select(
            action % 2 == 0, increase_position, decrease_position
        ) * (action != 0)
        u = u.at[action_to_coordinate_axis].set(u_val)
        u = u * self.entity_acceleration[agent_index] * self.is_moveable[agent_index]
        return u

    def _continuous_action_to_control_input(self, action: Array) -> tuple[float, float]:
        pass

    def _discrete_action_by_label_to_control_input(
            self, actions: MultiAgentAction
    ) -> Float[Array, f"{AgentIndexAxis} {CoordinateAxisIndexAxis}"]:
        actions = jnp.array(
            [actions[agent_label] for agent_label in self.agent_labels]
        ).reshape((self.num_agents, -1))

        return self._discrete_action_to_control_input(self.agent_indices, actions)

    @partial(jax.jit, static_argnums=[0])
    def reset(
            self,
            key: PRNGKey,
            initial_agent_communication_message: Float[Array, f"{AgentIndexAxis} ..."],
            initial_entity_position: Float[Array, f"{EntityIndexAxis} ..."],
    ) -> tuple[MultiAgentObservation, MultiAgentGraph, MPEState]:
        """Initialise with random positions"""

        key_agent, key_landmark, key_initial_coord, key_visibility_radius = (
            jax.random.split(key, 4)
        )

        r = jax.random.choice(key_initial_coord, self.entities_initial_coord_radius)
        agent_visibility_radius = jnp.full(
            self.num_agents,
            jax.random.choice(key_visibility_radius, self.agent_visibility_radius),
        )

        @partial(jax.jit, static_argnums=(0,))
        def sample_points(num_points, key, min_dist_between_points, bounds=(0, 1)):
            def body_fun(state):
                key, points, num_accepted = state
                key, subkey = jax.random.split(key)
                new_point = jax.random.uniform(
                    subkey, (2,), minval=bounds[0], maxval=bounds[1]
                )
                distances = jnp.sqrt(jnp.sum((points - new_point) ** 2, axis=1))

                # Create a boolean mask indicating which rows (points) are accepted
                # i.e., from index 0 up to num_accepted-1.
                mask = jnp.arange(num_points) < num_accepted

                # "Ignore" distances for unaccepted slots by setting them to +inf
                # so they won't affect the minimum-dist checks.
                distances = jnp.where(mask, distances, jnp.inf)

                is_valid = jnp.all(distances >= min_dist_between_points) | (
                        num_accepted == 0
                )

                points = jax.lax.dynamic_update_slice(
                    points, jnp.expand_dims(new_point, 0), (num_accepted, 0)
                )
                num_accepted += is_valid

                return key, points, num_accepted

            init_points = jnp.zeros((num_points, 2))
            init_state = (key, init_points, 0)

            final_state = jax.lax.while_loop(
                lambda state: state[2] < num_points, body_fun, init_state
            )

            return final_state[1]

        if initial_entity_position.size == 0:
            entity_positions = jnp.concatenate(
                [
                    jax.random.uniform(
                        key_agent, (self.num_agents, 2), minval=-r, maxval=+r
                    ),
                    sample_points(
                        self.num_landmarks, key_landmark, 0.5, bounds=(-r, +r)
                    ),
                ]
            )
        else:
            entity_positions = initial_entity_position

        state = MPEState(
            entity_positions=entity_positions,
            entity_velocities=jnp.zeros((self.num_entities, self.position_dim)),
            dones=jnp.full(self.num_agents, False),
            step=0,
            did_agent_die_this_time_step=jnp.full(self.num_agents, False),
            agent_communication_message=initial_agent_communication_message,
            agent_visibility_radius=agent_visibility_radius,
        )
        obs = self.get_observation(state)
        graph = self.get_graph(state)

        return obs, graph, state

    @partial(jax.jit, static_argnums=[0])
    def get_observation(self, state: MPEState) -> MultiAgentObservation:
        """Return dictionary of agent observations"""

        @partial(jax.vmap, in_axes=[0, None])
        def _observation(
                agent_idx: Int[Array, AgentIndexAxis], state: MPEState
        ) -> Float[Array, f"{AgentIndexAxis} 3*{CoordinateAxisIndexAxis}"]:
            """Return observation for agent i."""
            landmark_idx = self.num_agents + agent_idx
            landmark_position = state.entity_positions[landmark_idx]
            agent_position = state.entity_positions[agent_idx]
            agent_velocity = state.entity_velocities[agent_idx]
            landmark_relative_position = landmark_position - agent_position

            return jnp.concatenate(
                [
                    agent_position.flatten() - agent_position.flatten(),
                    agent_velocity.flatten(),
                    landmark_relative_position.flatten(),
                ]
            )

        observation = _observation(self.agent_indices, state)
        return {
            agent_label: observation[i]
            for i, agent_label in enumerate(self.agent_labels)
        }

    @partial(jax.jit, static_argnums=[0])
    def get_graph(self, state: MPEState) -> MultiAgentGraph:

        if self.agent_communication_type == CommunicationType.HIDDEN_STATE.value:
            landmark_communication_message = jnp.zeros_like(
                state.agent_communication_message
            )
            communication_message = jnp.concatenate(
                [state.agent_communication_message, landmark_communication_message]
            )
        elif (
                self.agent_communication_type == CommunicationType.PAST_ACTION.value
                or self.agent_communication_type == CommunicationType.CURRENT_ACTION.value
        ):
            landmark_communication_message = jnp.zeros_like(
                state.agent_communication_message
            )
            communication_message = jnp.concatenate(
                [state.agent_communication_message, landmark_communication_message]
            )

        @partial(jax.vmap, in_axes=(None, 0))
        @partial(jax.vmap, in_axes=(0, None))
        def get_node_feature(
                entity_idx: Int[Array, EntityIndexAxis],
                agent_id: Int[Array, AgentIndexAxis],
        ) -> tuple[Int[Array, f"{AgentIndexAxis} {EntityIndexAxis}  num_equivariant_features 2"], Int[
            Array, f"{AgentIndexAxis} {EntityIndexAxis}  num_non_equivariant_features"]]:
            goal_idx = jnp.where(
                entity_idx < self.num_agents, self.num_agents + entity_idx, entity_idx
            )
            goal_relative_coord = jnp.asarray([])
            if self.add_target_goal_to_nodes:
                goal_relative_coord = (
                        state.entity_positions[goal_idx] - state.entity_positions[agent_id]
                )
            relative_position = (
                    state.entity_positions[entity_idx] - state.entity_positions[agent_id]
            )
            relative_velocity = (
                    state.entity_velocities[entity_idx] - state.entity_velocities[agent_id]
            )
            entity_type = jnp.where(
                entity_idx < self.num_agents,
                self.agent_entity_type[entity_idx],
                self.landmark_entity_type[entity_idx - self.num_agents],
            )
            node_communication_message = jnp.asarray([])
            if self.agent_communication_type is not None:
                node_communication_message = communication_message[entity_idx]

            equivariant_node_features = jnp.stack(
                [relative_position,
                 relative_velocity,
                 goal_relative_coord]
            )
            non_equivariant_node_features = jnp.concatenate(
                [node_communication_message, jnp.array([entity_type])]
            )

            return equivariant_node_features, non_equivariant_node_features

        ### 2) Compute pairwise distances in one shot
        # agent_positions shape: (num_agents, 2)
        agent_positions = state.entity_positions[self.agent_indices]
        # entity_positions shape: (num_entities, 2)
        entity_positions = state.entity_positions
        # Broadcast to shape: (num_agents, num_entities, 2)
        # distances shape: (num_agents, num_entities)
        distances = jnp.linalg.norm(
            agent_positions[:, None] - entity_positions[None, :], axis=-1
        )
        mask = distances <= state.agent_visibility_radius[:, None]

        if not self.add_self_edges_to_nodes:
            mask = mask.at[self.agent_indices, self.agent_indices].set(False)

        max_num_edge = self.num_agents * self.num_entities
        valid_agent_idx, valid_entity_idx = jnp.nonzero(
            mask, size=max_num_edge, fill_value=-1
        )
        # Receivers = agent indices, Senders = entity indices (since edges go entity->agent here).
        receivers = valid_agent_idx  # shape: (num_valid_edges,)
        senders = valid_entity_idx  # shape: (num_valid_edges,)

        edge_features = distances[valid_agent_idx, valid_entity_idx][..., None]

        if self.add_self_edges_to_nodes:
            # add self edges for landmarks
            receivers = jnp.concatenate([self.landmark_indices, receivers])
            senders = jnp.concatenate([self.landmark_indices, senders])
        edge_features = jnp.concatenate(
            [edge_features, jnp.zeros(self.num_landmarks)[..., None]]
        )

        # edges = get_agent_to_entity_edge(self.agent_indices)
        # receivers, senders, edge_features = jax.tree.map(jnp.ravel, edges)
        # receivers, senders = add_landmark_self_edges(receivers, senders)

        equivariant_node_features, non_equivariant_node_features = get_node_feature(
            self.entity_indices, self.agent_indices
        )
        n_node = jnp.array([self.num_entities])
        n_edge = jnp.array([receivers.shape[0]])
        agent_label_to_graph = {
            agent_label: GraphsTupleWithAgentIndex(
                equivariant_nodes=equivariant_node_features[
                    self.agent_labels_to_index[agent_label]
                ],
                non_equivariant_nodes=non_equivariant_node_features[
                    self.agent_labels_to_index[agent_label]
                ],
                edges=edge_features,
                globals=None,
                receivers=receivers,
                senders=senders,
                n_node=n_node,
                n_edge=n_edge,
                agent_indices=jnp.asarray(self.agent_labels_to_index[agent_label]),
            )
            for agent_label in self.agent_labels
        }

        return agent_label_to_graph

    @partial(jax.vmap, in_axes=[None, 0, 0, 0, 0])
    def _control_to_agents_forces(
            self,
            key: PRNGKey,
            u: Float[Array, f"{AgentIndexAxis} {CoordinateAxisIndexAxis}"],
            u_noise: Int[Array, AgentIndexAxis],
            moveable: Bool[Array, AgentIndexAxis],
    ):
        noise = jax.random.normal(key, shape=u.shape) * u_noise
        zero_force = jnp.zeros_like(u)
        return jax.lax.select(moveable, u + noise, zero_force)

    def _add_environment_force(
            self,
            all_forces: Float[Array, f"{EntityIndexAxis} {CoordinateAxisIndexAxis}"],
            state: MPEState,
    ) -> Float[Array, f"{EntityIndexAxis} {CoordinateAxisIndexAxis}"]:
        """gather physical forces acting on entities"""

        @partial(jax.vmap, in_axes=[0])
        def _force_on_entities_from_all_other_entities(
                entity_i: Int[Array, EntityIndexAxis]
        ) -> Float[
            Array, f"{EntityIndexAxis} {EntityIndexAxis} {CoordinateAxisIndexAxis}"
        ]:
            @partial(jax.vmap, in_axes=[None, 0])
            def _force_between_pair_of_entities(
                    entity_a: int, entity_b: Int[Array, EntityIndexAxis]
            ) -> Float[
                Array, f"{EntityIndexAxis} {EntityIndexAxis} {CoordinateAxisIndexAxis}"
            ]:
                lower_triangle_in_axb = entity_b <= entity_a
                zero_collision_force = jnp.zeros((2, 2))
                collision_force_between_a_to_b_or_b_to_a = self._get_collision_force(
                    entity_a, entity_b, state  # type: ignore
                )
                collision_force = jax.lax.select(
                    lower_triangle_in_axb,
                    zero_collision_force,
                    collision_force_between_a_to_b_or_b_to_a,
                )
                return collision_force

            force_on_entity_from_all_other_entities = _force_between_pair_of_entities(
                entity_i, self.entity_indices  # type: ignore
            )

            ego_force = jnp.sum(
                force_on_entity_from_all_other_entities[:, 0], axis=0
            )  # ego force from other agents
            combined_forces = force_on_entity_from_all_other_entities[:, 1]
            combined_forces = combined_forces.at[entity_i].set(ego_force)

            return combined_forces

        forces = _force_on_entities_from_all_other_entities(self.entity_indices)
        forces = jnp.sum(forces, axis=0)

        return forces + all_forces

    # get collision forces for any contact between two entities
    def _get_collision_force(
            self, entity_a: int, entity_b: int, state: MPEState
    ) -> Float[Array, f"{EntityIndexAxis} {CoordinateAxisIndexAxis}"]:
        distance_min = self.entity_radius[entity_a] + self.entity_radius[entity_b]
        delta_position = (
                state.entity_positions[entity_a] - state.entity_positions[entity_b]
        )

        distance = jnp.sqrt(jnp.sum(jnp.square(delta_position)))

        # softmax penetration
        k = self.contact_margin
        penetration = jnp.logaddexp(0, -(distance - distance_min) / k) * k
        force = self.contact_force * delta_position / distance * penetration
        force_a = +force * self.is_moveable[entity_a]
        force_b = -force * self.is_moveable[entity_b]
        force = jnp.array([force_a, force_b])

        no_collision_condition = (
                (~self.can_entity_collide[entity_a])
                | (~self.can_entity_collide[entity_b])
                | (entity_a == entity_b)
        )
        zero_collision_force = jnp.zeros((2, 2))
        return jax.lax.select(no_collision_condition, zero_collision_force, force)

    @partial(jax.vmap, in_axes=[None, 0, 0, 0, 0, 0, 0])
    def _integrate_state(
            self,
            all_forces: Float[Array, f"{EntityIndexAxis} {CoordinateAxisIndexAxis}"],
            entity_positions: Float[Array, f"{EntityIndexAxis} {CoordinateAxisIndexAxis}"],
            entity_velocities: Float[Array, f"{EntityIndexAxis} {CoordinateAxisIndexAxis}"],
            mass: Float[Array, EntityIndexAxis],
            moveable: Bool[Array, EntityIndexAxis],
            max_speed: Float[Array, EntityIndexAxis],
    ):
        """integrate physical state"""

        entity_positions += entity_velocities * self.dt
        entity_velocities = entity_velocities * (1 - self.damping)

        entity_velocities += (all_forces / mass) * self.dt * moveable

        speed = jnp.sqrt(
            jnp.square(entity_velocities[0]) + jnp.square(entity_velocities[1])
        )
        over_max = entity_velocities / speed * max_speed

        entity_velocities = jax.lax.select(
            (speed > max_speed) & (max_speed >= 0), over_max, entity_velocities
        )

        return entity_positions, entity_velocities

    def _double_integrator_dynamics(
            self,
            key: PRNGKey,
            state: MPEState,
            u: Float[Array, f"{AgentIndexAxis} {CoordinateAxisIndexAxis}"],
            death_mask: Bool[Array, f"{AgentIndexAxis}"],
    ) -> tuple[
        Float[Array, f"{EntityIndexAxis} {CoordinateAxisIndexAxis}"],
        Float[Array, f"{EntityIndexAxis} {CoordinateAxisIndexAxis}"],
    ]:
        # apply agent physical controls
        key_noise = jax.random.split(key, self.num_agents)

        can_agent_move = self.is_moveable[: self.num_agents] & ~death_mask

        agents_forces = self._control_to_agents_forces(
            key_noise,
            u,
            self.agent_control_noise,
            can_agent_move,
        )

        # apply environment forces
        all_forces = jnp.concatenate(
            [agents_forces, jnp.zeros((self.num_landmarks, 2))]
        )
        all_forces = self._add_environment_force(all_forces, state)

        can_entity_move = jnp.concatenate(
            [can_agent_move, self.is_moveable[self.num_agents:]]
        )

        # integrate physical state
        entity_positions, entity_velocities = self._integrate_state(
            all_forces,
            state.entity_positions,
            state.entity_velocities,
            self.entity_mass,
            can_entity_move,
            self.entity_max_speed,
        )

        return entity_positions, entity_velocities

    def _step(
            self,
            key: PRNGKey,
            state: MPEState,
            actions: MultiAgentAction,
    ) -> tuple[
        MultiAgentObservation,
        MultiAgentGraph,
        MultiAgentState,
        MultiAgentReward,
        MultiAgentDone,
        Info,
    ]:
        u = self._discrete_action_by_label_to_control_input(actions)

        key, key_double_integrator = jax.random.split(key)

        # death masking
        is_agent_dead = jax.vmap(self.is_there_overlap, in_axes=(0, 0, None))(
            self.agent_indices, self.landmark_indices, state
        )

        entity_positions, entity_velocities = self._double_integrator_dynamics(
            key_double_integrator, state, u, is_agent_dead
        )
        dones = jnp.asarray(state.step >= self.max_steps) | is_agent_dead

        did_agent_die_this_time_step = (
                state.did_agent_die_this_time_step ^ is_agent_dead
        )

        agent_positions = jnp.where(
            did_agent_die_this_time_step[..., None],
            entity_positions[self.num_agents:],
            entity_positions[: self.num_agents],
        )

        agent_velocities = jnp.where(
            did_agent_die_this_time_step[..., None],
            entity_velocities[self.num_agents:],
            entity_velocities[: self.num_agents],
        )

        entity_positions = entity_positions.at[: self.num_agents].set(agent_positions)
        entity_velocities = entity_velocities.at[: self.num_agents].set(
            agent_velocities
        )

        state = MPEState(
            entity_positions=entity_positions,
            entity_velocities=entity_velocities,
            dones=dones,
            step=state.step + 1,
            did_agent_die_this_time_step=did_agent_die_this_time_step,
            agent_communication_message=state.agent_communication_message,
            agent_visibility_radius=state.agent_visibility_radius,
        )
        reward = self.reward(state)

        observation = self.get_observation(state)
        graph = self.get_graph(state)
        dones_with_agent_label = {
            agent_label: dones[i] for i, agent_label in enumerate(self.agent_labels)
        }
        dones_with_agent_label.update({"__all__": jnp.all(dones)})

        return observation, graph, state, reward, dones_with_agent_label, {}

    def reward(self, state: MPEState) -> dict[AgentLabel, Float]:
        """Return dictionary of agent rewards"""

        @partial(jax.vmap, in_axes=[0, None])
        def _dist_between_target_reward(
                agent_index: Int[Array, AgentIndexAxis], state: MPEState
        ) -> Float[Array, AgentIndexAxis]:
            # reward is the negative distance from agent to landmark
            corresponding_landmark_index = self.num_agents + agent_index
            return -jnp.sum(
                jnp.square(
                    state.entity_positions[agent_index]
                    - state.entity_positions[corresponding_landmark_index]
                ),
            )

        @partial(jax.vmap, in_axes=(0, None))
        def _collisions(agent_idx: Int[Array, "..."], other_idx: Int[Array, "..."]):
            return jax.vmap(self.is_collision, in_axes=(None, 0, None))(
                agent_idx,
                other_idx,
                state,  # type: ignore
            )

        agent_agent_collision = _collisions(
            self.agent_indices,
            self.agent_indices,
        )  # [agent, agent, collison]

        # def _agent_rew(agent_idx: int, collisions: Bool[Array, "..."]):
        #     rew = -1 * jnp.sum(collisions[agent_idx])
        #     return rew
        dist_reward = _dist_between_target_reward(self.agent_indices, state)

        global_dist_rew = self.distance_to_goal_reward_coefficient * jnp.sum(
            dist_reward
        )
        global_agent_collision_rew = jnp.sum(agent_agent_collision)

        global_reward = (
                global_dist_rew
                + self.collision_reward_coefficient * global_agent_collision_rew
        )
        one_time_reaching_goal_reward = jnp.sum(
            jax.lax.select(
                state.did_agent_die_this_time_step,
                self.one_time_death_reward,
                jnp.zeros_like(self.one_time_death_reward),
            )
        )

        return {
            agent_label: global_reward + one_time_reaching_goal_reward
            for agent_label, agent_index in self.agent_labels_to_index.items()
        }

    def is_there_overlap(self, a: EntityIndex, b: EntityIndex, state: MPEState):
        dist_min = self.entity_radius[a] + self.entity_radius[b]
        delta_pos = state.entity_positions[a] - state.entity_positions[b]
        dist = jnp.sqrt(jnp.sum(jnp.square(delta_pos)))
        return dist < dist_min

    def is_collision(self, a: EntityIndex, b: EntityIndex, state: MPEState):
        """check if two entities are colliding"""
        return (
                self.is_there_overlap(a, b, state)
                & (self.can_entity_collide[a] & self.can_entity_collide[b])
                & (a != b)
        )

    @partial(jax.vmap, in_axes=(None, None, 0), out_axes=1)
    @partial(jax.jit, static_argnums=(0, 1))
    def get_viz_states(self, lin_space_config: LinSpaceConfig, state: MPEState):

        # only produce grid mesh for first agent assuming other agents state are fixed.
        agent_idx = self.agent_indices[0]

        @partial(jax.vmap, in_axes=(None, 0, 0))
        def get_states_for_each_coord(state: MPEState, x: Array, y: Array) -> MPEState:
            return state.replace(
                entity_positions=state.entity_positions.at[agent_idx, 0]
                .set(x)
                .at[agent_idx, 1]
                .set(y)
            )

        x_lin_space = jnp.arange(
            start=lin_space_config.lin_range[0],
            stop=lin_space_config.lin_range[1],
            step=lin_space_config.lin_step,
        )
        y_lin_space = jnp.arange(
            start=lin_space_config.lin_range[0],
            stop=lin_space_config.lin_range[1],
            step=lin_space_config.lin_step,
        )
        x_v, y_v = jnp.meshgrid(x_lin_space, y_lin_space)

        x_v = x_v.flatten()
        y_v = y_v.flatten()

        return get_states_for_each_coord(state, x_v, y_v)

    # noinspection DuplicatedCode
    @partial(jax.vmap, in_axes=[None, 0])
    def discrete_action_to_viz_vector(
            self,
            action: Int[Array, "..."],
    ) -> Float[Array, f"{AgentIndexAxis} {CoordinateAxisIndexAxis}"]:
        u = jnp.zeros((self.position_dim,))
        x_axis = 0
        y_axis = 1
        action_to_coordinate_axis = jax.lax.select(action <= 2, x_axis, y_axis)
        increase_position = 1.0
        decrease_position = -1.0
        u_val = jax.lax.select(
            action % 2 == 0, increase_position, decrease_position
        ) * (action != 0)
        u = u.at[action_to_coordinate_axis].set(u_val)
        return u

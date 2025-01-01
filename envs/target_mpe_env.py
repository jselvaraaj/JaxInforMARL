from functools import partial

import jax
import jax.numpy as jnp
from flax import struct
from jaxtyping import Float, Array, Int, Bool

from .default_env_config import (
    DISCRETE_ACT,
    MAX_STEPS,
    DT,
    AGENT_COLOR,
    OBS_COLOR,
    CONTINUOUS_ACT,
    CONTACT_FORCE,
    CONTACT_MARGIN,
    DAMPING,
)
from .multiagent_env import (
    MultiAgentState,
    MultiAgentEnv,
    AgentLabel,
    PRNGKey,
    MultiAgentAction,
    entity_labels_to_indices,
    default,
)
from .schema import (
    AgentIndex,
    EntityIndex,
    RGB,
    CoordinateAxisIndex,
    MultiAgentObservation,
)
from .spaces import Discrete, Box


@struct.dataclass
class MPEState(MultiAgentState):
    """Basic MPE State"""

    entity_positions: Float[Array, f"{EntityIndex} {CoordinateAxisIndex}"]
    entity_velocities: Float[Array, f"{EntityIndex} {CoordinateAxisIndex}"]
    communication_messages: Float[Array, f"{AgentIndex} communication_dim"] | None = (
        None
    )
    goal: int | None = None


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
        local_ratio=0.5,
    ):
        super().__init__(
            num_agents=num_agents,
            max_steps=max_steps,
            action_spaces=action_spaces,
            observation_spaces=observation_spaces,
            agent_labels=agent_labels,
        )

        self.num_landmarks = num_agents
        self.num_entities = self.num_agents + self.num_landmarks
        self.agent_indices = jnp.arange(self.num_agents)
        self.entity_indices = jnp.arange(self.num_entities)
        self.landmark_indices = jnp.arange(self.num_agents, self.num_entities)

        assert 0.0 <= local_ratio <= 1.0, "local_ratio must be between 0.0 and 1.0"
        self.local_ratio = local_ratio

        # Assumption agent_i corresponds to landmark_i
        self.landmark_labels = [f"landmark_{i}" for i in range(self.num_landmarks)]
        self.landmark_labels_to_index = entity_labels_to_indices(
            self.landmark_labels, start=self.num_agents
        )

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

        self.observation_spaces = default(
            self.observation_spaces,
            {_id: Box(-jnp.inf, jnp.inf, (6,)) for _id in self.agent_labels},
        )

        assert (
            color is None or len(color) == num_agents + self.num_landmarks
        ), "color must have length num_agents + num_landmarks. Note num_landmark = num_agents"
        self.color = default(
            color, [AGENT_COLOR] * self.num_agents + [OBS_COLOR] * self.num_landmarks
        )

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
        self.is_agent_silent = jnp.full(self.num_agents, 1)
        self.can_entity_collide = jnp.concatenate(
            [
                jnp.full(self.num_agents, True),
                jnp.full(self.num_landmarks, False),
            ]
        )
        self.entity_mass = jnp.full(self.num_entities, 1.0)
        self.entity_acceleration = jnp.full(self.num_agents, 5.0)
        self.entity_max_speed = jnp.concatenate(
            [jnp.full(self.num_agents, -1), jnp.full(self.num_landmarks, 0.0)]
        )
        self.agent_control_noise = jnp.full(self.num_agents, 0)
        # self.communication_noise = self.velocity_noise = jnp.concatenate(
        #     [
        #         jnp.full(self.num_agents, 0),
        #         jnp.full(self.num_agents, 0),
        #     ]
        # )
        self.damping = DAMPING
        self.contact_force = CONTACT_FORCE
        self.contact_margin = CONTACT_MARGIN

    @partial(jax.vmap, in_axes=[None, 0, 0])
    def _discrete_action_to_control_input(
        self,
        agent_index: Int[Array, f"{AgentIndex}"],
        action: Int[Array, f"{AgentIndex}"],
    ) -> Float[Array, f"{AgentIndex} {CoordinateAxisIndex}"]:
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
    ) -> Float[Array, f"{AgentIndex} {CoordinateAxisIndex}"]:
        actions = jnp.array(
            [actions[agent_label] for agent_label in self.agent_labels]
        ).reshape((self.num_agents, -1))

        return self._discrete_action_to_control_input(self.agent_indices, actions)

    @partial(jax.jit, static_argnums=[0])
    def reset(self, key: PRNGKey) -> tuple[MultiAgentObservation, MPEState]:
        """Initialise with random positions"""

        key_agent, key_landmark = jax.random.split(key)

        entity_positions = jnp.concatenate(
            [
                jax.random.uniform(
                    key_agent, (self.num_agents, 2), minval=-1.0, maxval=+1.0
                ),
                jax.random.uniform(
                    key_landmark, (self.num_landmarks, 2), minval=-1.0, maxval=+1.0
                ),
            ]
        )

        state = MPEState(
            entity_positions=entity_positions,
            entity_velocities=jnp.zeros((self.num_entities, self.position_dim)),
            dones=jnp.full(self.num_agents, False),
            step=0,
        )

        return self.get_observations(state), state

    @partial(jax.jit, static_argnums=[0])
    def get_observations(self, state: MPEState) -> MultiAgentObservation:
        """Return dictionary of agent observations"""

        @partial(jax.vmap, in_axes=[0, None])
        def _observation(
            agent_idx: Int[Array, AgentIndex], state: MPEState
        ) -> Float[Array, f"{AgentIndex} 3*{CoordinateAxisIndex}"]:
            """Return observation for agent i."""
            landmark_idx = self.num_agents + agent_idx
            landmark_position = state.entity_positions[landmark_idx]
            agent_position = state.entity_positions[agent_idx]
            agent_velocity = state.entity_velocities[agent_idx]
            landmark_relative_position = landmark_position - agent_position

            return jnp.concatenate(
                [
                    agent_position.flatten(),
                    agent_velocity.flatten(),
                    landmark_relative_position.flatten(),
                ]
            )

        observation = _observation(self.agent_indices, state)
        return {
            agent_label: observation[i]
            for i, agent_label in enumerate(self.agent_labels)
        }

    @partial(jax.vmap, in_axes=[None, 0, 0, 0, 0])
    def _control_to_agents_forces(
        self,
        key: PRNGKey,
        u: Float[Array, f"{AgentIndex} {CoordinateAxisIndex}"],
        u_noise: Int[Array, AgentIndex],
        moveable: Bool[Array, AgentIndex],
    ):
        noise = jax.random.normal(key, shape=u.shape) * u_noise
        zero_force = jnp.zeros_like(u)
        return jax.lax.select(moveable, u + noise, zero_force)

    def _add_environment_force(
        self,
        all_forces: Float[Array, f"{EntityIndex} {CoordinateAxisIndex}"],
        state: MPEState,
    ) -> Float[Array, f"{EntityIndex} {CoordinateAxisIndex}"]:
        """gather physical forces acting on entities"""

        @partial(jax.vmap, in_axes=[0])
        def _force_on_entities_from_all_other_entities(
            entity_i: Int[Array, EntityIndex]
        ) -> Float[Array, f"{EntityIndex} {EntityIndex} {CoordinateAxisIndex}"]:
            @partial(jax.vmap, in_axes=[None, 0])
            def _force_between_pair_of_entities(
                entity_a: int, entity_b: Int[Array, EntityIndex]
            ) -> Float[Array, f"{EntityIndex} {EntityIndex} {CoordinateAxisIndex}"]:
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
    ) -> Float[Array, f"{EntityIndex} {CoordinateAxisIndex}"]:
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
        all_forces: Float[Array, f"{EntityIndex} {CoordinateAxisIndex}"],
        entity_positions: Float[Array, f"{EntityIndex} {CoordinateAxisIndex}"],
        entity_velocities: Float[Array, f"{EntityIndex} {CoordinateAxisIndex}"],
        mass: Float[Array, EntityIndex],
        moveable: Bool[Array, EntityIndex],
        max_speed: Float[Array, EntityIndex],
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
        u: Float[Array, f"{AgentIndex} {CoordinateAxisIndex}"],
    ) -> [
        Float[Array, f"{EntityIndex} {CoordinateAxisIndex}"],
        Float[Array, f"{EntityIndex} {CoordinateAxisIndex}"],
    ]:
        # apply agent physical controls
        key_noise = jax.random.split(key, self.num_agents)
        agents_forces = self._control_to_agents_forces(
            key_noise,
            u,
            self.agent_control_noise,
            self.is_moveable[: self.num_agents],
        )

        # apply environment forces
        all_forces = jnp.concatenate(
            [agents_forces, jnp.zeros((self.num_landmarks, 2))]
        )
        all_forces = self._add_environment_force(all_forces, state)

        # integrate physical state
        entity_positions, entity_velocities = self._integrate_state(
            all_forces,
            state.entity_positions,
            state.entity_velocities,
            self.entity_mass,
            self.is_moveable,
            self.entity_max_speed,
        )

        return entity_positions, entity_velocities

    def _step(
        self,
        key: PRNGKey,
        state: MPEState,
        actions: MultiAgentAction,
    ):
        u = self._discrete_action_by_label_to_control_input(actions)

        key, key_double_integrator = jax.random.split(key)

        entity_positions, entity_velocities = self._double_integrator_dynamics(
            key_double_integrator, state, u
        )
        dones = jnp.full(self.num_agents, state.step >= self.max_steps)

        state = MPEState(
            entity_positions=entity_positions,
            entity_velocities=entity_velocities,
            dones=dones,
            step=state.step + 1,
        )
        reward = self.reward(state)
        observation = self.get_observations(state)
        dones_with_agent_label = {
            agent_label: dones[i] for i, agent_label in enumerate(self.agent_labels)
        }
        dones_with_agent_label.update({"__all__": jnp.all(dones)})

        return observation, state, reward, dones_with_agent_label, {}

    def reward(self, state: MPEState) -> dict[AgentLabel, Float]:
        """Return dictionary of agent rewards"""

        @partial(jax.vmap, in_axes=[0, None])
        def _dist_between_target_reward(
            agent_index: Int[Array, AgentIndex], state: MPEState
        ) -> Float[Array, AgentIndex]:
            # reward is the negative distance from agent to landmark
            corresponding_landmark_index = self.num_agents + agent_index
            return -jnp.sum(
                jnp.square(
                    state.entity_positions[agent_index]
                    - state.entity_positions[corresponding_landmark_index]
                ),
            )

            # reward is the negative distance from agent to landmark that is closest to it
            # return -1 * jnp.min(
            #     jnp.sum(
            #         jnp.square(
            #             state.entity_positions[agent_index]
            #             - state.entity_positions[self.num_agents :]
            #         ),
            #         axis=1,
            #     )
            # )

        @partial(jax.vmap, in_axes=(0, None))
        def _collisions(agent_idx: Int[Array, "..."], other_idx: Int[Array, "..."]):
            return jax.vmap(self.is_collision, in_axes=(None, 0, None))(
                agent_idx,
                other_idx,
                state,
            )

        agent_agent_collision = _collisions(
            self.agent_indices,
            self.agent_indices,
        )  # [agent, agent, collison]

        def _agent_rew(agent_idx: int, collisions: Bool[Array, "..."]):
            rew = -1 * jnp.sum(collisions[agent_idx])
            return rew

        dist_reward = _dist_between_target_reward(self.agent_indices, state)

        return {
            agent_label: (1 - self.local_ratio) * dist_reward[agent_index]
            + self.local_ratio * _agent_rew(agent_index, agent_agent_collision)
            for agent_label, agent_index in self.agent_labels_to_index.items()
        }

    def is_collision(self, a: EntityIndex, b: EntityIndex, state: MPEState):
        """check if two entities are colliding"""
        dist_min = self.entity_radius[a] + self.entity_radius[b]
        delta_pos = state.entity_positions[a] - state.entity_positions[b]
        dist = jnp.sqrt(jnp.sum(jnp.square(delta_pos)))
        return (
            (dist < dist_min)
            & (self.can_entity_collide[a] & self.can_entity_collide[b])
            & (a != b)
        )

import unittest

import jax
import jax.numpy as jnp

from envs.schema import PRNGKey
from envs.target_mpe_env import TargetMPEEnvironment, MPEState


def get_direction_vector(vec1, vec2):
    dist_traveled = vec2 - vec1
    dist_traveled /= jnp.linalg.norm(dist_traveled, axis=1)[:, None]
    return dist_traveled


class TargetMPEEnvTest(unittest.TestCase):
    @staticmethod
    def set_up():
        # Parameters + random keys
        max_steps = 10
        key = jax.random.PRNGKey(0)
        key, key_r = jax.random.split(key, 2)

        num_agents = 3

        env = TargetMPEEnvironment(num_agents=num_agents)
        observation, graph, state = env.reset(key_r)

        return env, graph, state, max_steps, key

    @staticmethod
    def get_init_state(key: PRNGKey, env):
        key_agent, key_landmark = jax.random.split(key)

        assert env.num_agents == 3, "This test only works for 3 agents"

        entity_positions = jnp.concatenate(
            [
                jnp.asarray(
                    [
                        [0.0, 0.0],
                        [0.0, 200.0],
                        [200.0, 0.0],
                    ]
                ),
                jnp.asarray(
                    [
                        [1000.0, 1000.0],
                        [1000.0, 1200.0],
                        [1200.0, 1000.0],
                    ]
                ),
            ]
        )

        return MPEState(
            entity_positions=entity_positions,  # type: ignore
            entity_velocities=jnp.zeros((env.num_entities, env.position_dim)),  # type: ignore
            dones=jnp.full(env.num_agents, False),  # type: ignore
            step=0,  # type: ignore
        )

    # def test_target_mpe_rewards(self):
    #     """
    #     Test that the target mpe rewards is the negative of squared distance between agent and landmark.
    #     """
    #
    #     env, state, max_steps, key = TargetMPEEnvTest.set_up()
    #     state = TargetMPEEnvTest.get_init_state(key, env)
    #
    #     for _ in range(max_steps):
    #         key, key_act = jax.random.split(key)
    #         key_act = jax.random.split(key_act, env.num_agents)
    #         actions = {
    #             agent_label: env.action_space_for_agent(agent_label).sample(key_act[i])
    #             for i, agent_label in enumerate(env.agent_labels)
    #         }
    #
    #         obs, state, rew, dones, _ = env.step(key, state, actions)
    #         rewards = env.reward(state)
    #         for agent_label in env.agent_labels:
    #             agent_idx = env.agent_labels_to_index[agent_label]
    #             the_landmark_idx__the_agent_is_supposed_to_go = (
    #                 env.num_agents + agent_idx
    #             )
    #
    #             landmark_coord = state.entity_positions[
    #                 the_landmark_idx__the_agent_is_supposed_to_go
    #             ]
    #             agent_coord = state.entity_positions[agent_idx]
    #
    #             distance_between_landmark_and_agent = jnp.sum(
    #                 jnp.square(agent_coord - landmark_coord)
    #             )
    #
    #             self.assertTrue(
    #                 rewards[agent_label], -distance_between_landmark_and_agent
    #             )

    def test_target_mpe_do_nothing_action(self):
        """
        Test that the target mpe do nothing discrete action works.
        """
        with jax.disable_jit(True):
            env, graph, state, max_steps, key = TargetMPEEnvTest.set_up()
            state = TargetMPEEnvTest.get_init_state(key, env)

            graph = env.get_graph(state)

            prev_state = state

            for _ in range(max_steps):
                key, key_env = jax.random.split(key)
                action = {
                    agent_label: 0 for i, agent_label in enumerate(env.agent_labels)
                }

                obs, _, state, rew, dones, _ = env.step(key_env, state, action)
                self.assertTrue(
                    jnp.array_equal(
                        prev_state.entity_positions, state.entity_positions
                    ),
                    f"{prev_state.entity_positions} != {state.entity_positions}",
                )

                prev_state = state

    def test_target_mpe_left_action(self):
        """
        Test that the target mpe left discrete action works.
        """

        env, graph, state, max_steps, key = TargetMPEEnvTest.set_up()
        init_state = TargetMPEEnvTest.get_init_state(key, env)
        state = init_state

        for _ in range(max_steps):
            key, key_env = jax.random.split(key)
            action = {agent_label: 1 for i, agent_label in enumerate(env.agent_labels)}

            obs, _, state, rew, dones, _ = env.step(key_env, state, action)

        init_state_agent_positions = init_state.entity_positions[: env.num_agents]
        state_agent_positions = state.entity_positions[: env.num_agents]
        dist_travelled = get_direction_vector(
            init_state_agent_positions, state_agent_positions
        )

        self.assertTrue(
            jnp.all(dist_travelled == jnp.asarray([-1.0, 0.0])[None]),
        )

    def test_target_mpe_right_action(self):
        """
        Test that the target mpe right discrete action works.
        """

        env, graph, state, max_steps, key = TargetMPEEnvTest.set_up()
        init_state = TargetMPEEnvTest.get_init_state(key, env)
        state = init_state

        for _ in range(max_steps):
            key, key_env = jax.random.split(key)
            action = {agent_label: 2 for i, agent_label in enumerate(env.agent_labels)}

            obs, _, state, rew, dones, _ = env.step(key_env, state, action)

        init_state_agent_positions = init_state.entity_positions[: env.num_agents]
        state_agent_positions = state.entity_positions[: env.num_agents]
        dist_travelled = get_direction_vector(
            init_state_agent_positions, state_agent_positions
        )

        self.assertTrue(
            jnp.all(dist_travelled == jnp.asarray([1.0, 0.0])[None]),
        )

    def test_target_mpe_down_action(self):
        """
        Test that the target mpe down discrete action works.
        """

        env, graph, state, max_steps, key = TargetMPEEnvTest.set_up()
        init_state = TargetMPEEnvTest.get_init_state(key, env)
        state = init_state

        for _ in range(max_steps):
            key, key_env = jax.random.split(key)
            action = {agent_label: 3 for i, agent_label in enumerate(env.agent_labels)}

            obs, _, state, rew, dones, _ = env.step(key_env, state, action)

        init_state_agent_positions = init_state.entity_positions[: env.num_agents]
        state_agent_positions = state.entity_positions[: env.num_agents]
        dist_travelled = get_direction_vector(
            init_state_agent_positions, state_agent_positions
        )

        self.assertTrue(
            jnp.all(dist_travelled == jnp.asarray([0.0, -1.0])[None]),
        )

    def test_target_mpe_up_action(self):
        """
        Test that the target mpe up discrete action works.
        """

        env, graph, state, max_steps, key = TargetMPEEnvTest.set_up()
        init_state = TargetMPEEnvTest.get_init_state(key, env)
        state = init_state

        for _ in range(max_steps):
            key, key_env = jax.random.split(key)
            action = {agent_label: 4 for i, agent_label in enumerate(env.agent_labels)}

            obs, _, state, rew, dones, _ = env.step(key_env, state, action)

        init_state_agent_positions = init_state.entity_positions[: env.num_agents]
        state_agent_positions = state.entity_positions[: env.num_agents]
        dist_travelled = get_direction_vector(
            init_state_agent_positions, state_agent_positions
        )

        self.assertTrue(
            jnp.all(dist_travelled == jnp.asarray([0.0, 1.0])[None]),
        )


if __name__ == "__main__":
    unittest.main()

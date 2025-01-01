"""
Built off JaxMARL wrapper/baselines.py
"""

from functools import partial

import jax
import jax.numpy as jnp
from flax import struct

from envs.multiagent_env import MultiAgentEnv
from envs.schema import (
    MultiAgentState,
    PRNGKey,
    MultiAgentObservation,
    MultiAgentAction,
)


class MARLWrapper(MultiAgentEnv):

    def __init__(self, env: MultiAgentEnv):
        super().__init__(
            num_agents=env.num_agents,
            max_steps=env.max_steps,
            action_spaces=env.action_spaces,
            observation_spaces=env.observation_spaces,
            agent_labels=env.agent_labels,
        )
        self._env = env

    def get_observations(self, state: MultiAgentState):
        return self._env.get_observations(state)

    def reset(self, key: PRNGKey) -> tuple[MultiAgentObservation, MultiAgentState]:
        return self._env.reset(key)

    def _step(self, key: PRNGKey, state: MultiAgentState, actions: MultiAgentAction):
        return self._env._step(key, state, actions)

    def _batchify_floats(self, x: dict):
        return jnp.stack([x[a] for a in self._env.agent_labels])

    def world_state_size(self):
        spaces = [
            self._env.observation_space_for_agent(agent)
            for agent in self._env.agent_labels
        ]
        return sum([space.shape[-1] for space in spaces])


class MPEWorldStateWrapper(MARLWrapper):

    @partial(jax.jit, static_argnums=0)
    def reset(self, key):
        obs, env_state = self._env.reset(key)
        obs["world_state"] = self.world_state(obs)
        return obs, env_state

    @partial(jax.jit, static_argnums=0)
    def step(self, key, state, action):
        obs, env_state, reward, done, info = self._env.step(key, state, action)
        obs["world_state"] = self.world_state(obs)
        return obs, env_state, reward, done, info

    @partial(jax.jit, static_argnums=0)
    def world_state(self, obs):
        """
        For each agent: [agent obs, all other agent obs]
        """

        @partial(jax.vmap, in_axes=(0, None))
        def _roll_obs(agent_idx, all_obs):
            robs = jnp.roll(all_obs, -agent_idx, axis=0)
            robs = robs.flatten()
            return robs

        all_obs = jnp.array([obs[agent] for agent in self._env.agent_labels]).flatten()
        all_obs = jnp.expand_dims(all_obs, axis=0).repeat(self._env.num_agents, axis=0)
        return all_obs


@struct.dataclass
class LogEnvState:
    env_state: MultiAgentState
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int


class LogWrapper(MARLWrapper):
    """Log the episode returns and lengths.
    NOTE for now for envs where agents terminate at the same time.
    """

    def __init__(self, env: MultiAgentEnv, replace_info: bool = False):
        super().__init__(env)
        self.replace_info = replace_info

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: PRNGKey) -> tuple[MultiAgentObservation, LogEnvState]:
        obs, env_state = self._env.reset(key)
        state = LogEnvState(
            env_state,
            jnp.zeros((self._env.num_agents,)),
            jnp.zeros((self._env.num_agents,)),
            jnp.zeros((self._env.num_agents,)),
            jnp.zeros((self._env.num_agents,)),
        )
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: PRNGKey,
        state: LogEnvState,
        action: MultiAgentAction,
    ) -> tuple[MultiAgentObservation, LogEnvState, float, bool, dict]:
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action
        )
        ep_done = done["__all__"]
        new_episode_return = state.episode_returns + self._batchify_floats(reward)
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - ep_done),
            episode_lengths=new_episode_length * (1 - ep_done),
            returned_episode_returns=state.returned_episode_returns * (1 - ep_done)
            + new_episode_return * ep_done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - ep_done)
            + new_episode_length * ep_done,
        )
        if self.replace_info:
            info = {}
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["returned_episode"] = jnp.full((self._env.num_agents,), ep_done)
        return obs, state, reward, done, info


class MPELogWrapper(LogWrapper):
    """Times reward signal by number of agents within the environment,
    to match the on-policy codebase."""

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        key: PRNGKey,
        state: LogEnvState,
        action: MultiAgentAction,
    ) -> tuple[MultiAgentObservation, LogEnvState, float, bool, dict]:
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action
        )
        reward_log = jax.tree.map(
            lambda x: x * self._env.num_agents, reward
        )  # As per on-policy codebase
        ep_done = done["__all__"]
        new_episode_return = state.episode_returns + self._batchify_floats(reward_log)
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - ep_done),
            episode_lengths=new_episode_length * (1 - ep_done),
            returned_episode_returns=state.returned_episode_returns * (1 - ep_done)
            + new_episode_return * ep_done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - ep_done)
            + new_episode_length * ep_done,
        )
        if self.replace_info:
            info = {}
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["returned_episode"] = jnp.full((self._env.num_agents,), ep_done)
        return obs, state, reward, done, info

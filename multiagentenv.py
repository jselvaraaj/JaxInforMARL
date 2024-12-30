from abc import abstractmethod, ABC
from functools import partial
from typing import final

import jax
from jax._src.checkify import Error
from jax.experimental import checkify

from schema import (
    PRNGKey,
    MultiAgentObservations,
    MultiAgentState,
    AgentLabel,
    MultiAgentEnvOutput,
    MultiAgentActions,
)


class MultiAgentEnv(ABC):
    def __init__(self, num_agents: int, max_steps: int):
        """
        num_agents (int): maximum number of agents within the environment
        """
        self.num_agents = num_agents
        self.observation_spaces = dict()
        self.action_spaces = dict()
        self.max_steps = max_steps

    @abstractmethod
    def reset(self, key: PRNGKey) -> tuple[MultiAgentObservations, MultiAgentState]:
        """Performs resetting of the environment."""

    @abstractmethod
    def _step(
        self, key: PRNGKey, state: MultiAgentState, actions: MultiAgentActions
    ) -> MultiAgentEnvOutput:
        """Environment-specific step transition."""

    @final
    @partial(jax.jit, static_argnums=(0,))
    @checkify.checkify
    def step(
        self,
        key: PRNGKey,
        state: MultiAgentState,
        actions: MultiAgentActions,
    ) -> (Error, MultiAgentEnvOutput):
        """Performs step transitions in the environment. Do not override this method.
        Override _step instead.
        """
        checkify.check(
            state.step < self.max_steps,
            "Cannot step on done state. Reset the environment first.",
        )

        obs, states, rewards, dones, infos = self._step(key, state, actions)

        return obs, states, rewards, dones, infos

    @abstractmethod
    def get_observations(self, state: MultiAgentState) -> MultiAgentObservations:
        """Returns the observations for the given state."""

    @property
    def name(self) -> str:
        """Environment name."""
        return type(self).__name__

    def observation_space_for_agent(self, agent: AgentLabel):
        """Observation space for a given agent."""
        return self.observation_spaces[agent]

    def action_space_for_agent(self, agent: AgentLabel):
        """Action space for a given agent."""
        return self.action_spaces[agent]

from abc import ABC, abstractmethod
from functools import partial

import jax
from jaxtyping import Array, Float

from .schema import (
    AgentIndexAxis,
    AgentLabel,
    EntityIndexAxis,
    EntityLabel,
    MultiAgentAction,
    MultiAgentGraph,
    MultiAgentObservation,
    MultiAgentState,
    PRNGKey,
)
from .spaces import Space


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def entity_labels_to_indices(
        ids: list[EntityLabel], start: int
) -> dict[EntityLabel, int]:
    return {_id: start + i for i, _id in enumerate(ids)}


def is_dictionary_of_spaces_for_entities(
        spaces: dict[EntityLabel, Space], num_entities: int
):
    return (
            all(isinstance(value, Space) for value in spaces.values())
            and len(spaces) == num_entities
    )


class MultiAgentEnv(ABC):
    def __init__(
            self,
            num_agents: int,
            max_steps: int,
            action_spaces: dict[AgentLabel, Space],
            observation_spaces: dict[AgentLabel, Space],
            agent_labels=None,
    ):
        """
        num_agents (int): maximum number of agents within the environment
        """
        self.num_agents = num_agents

        assert action_spaces is None or is_dictionary_of_spaces_for_entities(
            action_spaces, num_agents
        ), "action_spaces must be None or have length num_agents"
        self.action_spaces = action_spaces

        assert observation_spaces is None or is_dictionary_of_spaces_for_entities(
            observation_spaces, num_agents
        ), "observation_spaces must be None or have length num_agents"
        self.observation_spaces = observation_spaces

        self.max_steps = max_steps

        assert agent_labels is None or (
                len(agent_labels) == num_agents
        ), "agent_ids must be None or have length num_agents"
        self.agent_labels = default(
            agent_labels, [f"agent_{i}" for i in range(num_agents)]
        )
        self.agent_labels_to_index = entity_labels_to_indices(
            self.agent_labels, start=0
        )

    @abstractmethod
    def reset(
            self,
            key: PRNGKey,
            initial_agent_communication_message: Float[Array, f"{AgentIndexAxis} ..."],
            initial_entity_position: Float[Array, f"{EntityIndexAxis} ..."],
    ):
        """Performs resetting of the environment."""

    @abstractmethod
    def _step(
            self,
            key: PRNGKey,
            state: MultiAgentState,
            actions: MultiAgentAction,
    ):
        """Environment-specific step transition."""

    @partial(jax.jit, static_argnums=(0,))
    def step(
            self,
            key: PRNGKey,
            state: MultiAgentState,
            actions: MultiAgentAction,
            initial_agent_communication_message: Float[Array, f"{AgentIndexAxis} ..."],
            initial_entity_position: Float[Array, f"{EntityIndexAxis} ..."],
    ):
        """Performs step transitions in the environment. Do not override this method.
        Override _step instead.
        """

        obs, graph, state, reward, done, info = self._step(key, state, actions)
        key, key_reset = jax.random.split(key)

        obs_reset, graph_reset, states_reset = self.reset(
            key_reset, initial_agent_communication_message, initial_entity_position
        )

        # Auto-reset environment based on termination
        state = jax.tree.map(
            lambda x, y: jax.lax.select(done["__all__"], x, y),
            states_reset,
            state,
        )
        obs = jax.tree.map(
            lambda x, y: jax.lax.select(done["__all__"], x, y), obs_reset, obs
        )
        graph = jax.tree.map(
            lambda x, y: jax.lax.select(done["__all__"], x, y), graph_reset, graph
        )

        return obs, graph, state, reward, done, info

    @abstractmethod
    def get_observation(self, state: MultiAgentState) -> MultiAgentObservation:
        """Returns the observations for the given state."""

    @abstractmethod
    def get_graph(self, state: MultiAgentState) -> MultiAgentGraph:
        """Returns the neighborhood graph for the given state."""

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

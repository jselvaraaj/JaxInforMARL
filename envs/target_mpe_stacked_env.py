from typing import NamedTuple

import jax.numpy as jnp
from jaxtyping import Array, Float, Bool

from envs import TargetMPEEnvironment
from .multiagent_env import (
    MultiAgentAction,
    PRNGKey,
)
from .schema import (
    AgentIndexAxis,
    CoordinateAxisIndexAxis,
    EntityIndexAxis,
    MultiAgentGraph,
    MultiAgentObservation,
)
from .target_mpe_env import MPEState


class MPEStateWithBuffer(NamedTuple):
    """Basic MPE State"""
    dones: Bool[Array, AgentIndexAxis]
    step: int
    entity_positions: Float[Array, f"{EntityIndexAxis} {CoordinateAxisIndexAxis}"]
    entity_velocities: Float[Array, f"{EntityIndexAxis} {CoordinateAxisIndexAxis}"]
    did_agent_die_this_time_step: Float[Array, f"{AgentIndexAxis}"]
    agent_communication_message: Float[Array, f"{AgentIndexAxis} ..."] | None
    agent_visibility_radius: Float[Array, f"{AgentIndexAxis}"]
    obs_buffer: MultiAgentObservation
    nodes_buffer: MultiAgentObservation


def stack_arrays(latest_array, buffer):
    latest_array = latest_array[None]
    buffer = jnp.concatenate([latest_array, buffer[:-1]], axis=0)  # Shift and append
    return buffer


class StackedTargetMPEEnvironment(TargetMPEEnvironment):
    def __init__(self, agent_previous_obs_stack_size, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stack_size = agent_previous_obs_stack_size

    def reset(self, *args, **kwargs) -> tuple[MultiAgentObservation, MultiAgentGraph, MPEStateWithBuffer]:
        multi_agent_obs, multi_agent_graph, mpe_state = super().reset(*args, **kwargs)

        obs_buffer = {}
        nodes_buffer = {}
        for agent in self.agent_labels:
            obs_buffer[agent] = jnp.zeros((self.stack_size, self.observation_space_for_agent(agent).shape[-1]))
            nodes_buffer[agent] = jnp.zeros((self.stack_size,) + multi_agent_graph[agent].nodes.shape)

            obs_buffer[agent] = stack_arrays(multi_agent_obs[agent],
                                             obs_buffer[agent])
            multi_agent_obs[agent] = obs_buffer[agent]

            nodes_buffer[agent] = stack_arrays(multi_agent_graph[agent].nodes,
                                               nodes_buffer[agent])
            _nodes = nodes_buffer[agent].swapaxes(0, 1)
            multi_agent_graph[agent] = multi_agent_graph[agent]._replace(nodes=_nodes)

        mpe_state = MPEStateWithBuffer(*mpe_state, obs_buffer=obs_buffer, nodes_buffer=nodes_buffer)

        return multi_agent_obs, multi_agent_graph, mpe_state

    def _step(self,
              key: PRNGKey,
              state: MPEStateWithBuffer,
              actions: MultiAgentAction, ):

        state_with_buffer = state
        state_without_buffer = MPEState(*state[:-2])
        multi_agent_obs, multi_agent_graph, mpe_state, reward, done, info = super()._step(key, state_without_buffer,
                                                                                          actions)
        obs_buffer = state_with_buffer.obs_buffer
        nodes_buffer = state_with_buffer.nodes_buffer
        for agent in self.agent_labels:
            obs_buffer[agent] = stack_arrays(multi_agent_obs[agent],
                                             obs_buffer[agent])
            multi_agent_obs[agent] = obs_buffer[agent]

            nodes_buffer[agent] = stack_arrays(multi_agent_graph[agent].nodes,
                                               nodes_buffer[agent])

            _nodes = nodes_buffer[agent].swapaxes(0, 1)

            multi_agent_graph[agent] = multi_agent_graph[agent]._replace(nodes=_nodes)

        mpe_state = MPEStateWithBuffer(*mpe_state, obs_buffer=obs_buffer, nodes_buffer=nodes_buffer)

        return multi_agent_obs, multi_agent_graph, mpe_state, reward, done, info

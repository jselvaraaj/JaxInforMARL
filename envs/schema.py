from typing import NamedTuple, TypeAlias

from beartype import beartype as typechecker
from jaxtyping import Array, Bool, jaxtyped


class GraphsTupleWithAgentIndex(NamedTuple):
    equivariant_nodes: Array | None
    non_equivariant_nodes: Array | None
    edges: Array | None
    receivers: Array | None
    senders: Array | None
    globals: Array | None
    n_node: Array
    n_edge: Array
    agent_indices: Array | None


# Used in python datastructures
EntityLabel: TypeAlias = str
AgentLabel: TypeAlias = EntityLabel
MultiAgentObservation: TypeAlias = dict[AgentLabel, Array]
MultiAgentAction: TypeAlias = dict[AgentLabel, int]
MultiAgentReward: TypeAlias = dict[AgentLabel, float]
MultiAgentGraph: TypeAlias = dict[AgentLabel, GraphsTupleWithAgentIndex]
MultiAgentDone: TypeAlias = dict[AgentLabel, Bool]
Info: TypeAlias = dict

EntityIndex = int

# Used in JAX Arrays
EntityIndexAxis = "entity_index"
AgentIndexAxis = "agent_index"
CoordinateAxisIndexAxis = "position_index"


@jaxtyped(typechecker=typechecker)
class MultiAgentState(NamedTuple):
    dones: Bool[Array, AgentIndexAxis]
    step: int


PRNGKey: TypeAlias = Array
RGB: TypeAlias = tuple[int, int, int]

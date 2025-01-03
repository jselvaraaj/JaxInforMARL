from typing import TypeAlias

from beartype import beartype as typechecker
from flax import struct
from jaxtyping import Array, Bool, jaxtyped
from jraph import GraphsTuple

# Used in python datastructures
EntityLabel: TypeAlias = str
AgentLabel: TypeAlias = EntityLabel
MultiAgentObservation: TypeAlias = dict[AgentLabel, Array]
MultiAgentAction: TypeAlias = dict[AgentLabel, int]
MultiAgentReward: TypeAlias = dict[AgentLabel, float]
MultiAgentGraph: TypeAlias = GraphsTuple
MultiAgentDone: TypeAlias = dict[AgentLabel, Bool]
Info: TypeAlias = dict


# Used in JAX Arrays
EntityIndex = "entity_index"
AgentIndex = "agent_index"
CoordinateAxisIndex = "position_index"


@jaxtyped(typechecker=typechecker)
@struct.dataclass
class MultiAgentState:
    dones: Bool[Array, AgentIndex]
    step: int


PRNGKey: TypeAlias = Array
RGB: TypeAlias = tuple[int, int, int]

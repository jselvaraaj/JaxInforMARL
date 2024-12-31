from typing import TypeAlias

from flax import struct
from jaxtyping import Array, Bool

# Used in python datastructures
EntityLabel: TypeAlias = str
AgentLabel: TypeAlias = EntityLabel
MultiAgentObservations: TypeAlias = dict[AgentLabel, Array]
MultiAgentActions: TypeAlias = dict[AgentLabel, int]
MultiAgentRewards: TypeAlias = dict[AgentLabel, float]
MultiAgentDones: TypeAlias = dict[AgentLabel, Bool]
Infos: TypeAlias = dict

# Used in JAX Arrays
EntityIndex = "entity_index"
AgentIndex = "agent_index"
CoordinateAxisIndex = "position_index"


@struct.dataclass
class MultiAgentState:
    dones: Bool[Array, AgentIndex]
    step: int


MultiAgentEnvOutput: TypeAlias = tuple[
    MultiAgentObservations,
    MultiAgentState,
    MultiAgentRewards,
    MultiAgentDones,
    Infos,
]

PRNGKey: TypeAlias = Array
RGB: TypeAlias = tuple[int, int, int]

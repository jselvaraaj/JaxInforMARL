from flax import struct
from jaxmarl.environments.mpe.default_params import DISCRETE_ACT
from jaxmarl.environments.mpe.simple import SimpleMPE
from jaxtyping import Array, Float, Bool


@struct.dataclass
class State:
    """Basic MPE State"""

    p_pos: Float[Array, "num_entities x y"]
    p_vel: Float[Array, "num_entities x y"]
    c: Float[Array, "num_entities dim_c"]
    done: Bool[Array, "num_agents"]
    step: int
    goal: int | None = None


class TargetMPEEnvironment(SimpleMPE):
    def __init__(self, num_agents=3, local_ratio=0.5):
        super().__init__(num_agents=num_agents, discrete_action=DISCRETE_ACT)
        self.local_ratio = local_ratio
        assert 0.0 <= self.local_ratio <= 1.0, "local_ratio must be between 0.0 and 1.0"

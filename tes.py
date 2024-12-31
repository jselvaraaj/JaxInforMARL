import jax.numpy as jnp
from beartype import beartype as typechecker
from flax.struct import dataclass
from jaxtyping import jaxtyped, Float, Array


@jaxtyped(typechecker=typechecker)
@dataclass
class MyDataclass:
    x: int
    y: Float[Array, "2"]


@jaxtyped(typechecker=typechecker)
def fun(data: MyDataclass) -> MyDataclass:
    return data


x = MyDataclass(x="hello", y=jnp.array([1.0, 2.0]))

fun(x)

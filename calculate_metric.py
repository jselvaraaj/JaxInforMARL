from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import Int, Array

from envs import TargetMPEEnvironment
from envs.target_mpe_env import MPEState


@partial(jax.jit, static_argnums=(0,))
def get_stats_for_state(env: TargetMPEEnvironment, state: MPEState):
    @partial(jax.vmap, in_axes=(0, None, None))
    def _collisions(agent_idx: Int[Array, "..."], other_idx: Int[Array, "..."], state):
        return jax.vmap(env.is_collision, in_axes=(None, 0, None))(
            agent_idx,
            other_idx,
            state,  # type: ignore
        )

    is_agent_dead = jax.vmap(env.is_there_overlap, in_axes=(0, 0, None))(
        env.agent_indices, env.landmark_indices, state
    )
    num_collisions = (
            jnp.sum(
                _collisions(env.agent_indices, env.agent_indices, state)
                & ~is_agent_dead[..., None]
            )
            / 2
    )

    num_agent_died = jnp.sum(
        jax.vmap(env.is_there_overlap, in_axes=(0, 0, None))(
            env.agent_indices, env.landmark_indices, state
        )
    )
    return num_collisions, num_agent_died

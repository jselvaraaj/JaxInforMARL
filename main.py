import os

import jax
import jax.numpy as jnp
import orbax

from algorithm.marl_ppo import make_env_from_config
from config.mappo_config import MAPPOConfig
from model.actor_critic_rnn import ActorRNN, ScannedRNN

config: MAPPOConfig = MAPPOConfig.create()
env = make_env_from_config(config)
rng = jax.random.PRNGKey(config.training_config.seed)

rng, _rng_actor = jax.random.split(rng, 2)

num_envs = config.training_config.num_envs
num_actions = env.action_space_for_agent(env.agent_labels[0]).n
obs_dim = env.observation_space_for_agent(env.agent_labels[0]).shape[0]
actor_network = ActorRNN(num_actions, config=config)
actor_init_x = (
    jnp.zeros(
        (
            1,
            num_envs,
            obs_dim,
        )
    ),
    jnp.zeros((1, num_envs)),
)
actor_init_hidden_state = ScannedRNN.initialize_carry(
    num_envs, config.network.gru_hidden_dim
)

actor_network_params = actor_network.init(
    _rng_actor, actor_init_hidden_state, actor_init_x
)

running_script_path = os.path.abspath(".")
checkpoint_dir = os.path.join(running_script_path, "algorithm/PPO_Runner_Checkpoint")

sharding = jax.sharding.NamedSharding(
    jax.sharding.Mesh(jax.devices(), ("model",)),
    jax.sharding.PartitionSpec(
        "model",
    ),
)

abstract_actor_params = jax.tree_util.tree_map(
    orbax.checkpoint.utils.to_shape_dtype_struct, actor_network_params
)

abstract_state = {
    "actor_train_params": abstract_actor_params,
}

ckptr = orbax.checkpoint.AsyncCheckpointer(orbax.checkpoint.StandardCheckpointHandler())

raw_restored = ckptr.restore(
    checkpoint_dir,
    args=orbax.checkpoint.args.StandardRestore(abstract_state, strict=False),
)

print()

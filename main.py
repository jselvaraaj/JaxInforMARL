import os

import jax
import jax.numpy as jnp
import orbax

from algorithm.marl_ppo import make_env_from_config
from config.mappo_config import MAPPOConfig
from envs import TargetMPEEnvironment
from envs.mpe_visualizer import MPEVisualizer
from model.actor_critic_rnn import ActorRNN, ScannedRNN


def get_restored_actor():
    config: MAPPOConfig = MAPPOConfig.create()
    env = make_env_from_config(config)
    rng = jax.random.PRNGKey(config.training_config.seed)

    rng, _rng_actor = jax.random.split(rng, 2)

    num_agents = config.env_config.kwargs.num_agents
    num_actions = env.action_space_for_agent(env.agent_labels[0]).n
    obs_dim = env.observation_space_for_agent(env.agent_labels[0]).shape[0]
    actor_network = ActorRNN(num_actions, config=config)
    actor_init_x = (
        jnp.zeros(
            (
                1,
                num_agents,
                obs_dim,
            )
        ),
        jnp.zeros((1, num_agents)),
    )
    actor_init_hidden_state = ScannedRNN.initialize_carry(
        num_agents, config.network.gru_hidden_dim
    )

    actor_network_params = actor_network.init(
        _rng_actor, actor_init_hidden_state, actor_init_x
    )

    running_script_path = os.path.abspath(".")
    checkpoint_dir = os.path.join(
        running_script_path, "artifacts/PPO_RNN_Runner_State:v1"
    )

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

    ckptr = orbax.checkpoint.AsyncCheckpointer(
        orbax.checkpoint.StandardCheckpointHandler()
    )

    raw_restored = ckptr.restore(
        checkpoint_dir,
        args=orbax.checkpoint.args.StandardRestore(abstract_state, strict=False),
    )

    restored_actor_params = raw_restored["actor_train_params"]

    return config, actor_network, restored_actor_params, actor_init_hidden_state


if __name__ == "__main__":
    config, actor, restored_params, actor_init_hidden_state = get_restored_actor()

    max_steps = config.env_config.kwargs.max_steps
    key = jax.random.PRNGKey(0)
    key, key_r = jax.random.split(key, 2)

    num_agents = config.env_config.kwargs.num_agents

    env = TargetMPEEnvironment(num_agents=num_agents)
    obs, state = env.reset(key_r)

    hidden_state = actor_init_hidden_state
    state_seq = []
    for _ in range(max_steps):
        key, key_env, key_actor = jax.random.split(key, 3)

        state_seq.append(state)

        obs = jnp.stack(list(obs.values()))[None]
        # dones = jnp.repeat(state.dones[None], num_agents, axis=0)
        dones = state.dones[None, ...]
        x = (obs, dones)
        hidden_state, pi = actor.apply(restored_params, hidden_state, x)

        action = pi.sample(seed=key_actor).squeeze()

        action = {
            agent_label: action[env.agent_labels_to_index[agent_label]].item()
            for agent_label in env.agent_labels
        }

        obs, state, rew, _, _ = env.step(key_env, state, action)

    viz = MPEVisualizer(env, state_seq)

    viz.animate(view=True)

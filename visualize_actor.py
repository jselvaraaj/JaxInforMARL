import os
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
import orbax
import wandb
from flax.training.train_state import TrainState

from algorithm.marl_ppo import (
    make_env_from_config,
    get_actor_init_input,
    get_init_communication_message,
    _env_step,
    StaticVariables,
    get_critic_init_input,
    ActorAndCriticTrainStates,
    ActorAndCriticHiddenStates,
    EnvStepRunnerState,
)
from config.mappo_config import MAPPOConfig
from envs.mpe_visualizer import MPEVisualizer
from model.actor_critic_rnn import GraphAttentionActorRNN, CriticRNN


def get_restored_actor(artifact_name):
    config: MAPPOConfig = MAPPOConfig.create()
    env = make_env_from_config(config)
    rng = jax.random.PRNGKey(config.training_config.seed)

    rng, _rng_actor, _rng_critic = jax.random.split(rng, 3)

    num_actions = env.action_space_for_agent(env.agent_labels[0]).n

    actor_network = GraphAttentionActorRNN(num_actions, config=config)
    critic_network = CriticRNN(config=config)

    ac_init_x, ac_init_h_state, graph_init = get_actor_init_input(config, env)

    initial_communication_message, initial_communication_message_env_input = (
        get_init_communication_message(config, env, ac_init_h_state)
    )

    actor_network_params = actor_network.init(_rng_actor, ac_init_h_state, ac_init_x)

    cr_init_x, cr_init_h_state = get_critic_init_input(config, env, graph_init)

    critic_network_params = critic_network.init(_rng_critic, cr_init_h_state, cr_init_x)

    running_script_path = os.path.abspath(".")
    checkpoint_dir = os.path.join(running_script_path, artifact_name)

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

    ck_ptr = orbax.checkpoint.AsyncCheckpointer(
        orbax.checkpoint.StandardCheckpointHandler()
    )

    raw_restored = ck_ptr.restore(
        checkpoint_dir,
        args=orbax.checkpoint.args.StandardRestore(abstract_state, strict=False),
    )

    restored_actor_params = raw_restored["actor_train_params"]

    return (
        config,
        actor_network,
        restored_actor_params,
        critic_network,
        critic_network_params,
        ac_init_h_state,
        cr_init_h_state,
        env,
        initial_communication_message_env_input,
        initial_communication_message,
        rng,
    )


if __name__ == "__main__":

    artifact_version = "201"
    artifact_name = f"artifacts/PPO_RNN_Runner_State:v{artifact_version}"
    if not Path(artifact_name).is_dir():
        artifact_remote_name = (
            f"josssdan/JaxInforMARL/PPO_RNN_Runner_State:v{artifact_version}"
        )

        api = wandb.Api()
        artifact = api.artifact(artifact_remote_name, type="model")
        artifact_dir = artifact.download()

    (
        config,
        actor_network,
        actor_restored_params,
        critic_network,
        critic_network_params,
        ac_init_h_state,
        cr_init_h_state,
        env,
        initial_communication_message_env_input,
        initial_communication_message,
        key,
    ) = get_restored_actor(artifact_name)

    max_steps = config.env_config.kwargs.max_steps
    num_env = config.training_config.num_envs

    key, key_r = jax.random.split(key, 2)

    env_key = jax.random.split(key_r, num_env)

    obs_v, graph_v, env_state = jax.vmap(
        env.reset,
        in_axes=(
            0,
            0 if initial_communication_message_env_input.size != 0 else None,
        ),
    )(env_key, initial_communication_message_env_input)

    key, _rng = jax.random.split(key, 2)

    max_grad_norm = config.training_config.ppo_config.max_grad_norm
    lr = config.training_config.lr

    actor_tx = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(lr, eps=1e-5),
    )
    critic_tx = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(lr, eps=1e-5),
    )

    actor_train_state = TrainState.create(
        apply_fn=actor_network.apply,
        params=actor_restored_params,
        tx=actor_tx,
    )
    critic_train_state = TrainState.create(
        apply_fn=critic_network.apply,
        params=critic_network_params,
        tx=critic_tx,
    )

    actor_critic_train_states = ActorAndCriticTrainStates(
        actor_train_state, critic_train_state
    )
    actor_critic_hidden_state = ActorAndCriticHiddenStates(
        ac_init_h_state, cr_init_h_state
    )
    env_step_runner_state = EnvStepRunnerState(
        actor_critic_train_states,
        env_state,
        obs_v,
        graph_v,
        jnp.zeros(config.derived_values.num_actors, dtype=bool),
        actor_critic_hidden_state,
        initial_communication_message,
        _rng,
    )

    store_env_state = True
    _env_step_with_static_args = partial(
        _env_step,
        StaticVariables(
            env,
            config,
            actor_network,
            ac_init_h_state,
            critic_network,
            initial_communication_message_env_input,
            store_env_state,
        ),
    )
    runner_state, traj_batch = jax.lax.scan(
        _env_step_with_static_args, env_step_runner_state, None, max_steps
    )
    viz = MPEVisualizer(env._env._env, traj_batch.env_state.env_state, config)

    viz.animate(view=True)

import functools
from dataclasses import asdict
from functools import partial
from typing import Sequence, NamedTuple, TypeAlias, cast

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from beartype.door import die_if_unbearable
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from jaxtyping import Int, Array, Float, Bool

import envs
from config.mappo_rnn_config import Config
from envs.schema import (
    PRNGKey,
    AgentIndex,
    MultiAgentObservations,
    MultiAgentState,
    MultiAgentActions,
    AgentLabel,
)
from envs.spaces import Discrete
from wrapper import MARLEnvWrapper

RunnerState: TypeAlias = tuple[
    tuple[TrainState, TrainState],
    MultiAgentState,
    MultiAgentObservations,
    Bool[Array, "..."],
    tuple[Float[Array, "..."], Float[Array, "..."]],
    PRNGKey,
]
UpdateRunnerState: TypeAlias = tuple[
    RunnerState,
    int,
]


class MPEAddWorldStateToObsWrapper(MARLEnvWrapper):
    @property
    def name(self):
        return self._env.name

    @partial(jax.jit, static_argnums=0)
    def reset(self, key: PRNGKey):
        obs, env_state = self._env.reset(key)
        obs["world_state"] = self.world_state(obs)
        return obs, env_state

    @partial(jax.jit, static_argnums=0)
    def step(
        self,
        key: PRNGKey,
        state: MultiAgentState,
        actions: MultiAgentActions,
    ):
        obs, env_state, reward, done, info = self._env.step(key, state, actions)
        obs["world_state"] = self.world_state(obs)
        return obs, env_state, reward, done, info

    @partial(jax.jit, static_argnums=0)
    def world_state(
        self, obs: MultiAgentObservations
    ) -> Float[Array, f"{AgentIndex} ..."]:
        """
        For each agent: [agent obs, all other agent obs]
        """

        @partial(jax.vmap, in_axes=(0, None))
        def _roll_obs(agent_idx: Int[Array, f"{AgentIndex}"], all_obs):
            robs = jnp.roll(all_obs, -agent_idx, axis=0)
            robs = robs.flatten()
            return robs

        all_obs = jnp.array(
            [obs[agent_label] for agent_label in self._env.agent_labels]
        ).flatten()
        all_obs = jnp.expand_dims(all_obs, axis=0).repeat(self._env.num_agents, axis=0)
        return all_obs

    def world_state_size(self):
        spaces = [
            self._env.observation_space_for_agent(agent_label)
            for agent_label in self._env.agent_labels
        ]
        return sum([space.shape[-1] for space in spaces])


class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, jnp.newaxis],
            self.initialize_carry(*rnn_state.shape),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size) -> Float[Array, "..."]:
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class BaseRNN(nn.Module):
    config: Config.NetworkConfig

    @nn.compact
    def __call__(
        self, hidden: Float[Array, "..."], x: Float[Array, "..."]
    ) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
        obs, dones = x
        embedding = nn.Dense(
            self.config.fc_dim_size,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        latent_mean = nn.Dense(
            self.config.gru_hidden_dim,
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        latent_mean = nn.relu(latent_mean)

        return hidden, latent_mean


class ActorRNN(nn.Module):
    action_dim: Sequence[int]
    config: Config.NetworkConfig

    @nn.compact
    def __call__(
        self, hidden: Float[Array, "..."], x: Float[Array, "..."]
    ) -> tuple[Float[Array, "..."], distrax.Categorical]:

        base_rnn = BaseRNN(self.config)
        hidden, latent_features = base_rnn(hidden, x)

        action_logits = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(latent_features)

        pi = distrax.Categorical(logits=action_logits)

        return hidden, pi


class CriticRNN(nn.Module):
    config: Config.NetworkConfig

    @nn.compact
    def __call__(
        self, hidden: Float[Array, "..."], x: Float[Array, "..."]
    ) -> tuple[Float[Array, "..."], Float[Array, "..."]]:

        base_rnn = BaseRNN(self.config)
        hidden, latent_features = base_rnn(hidden, x)

        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            latent_features
        )

        return hidden, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    global_done: Bool[Array, "..."]
    done: Bool[Array, "..."]
    action: Int[Array, "..."]
    value: Float[Array, "..."]
    reward: Float[Array, "..."]
    log_prob: Float[Array, "..."]
    obs: Float[Array, "..."]
    world_state: Float[Array, "..."]
    info: Float[Array, "..."]


def batchify(
    x: dict[AgentLabel, Array], agent_labels: list[AgentLabel], num_actors: int
):
    x = jnp.stack([x[a] for a in agent_labels])
    return x.reshape((num_actors, -1))


def unbatchify(
    x: Array, agent_labels: list[AgentLabel], num_envs: int, num_actors: int
):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_labels)}


def make_train(config: Config):

    # Make MARL env from config
    env_class_name = config.env_config.cls_name
    env_class = getattr(envs, env_class_name)
    env_kwargs = asdict(config.env_config.kwargs)
    env = MPEAddWorldStateToObsWrapper(env_class(**env_kwargs))

    ppo_config = config.training_config.ppo_config

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (ppo_config.num_minibatches * ppo_config.update_epochs))
            / config.derived_values.num_updates_per_env
        )
        return config.training_config.lr * frac

    def train(rng: PRNGKey):
        # Currently only supports discrete action spaces
        die_if_unbearable(env.action_spaces, dict[AgentLabel, Discrete])

        train_config = config.training_config
        action_dim = env.action_space_for_agent(env.agent_labels[0]).n
        # INIT NETWORK
        actor_network = ActorRNN(action_dim, config=config.network)
        critic_network = CriticRNN(config=config.network)
        rng, _rng_actor, _rng_critic = jax.random.split(rng, 3)

        obs_dim = env.observation_space_for_agent(env.agent_labels[0]).shape[0]
        ac_init_x = (
            jnp.zeros(
                (
                    1,
                    train_config.num_envs,
                    obs_dim,
                )
            ),
            jnp.zeros((1, train_config.num_envs)),
        )
        actor_init_hidden_state = ScannedRNN.initialize_carry(
            train_config.num_envs, config.network.gru_hidden_dim
        )
        actor_network_params = actor_network.init(
            _rng_actor, actor_init_hidden_state, ac_init_x
        )

        cr_init_x = (
            jnp.zeros(
                (
                    1,
                    train_config.num_envs,
                    env.world_state_size(),
                )
            ),  #  + env.observation_space(env.agents[0]).shape[0]
            jnp.zeros((1, train_config.num_envs)),
        )
        critic_init_hidden_state = ScannedRNN.initialize_carry(
            train_config.num_envs, config.network.gru_hidden_dim
        )
        critic_network_params = critic_network.init(
            _rng_critic, critic_init_hidden_state, cr_init_x
        )

        max_grad_norm = train_config.ppo_config.max_grad_norm
        if train_config.anneal_lr:
            actor_optimizer = optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
            critic_optimizer = optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            actor_optimizer = optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.adam(train_config.lr, eps=1e-5),
            )
            critic_optimizer = optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.adam(train_config.lr, eps=1e-5),
            )
        actor_train_state = TrainState.create(
            apply_fn=actor_network.apply,
            params=actor_network_params,
            tx=actor_optimizer,
        )
        critic_train_state = TrainState.create(
            apply_fn=critic_network.apply,
            params=critic_network_params,
            tx=critic_optimizer,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, train_config.num_envs)
        obs_v, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        actor_init_hidden_state = ScannedRNN.initialize_carry(
            config.derived_values.num_actors, config.network.gru_hidden_dim
        )
        critic_init_hidden_state = ScannedRNN.initialize_carry(
            config.derived_values.num_actors, config.network.gru_hidden_dim
        )

        # TRAIN LOOP
        # noinspection DuplicatedCode
        def _update_step(
            update_runner_state: UpdateRunnerState, unused: int
        ) -> tuple[UpdateRunnerState, Array]:
            # COLLECT TRAJECTORIES
            runner_state, update_steps = update_runner_state

            # noinspection DuplicatedCode
            def _env_step(
                runner_state: RunnerState, unused: int
            ) -> tuple[RunnerState, Transition]:
                train_states, env_state, last_obs, last_done, hidden_states, rng = (
                    runner_state
                )

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                obs_batch = batchify(
                    last_obs, env.agent_labels, config.derived_values.num_actors
                )
                actor_input = (
                    obs_batch[jnp.newaxis, :],
                    last_done[jnp.newaxis, :],
                )
                actor_hidden_state, pi = cast(
                    tuple[Array, distrax.Categorical],
                    actor_network.apply(
                        train_states[0].params, hidden_states[0], actor_input
                    ),
                )
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_actions = unbatchify(
                    action, env.agent_labels, train_config.num_envs, env.num_agents
                )
                # VALUE
                # output of wrapper is (num_envs, num_agents, world_state_size)
                # swap axes to (num_agents, num_envs, world_state_size) before reshaping to (num_actors, world_state_size)
                world_state = last_obs["world_state"].swapaxes(0, 1)
                world_state = world_state.reshape(
                    (config.derived_values.num_actors, -1)
                )
                critic_input = (
                    world_state[None, :],
                    last_done[None, :],
                )
                critic_hidden_state, value = cast(
                    tuple[Float[Array, "..."], Float[Array, "..."]],
                    critic_network.apply(
                        train_states[1].params, hidden_states[1], critic_input
                    ),
                )

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, train_config.num_envs)
                obs_v, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_actions)

                info = jax.tree.map(
                    lambda x: x.reshape(config.derived_values.num_actors), info
                )
                done_batch = batchify(
                    done, env.agent_labels, config.derived_values.num_actors
                ).squeeze()
                transition = Transition(
                    jnp.tile(done["__all__"], env.num_agents),
                    last_done,
                    action.squeeze(),
                    value.squeeze(),
                    batchify(
                        reward, env.agent_labels, config.derived_values.num_actors
                    ).squeeze(),
                    log_prob.squeeze(),
                    obs_batch,
                    world_state,
                    info,
                )
                runner_state = (
                    train_states,
                    env_state,
                    obs_v,
                    done_batch,
                    (actor_hidden_state, critic_hidden_state),
                    rng,
                )
                return runner_state, transition

            initial_hidden_states = runner_state[-2]
            runner_state, traj_batch = jax.lax.scan(
                _env_step,
                runner_state,
                None,
                train_config.ppo_config.num_steps_per_update_per_env,
            )

            # CALCULATE ADVANTAGE
            train_states, env_state, last_obs, last_done, hidden_state, rng = (
                runner_state
            )

            last_world_state = last_obs["world_state"].swapaxes(0, 1)
            last_world_state = last_world_state.reshape(
                (config.derived_values.num_actors, -1)
            )
            critic_input = (
                last_world_state[None, :],
                last_done[None, :],
            )
            _, last_val = cast(
                tuple[Float[Array, "..."], Float[Array, "..."]],
                critic_network.apply(
                    train_states[1].params, hidden_state[1], critic_input
                ),
            )
            last_val = last_val.squeeze()

            def _calculate_gae(traj_batch: Transition, last_val: Float[Array, "..."]):
                def _get_advantages(gae_and_next_value, transition: Transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.global_done,
                        transition.value,
                        transition.reward,
                    )
                    delta = (
                        reward
                        + config.env_config.gamma * next_value * (1 - done)
                        - value
                    )
                    gae = (
                        delta
                        + config.env_config.gamma
                        * config.training_config.ppo_config.gae_lambda
                        * (1 - done)
                        * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minibatch(train_states, batch_info):
                    actor_train_state, critic_train_state = train_states
                    (
                        actor_init_hidden_state,
                        critic_init_hidden_state,
                        traj_batch,
                        advantages,
                        targets,
                    ) = batch_info

                    def _actor_loss_fn(
                        actor_params, init_hidden_state, traj_batch, gae
                    ):
                        # RERUN NETWORK
                        _, pi = cast(
                            tuple[Float[Array, "..."], distrax.Categorical],
                            actor_network.apply(
                                actor_params,
                                init_hidden_state.squeeze(),
                                (traj_batch.obs, traj_batch.done),
                            ),
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE ACTOR LOSS
                        log_ratio = log_prob - traj_batch.log_prob
                        ratio = jnp.exp(log_ratio)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - train_config.ppo_config.clip_eps,
                                1.0 + train_config.ppo_config.clip_eps,
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        # debug
                        approx_kl = ((ratio - 1) - log_ratio).mean()
                        clip_frac = jnp.mean(
                            jnp.abs(ratio - 1) > train_config.ppo_config.clip_eps
                        )

                        actor_loss = (
                            loss_actor
                            - train_config.ppo_config.entropy_coefficient * entropy
                        )
                        return actor_loss, (
                            loss_actor,
                            entropy,
                            ratio,
                            approx_kl,
                            clip_frac,
                        )

                    def _critic_loss_fn(
                        critic_params, init_hidden_state, traj_batch, targets
                    ):
                        # RERUN NETWORK
                        _, value = cast(
                            tuple[Float[Array, "..."], Float[Array, "..."]],
                            critic_network.apply(
                                critic_params,
                                init_hidden_state.squeeze(),
                                (traj_batch.world_state, traj_batch.done),
                            ),
                        )

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(
                            -train_config.ppo_config.clip_eps,
                            train_config.ppo_config.clip_eps,
                        )
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )
                        critic_loss = (
                            config.training_config.ppo_config.value_coefficient
                            * value_loss
                        )
                        return critic_loss, value_loss

                    actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
                    actor_loss, actor_grads = actor_grad_fn(
                        actor_train_state.params,
                        actor_init_hidden_state,
                        traj_batch,
                        advantages,
                    )
                    critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
                    critic_loss, critic_grads = critic_grad_fn(
                        critic_train_state.params,
                        critic_init_hidden_state,
                        traj_batch,
                        targets,
                    )

                    actor_train_state = actor_train_state.apply_gradients(
                        grads=actor_grads
                    )
                    critic_train_state = critic_train_state.apply_gradients(
                        grads=critic_grads
                    )

                    total_loss = actor_loss[0] + critic_loss[0]
                    loss_info = {
                        "total_loss": total_loss,
                        "actor_loss": actor_loss[0],
                        "value_loss": critic_loss[0],
                        "entropy": actor_loss[1][1],
                        "ratio": actor_loss[1][2],
                        "approx_kl": actor_loss[1][3],
                        "clip_frac": actor_loss[1][4],
                    }

                    return (actor_train_state, critic_train_state), loss_info

                (
                    train_states,
                    init_hidden_states,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state
                rng, _rng = jax.random.split(rng)

                init_hidden_states = jax.tree.map(
                    lambda x: jnp.reshape(x, (1, config.derived_values.num_actors, -1)),
                    init_hidden_states,
                )

                batch = (
                    init_hidden_states[0],
                    init_hidden_states[1],
                    traj_batch,
                    advantages.squeeze(),
                    targets.squeeze(),
                )
                permutation = jax.random.permutation(
                    _rng, config.derived_values.num_actors
                )

                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )
                minibatches = jax.tree.map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [
                                x.shape[0],
                                config.training_config.ppo_config.num_minibatches,
                                -1,
                            ]
                            + list(x.shape[2:]),
                        ),
                        1,
                        0,
                    ),
                    shuffled_batch,
                )

                train_states, loss_info = jax.lax.scan(
                    _update_minibatch, train_states, minibatches
                )
                update_state = (
                    train_states,
                    jax.tree.map(lambda x: x.squeeze(), init_hidden_states),
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, loss_info

            update_state = (
                train_states,
                initial_hidden_states,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, train_config.ppo_config.update_epochs
            )
            loss_info["ratio_0"] = loss_info["ratio"].at[0, 0].get()
            loss_info = jax.tree.map(lambda x: x.mean(), loss_info)

            train_states = update_state[0]
            metric = traj_batch.info
            metric["loss"] = loss_info
            metric["returned_episode_returns"] = traj_batch.reward
            rng = update_state[-1]

            def callback(metric):
                print(
                    {
                        "returns": metric["returned_episode_returns"][-1, :].mean(),
                        "env_step": metric["update_steps"]
                        * train_config.num_envs
                        * train_config.ppo_config.num_steps_per_update_per_env,
                        **metric["loss"],
                    }
                )

            metric["update_steps"] = update_steps
            jax.experimental.io_callback(callback, None, metric)
            update_steps += 1
            runner_state = (
                train_states,
                env_state,
                last_obs,
                last_done,
                hidden_state,
                rng,
            )
            return (runner_state, update_steps), metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            (actor_train_state, critic_train_state),
            env_state,
            obs_v,
            jnp.zeros(config.derived_values.num_actors, dtype=bool),
            (actor_init_hidden_state, critic_init_hidden_state),
            _rng,
        )

        # use None when jit is enabled back
        dummy_sequence = jnp.zeros(config.derived_values.num_updates_per_env)

        runner_state, metric = jax.lax.scan(
            _update_step,
            (runner_state, 0),
            dummy_sequence,
            config.derived_values.num_updates_per_env,
        )
        return {"runner_state": runner_state}

    return train


def main():
    config = Config.create()
    rng = jax.random.PRNGKey(config.training_config.seed)
    with jax.disable_jit(True):
        train_jit = jax.jit(make_train(config))
        out = train_jit(rng)


if __name__ == "__main__":
    main()

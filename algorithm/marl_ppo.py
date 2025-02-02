"""
Built off JaxMARL( https://github.com/FLAIROx/JaxMARL) baselines/MAPPO/mappo_rnn_mpe.py
"""

import os
from functools import partial
from typing import Any, NamedTuple, cast

import distrax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax
import wandb
from beartype import beartype
from flax.training import orbax_utils
from flax.training.train_state import TrainState
from jax import block_until_ready
from jaxtyping import Array, Float, jaxtyped

import envs
from config.config_format_conversion import config_to_dict
from config.mappo_config import (
    CommunicationType,
)
from config.mappo_config import (
    MAPPOConfig as MAPPOConfig,
)
from envs.multiagent_env import MultiAgentEnv
from envs.schema import EntityIndexAxis, MultiAgentGraph, MultiAgentObservation, PRNGKey
from envs.target_mpe_env import GraphsTupleWithAgentIndex, LinSpaceConfig
from envs.wrapper import LogEnvState, MPELogWrapper, MPEWorldStateWrapper
from model.actor_critic_rnn import CriticRNN, GraphAttentionActorRNN, ScannedRNN


class Transition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    graph: GraphsTupleWithAgentIndex
    world_state: jnp.ndarray
    info: jnp.ndarray


class TransitionForVisualization(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    graph: GraphsTupleWithAgentIndex
    world_state: jnp.ndarray
    info: jnp.ndarray
    env_state: LogEnvState


class TransitionWithActionField(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    graph: GraphsTupleWithAgentIndex
    world_state: jnp.ndarray
    info: jnp.ndarray
    env_state: LogEnvState
    action_field: jnp.ndarray


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape(
        (num_actors, -1)  # this will concatenate the obs from all previous rolling memory
        # i don't want that but i am not using obs any ways so deferring to fix it.
    )  # [agent_0_env_1, agent_0_env_2 ....agent_n_env_(m-1), agent_n_env_m]


def unbatchify(x: Array, agent_list, num_envs, num_agents):
    x = x.reshape((num_agents, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


@jaxtyped(typechecker=beartype)
def batchify_graph(graph: MultiAgentGraph, agent_label_index: dict[str, int]):
    equivariant_nodes_for_all_agents = []
    non_equivariant_nodes_for_all_agents = []

    edges_for_all_agents = []
    receivers_for_all_agents = []
    senders_for_all_agents = []
    n_node_for_all_agents = []
    agent_indices_for_all_agents = []
    n_edge_for_all_agents = []
    for agent_label in graph:
        equivariant_nodes, non_equivariant_nodes, edges, receivers, senders, _, n_node, n_edge, agent_indices = graph[
            agent_label
        ]
        num_env, *_ = equivariant_nodes.shape
        receivers = receivers.astype(jnp.int32)
        senders = senders.astype(jnp.int32)

        n_node = n_node.flatten()
        n_edge = n_edge.flatten()

        equivariant_nodes_for_all_agents.append(equivariant_nodes)
        non_equivariant_nodes_for_all_agents.append(non_equivariant_nodes)
        edges_for_all_agents.append(edges)
        receivers_for_all_agents.append(receivers)
        senders_for_all_agents.append(senders)
        n_node_for_all_agents.append(n_node)
        agent_indices_for_all_agents.append(agent_indices)
        n_edge_for_all_agents.append(n_edge)

    def _stack_all_agent(x):
        return jnp.stack(x).reshape(-1, *x[0].shape[1:])

    return GraphsTupleWithAgentIndex(
        equivariant_nodes=_stack_all_agent(equivariant_nodes_for_all_agents),
        non_equivariant_nodes=_stack_all_agent(non_equivariant_nodes_for_all_agents),
        n_node=_stack_all_agent(n_node_for_all_agents),
        edges=_stack_all_agent(edges_for_all_agents),
        receivers=_stack_all_agent(receivers_for_all_agents),
        senders=_stack_all_agent(senders_for_all_agents),
        n_edge=_stack_all_agent(n_edge_for_all_agents),
        agent_indices=_stack_all_agent(agent_indices_for_all_agents),
        globals=None,
    )


def make_env_from_config(config: MAPPOConfig):
    env_class_name = config.env_config.env_cls_name
    env_class = getattr(envs, env_class_name)
    env_kwargs = config_to_dict(config.env_config.env_kwargs)
    env = MPEWorldStateWrapper(env_class(**env_kwargs))
    env = MPELogWrapper(env)
    return env


def get_actor_init_input(config: MAPPOConfig, env):
    num_env = config.training_config.num_envs
    num_coordinate = 2
    node_num_equivariant_feature = 2
    if config.env_config.env_kwargs.add_target_goal_to_nodes:
        node_num_equivariant_feature += 1
    communication_type = config.env_config.env_kwargs.agent_communication_type
    agent_previous_obs_stack_size = config.env_config.env_kwargs.agent_previous_obs_stack_size
    node_non_equivariant_feature_dim = 1
    if communication_type == CommunicationType.HIDDEN_STATE.value:
        node_non_equivariant_feature_dim += config.network_config.gru_hidden_dim
    elif (
            communication_type == CommunicationType.PAST_ACTION.value
            or communication_type == CommunicationType.CURRENT_ACTION.value
    ):
        node_non_equivariant_feature_dim += 1
    equivariant_nodes = jnp.zeros(
        (
            num_env,
            env.num_entities,
            agent_previous_obs_stack_size,
            node_num_equivariant_feature,
            num_coordinate
        )
    )
    non_equivariant_nodes = jnp.zeros(
        (
            num_env,
            env.num_entities,
            agent_previous_obs_stack_size,
            node_non_equivariant_feature_dim,
        )
    )

    edges = jnp.arange(num_env * 2 * env.num_agents).reshape(
        num_env,
        2 * env.num_agents,
        1,
    )
    sender_receiver_shape = (
        num_env,
        2 * env.num_agents,
    )
    receivers = jnp.broadcast_to(
        jnp.concatenate(
            [
                jnp.arange(env.num_agents),
                jnp.arange(env.num_agents),
            ]
        ),
        sender_receiver_shape,
    )
    senders = jnp.broadcast_to(
        jnp.concatenate(
            [
                jnp.arange(env.num_agents),
                jnp.flip(jnp.arange(env.num_agents)),
            ]
        ),
        sender_receiver_shape,
    )
    n_node = jnp.array(num_env * [env.num_entities])
    n_edge = jnp.array(num_env * [2 * env.num_agents])
    agent_indices = jnp.full((num_env,), 0)
    graph_init = batchify_graph(
        {
            agent_label: GraphsTupleWithAgentIndex(
                equivariant_nodes=equivariant_nodes,
                non_equivariant_nodes=non_equivariant_nodes,
                edges=edges,
                globals=None,
                receivers=receivers,
                senders=senders,
                n_node=n_node,
                n_edge=n_edge,
                agent_indices=agent_indices,
            )
            for agent_label in env.agent_labels
        },
        env.agent_labels_to_index,
    )
    graph_init = jax.tree.map(lambda x: x[jnp.newaxis, ...], graph_init)
    num_actors = config.derived_values.num_actors
    ac_init_x = (
        jnp.zeros(
            (
                1,
                num_actors,
                env.observation_space_for_agent(env.agent_labels[0]).shape[0] * agent_previous_obs_stack_size,
            )
        ),
        graph_init,
        jnp.zeros((1, num_actors)),
    )
    ac_init_h_state = ScannedRNN.initialize_carry(
        num_actors, config.network_config.gru_hidden_dim
    )

    return (
        ac_init_x,
        ac_init_h_state,
        graph_init,
    )


def get_critic_init_input(config: MAPPOConfig, env, graph_init):
    num_actors = config.derived_values.num_actors

    cr_init_x = (
        jnp.zeros(
            (
                1,
                num_actors,
                env.world_state_size(),
            )
        ),
        graph_init,
        jnp.zeros((1, num_actors)),
    )
    cr_init_h_state = ScannedRNN.initialize_carry(
        num_actors, config.network_config.gru_hidden_dim
    )
    return cr_init_x, cr_init_h_state


def get_init_communication_message(config: MAPPOConfig, env, ac_init_h_state):
    communication_type = config.env_config.env_kwargs.agent_communication_type

    num_env = config.training_config.num_envs

    initial_communication_message = jnp.asarray([])
    if communication_type == CommunicationType.HIDDEN_STATE.value:
        initial_communication_message = ac_init_h_state
    elif (
            communication_type == CommunicationType.PAST_ACTION.value
            or communication_type == CommunicationType.CURRENT_ACTION.value
    ):
        initial_communication_message = jnp.full(
            (config.derived_values.num_actors, 1), -1
        )

    initial_communication_message_env_input = initial_communication_message
    if initial_communication_message_env_input.size != 0:
        initial_communication_message_env_input = (
            initial_communication_message_env_input.reshape(
                num_env,
                env.num_agents,
                *initial_communication_message_env_input.shape[1:],
            )
        )

    return initial_communication_message, initial_communication_message_env_input


def critic_apply_cast(result: Any):
    return cast(
        tuple[Array, Array],
        result,
    )


def actor_apply_cast(result: Any):
    return cast(
        tuple[Array, distrax.Categorical],
        result,
    )


@jaxtyped(typechecker=beartype)
class StaticVariables(NamedTuple):
    env: MultiAgentEnv
    config: MAPPOConfig
    actor_network: GraphAttentionActorRNN
    critic_network: CriticRNN
    initial_communication_message: Float[Array, "..."]
    is_running_in_viz_mode: bool
    store_action_field: bool


@jaxtyped(typechecker=beartype)
class ActorAndCriticTrainStates(NamedTuple):
    actor_train_state: TrainState
    critic_train_state: TrainState


@jaxtyped(typechecker=beartype)
class ActorAndCriticHiddenStates(NamedTuple):
    actor_hidden_state: Array
    critic_hidden_state: Array


@jaxtyped(typechecker=beartype)
class EnvStepRunnerState(NamedTuple):
    network_train_states: ActorAndCriticTrainStates
    env_state: LogEnvState
    obs: MultiAgentObservation
    graph: MultiAgentGraph
    dones: Array
    hidden_states: ActorAndCriticHiddenStates
    communication_message: Float[Array, "..."]
    initial_entity_position: Float[Array, f"{EntityIndexAxis} ..."]
    rng_keys: PRNGKey


@jaxtyped(typechecker=beartype)
class UpdateStepRunnerState(NamedTuple):
    update_step_runner_state: EnvStepRunnerState
    update_step_counter: int


@jaxtyped(typechecker=beartype)
class UpdateEpochState(NamedTuple):
    network_train_states: ActorAndCriticTrainStates
    hidden_states: ActorAndCriticHiddenStates
    traj_batch: Transition
    advantages: Array
    targets: Array
    rng_keys: PRNGKey


@jaxtyped(typechecker=beartype)
def _env_step(
        env_step_static_variables: StaticVariables,
        runner_state: EnvStepRunnerState,
        unused,
):
    (
        env,
        config,
        actor_network,
        critic_network,
        initial_communication_message,
        is_running_in_viz_mode,
        store_action_field,
    ) = env_step_static_variables

    num_env = config.training_config.num_envs

    (
        train_states,
        log_env_state,
        last_obs,
        last_graph,
        last_done,
        h_states,
        last_communication_message,
        initial_entity_position,
        rng,
    ) = runner_state

    communication_type = config.env_config.env_kwargs.agent_communication_type

    initial_agent_communication_message = initial_communication_message
    agent_communication_message = initial_agent_communication_message
    if communication_type == CommunicationType.HIDDEN_STATE.value:
        agent_communication_message = last_communication_message.reshape(
            num_env, env.num_agents, *last_communication_message.shape[1:]
        )
    elif communication_type == CommunicationType.PAST_ACTION.value:
        agent_communication_message = last_communication_message.reshape(
            num_env, env.num_agents, *last_communication_message.shape[1:]
        )

    # SELECT ACTION
    rng, _rng = jax.random.split(rng)
    obs_batch = batchify(last_obs, env.agent_labels, config.derived_values.num_actors)
    graph_batch = batchify_graph(last_graph, env.agent_labels_to_index)
    graph_network_input = jax.tree.map(lambda x: x[jnp.newaxis, ...], graph_batch)
    ac_in = (
        obs_batch[jnp.newaxis, :],
        graph_network_input,
        last_done[jnp.newaxis, :],
    )
    ac_h_state, pi = actor_apply_cast(
        actor_network.apply(
            train_states.actor_train_state.params, h_states.actor_hidden_state, ac_in
        )
    )
    action_field = jnp.asarray([])
    if is_running_in_viz_mode and store_action_field:
        rng, lin_space_rng = jax.random.split(rng)
        lin_space_env_state = env.get_viz_states(
            LinSpaceConfig(lin_range=(-10, 10), lin_step=0.4), log_env_state.env_state
        )

        @partial(jax.vmap)
        def get_lin_spaced_data(state):
            obs = jax.vmap(env.get_observation)(state)
            graph = jax.vmap(env.get_graph)(state)

            obs = batchify(obs, env.agent_labels, config.derived_values.num_actors)
            graph = batchify_graph(graph, env.agent_labels_to_index)
            dones_with_agent_label = {
                agent_label: state.dones[:, i]
                for i, agent_label in enumerate(env.agent_labels)
            }
            dones = batchify(
                dones_with_agent_label,
                env.agent_labels,
                config.derived_values.num_actors,
            ).squeeze()

            return obs, graph, dones

        @partial(jax.vmap, in_axes=(0, None, None, 0), out_axes=1)
        def get_action_field_for_single_lin_space(
                _rng, actor_params: dict, actor_h_state, ac_lin_in
        ):
            ac_lin_in = jax.tree.map(lambda x: x[None], ac_lin_in)

            _, line_spaced_pi = actor_apply_cast(
                actor_network.apply(
                    actor_params,
                    actor_h_state,
                    ac_lin_in,
                )
            )
            return line_spaced_pi.sample(seed=_rng).squeeze()

        lin_spaced_ac_in = get_lin_spaced_data(lin_space_env_state)
        lin_space_rng = jax.random.split(
            lin_space_rng, num=lin_spaced_ac_in[-1].shape[0]
        )
        line_spaced_pi = get_action_field_for_single_lin_space(
            lin_space_rng,
            train_states.actor_train_state.params,
            h_states.actor_hidden_state,
            lin_spaced_ac_in,
        )

        action_field = line_spaced_pi

    action = pi.sample(seed=_rng)

    if communication_type == CommunicationType.CURRENT_ACTION.value:
        agent_communication_message = action.reshape(
            num_env, env.num_agents, *last_communication_message.shape[1:]
        )
    log_env_state = log_env_state.replace(
        env_state=log_env_state.env_state._replace(
            agent_communication_message=agent_communication_message
        )
    )

    log_prob = pi.log_prob(action)
    env_act = unbatchify(action, env.agent_labels, num_env, env.num_agents)
    # VALUE
    # output of wrapper is (num_envs, num_agents, world_state_size)
    # swap axes to (num_agents, num_envs, world_state_size) before reshaping to (num_actors, world_state_size)
    world_state = last_obs["world_state"].swapaxes(0, 1)
    world_state = world_state.reshape((config.derived_values.num_actors, -1))
    cr_in = (
        world_state[None, :],
        graph_network_input,
        last_done[jnp.newaxis, :],
    )
    cr_h_state, value = critic_apply_cast(
        critic_network.apply(
            train_states.critic_train_state.params, h_states.critic_hidden_state, cr_in
        )
    )

    # STEP ENV
    rng, _rng = jax.random.split(rng)
    rng_step = jax.random.split(_rng, num_env)
    env_input = (
        rng_step,
        log_env_state,
        env_act,
        initial_agent_communication_message,
        initial_entity_position,
    )
    env_in_axes = jax.tree.map(lambda leaf: 0 if leaf.size != 0 else None, env_input)
    obs_v, graph_v, log_env_state, reward, done, info = jax.vmap(
        env.step,
        in_axes=env_in_axes,
    )(
        *env_input,
    )
    info = jax.tree.map(lambda x: x.reshape(config.derived_values.num_actors), info)
    done_batch = batchify(
        done, env.agent_labels, config.derived_values.num_actors
    ).squeeze()

    if communication_type == CommunicationType.HIDDEN_STATE.value:
        last_communication_message = ac_h_state
    elif (
            communication_type == CommunicationType.PAST_ACTION.value
            or communication_type == CommunicationType.CURRENT_ACTION.value
    ):
        last_communication_message = action.squeeze()[..., None]

    transition = Transition(
        jnp.tile(done["__all__"], env.num_agents),
        last_done,
        action.squeeze(),
        value.squeeze(),
        batchify(reward, env.agent_labels, config.derived_values.num_actors).squeeze(),
        log_prob.squeeze(),
        obs_batch,
        graph_batch,
        world_state,
        info,
    )
    if is_running_in_viz_mode:
        tiled_log_env_state = jax.tree.map(
            lambda x: jnp.tile(x, (env.num_agents,) + (1,) * (x.ndim - 1)),
            log_env_state,
        )
        transition = TransitionForVisualization(
            *transition,
            env_state=tiled_log_env_state,
        )
        if store_action_field:
            transition = TransitionWithActionField(
                *transition,
                action_field=action_field,
            )
    runner_state = EnvStepRunnerState(
        train_states,
        log_env_state,
        obs_v,
        graph_v,
        done_batch,
        ActorAndCriticHiddenStates(ac_h_state, cr_h_state),
        last_communication_message,
        initial_entity_position,
        rng,
    )
    return runner_state, transition


# UPDATE NETWORK
@jaxtyped(typechecker=beartype)
def _update_epoch(
        update_epoch_static_variables: StaticVariables,
        update_state: UpdateEpochState,
        unused,
):
    _, config, actor_network, critic_network, _, _, _ = update_epoch_static_variables
    ppo_config = config.training_config.ppo_config

    def _update_minibatch(train_states, batch_info):
        actor_train_state, critic_train_state = train_states
        (
            ac_init_h_state,
            cr_init_h_state,
            traj_batch,
            advantages,
            targets,
        ) = batch_info

        def _actor_loss_fn(actor_params, init_h_state, traj_batch, gae):
            nonlocal ppo_config, actor_network
            # RERUN NETWORK
            _, pi = actor_apply_cast(
                actor_network.apply(
                    actor_params,
                    init_h_state.squeeze(),
                    (traj_batch.obs, traj_batch.graph, traj_batch.done),
                )
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
                        1.0 - ppo_config.clip_eps,
                        1.0 + ppo_config.clip_eps,
                    )
                    * gae
            )
            loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
            loss_actor = loss_actor.mean()
            entropy = pi.entropy().mean()

            # debug
            approx_kl = ((ratio - 1) - log_ratio).mean()
            clip_frac = jnp.mean(jnp.abs(ratio - 1) > ppo_config.clip_eps)

            actor_loss = loss_actor - ppo_config.entropy_coefficient * entropy
            return actor_loss, (
                loss_actor,
                entropy,
                ratio,
                approx_kl,
                clip_frac,
            )

        def _critic_loss_fn(critic_params, init_h_state, traj_batch, targets):
            nonlocal critic_train_state, ppo_config, critic_network
            # RERUN NETWORK
            _, value = critic_apply_cast(
                critic_network.apply(
                    critic_params,
                    init_h_state.squeeze(),
                    (
                        traj_batch.world_state,
                        traj_batch.graph,
                        traj_batch.done,
                    ),
                )
            )

            # CALCULATE VALUE LOSS
            value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
                -ppo_config.clip_eps, ppo_config.clip_eps
            )
            value_losses = jnp.square(value - targets)
            value_losses_clipped = jnp.square(value_pred_clipped - targets)
            value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
            critic_loss = ppo_config.value_coefficient * value_loss
            return critic_loss, value_loss

        actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
        actor_loss, actor_grads = actor_grad_fn(
            actor_train_state.params,
            ac_init_h_state,
            traj_batch,
            advantages,
        )
        critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
        critic_loss, critic_grads = critic_grad_fn(
            critic_train_state.params, cr_init_h_state, traj_batch, targets
        )

        actor_train_state = actor_train_state.apply_gradients(grads=actor_grads)
        critic_train_state = critic_train_state.apply_gradients(grads=critic_grads)

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

        return (
            ActorAndCriticTrainStates(actor_train_state, critic_train_state),
            loss_info,
        )

    (
        train_states,
        last_ppo_step_h_state,
        traj_batch,
        advantages,
        targets,
        rng,
    ) = update_state
    rng, _rng = jax.random.split(rng)

    last_ppo_step_h_state = jax.tree.map(
        lambda x: jnp.reshape(x, (1, config.derived_values.num_actors, -1)),
        last_ppo_step_h_state,
    )
    # traj_batch.graph.nodes: Float[Array, "num_steps num_actors; num_entities, node_feature_dim"]
    # remember last two axis are for one graph
    batch = (
        last_ppo_step_h_state.actor_hidden_state,
        last_ppo_step_h_state.critic_hidden_state,
        traj_batch,
        advantages.squeeze(),
        targets.squeeze(),
    )
    permutation = jax.random.permutation(_rng, config.derived_values.num_actors)

    shuffled_batch = jax.tree.map(lambda x: jnp.take(x, permutation, axis=1), batch)

    minibatches = jax.tree.map(
        lambda x: jnp.swapaxes(
            jnp.reshape(
                x,
                [x.shape[0], ppo_config.num_minibatches_actors, -1] + list(x.shape[2:]),
            ),
            1,
            0,
        ),
        shuffled_batch,
    )

    train_states, loss_info = jax.lax.scan(_update_minibatch, train_states, minibatches)
    update_state = UpdateEpochState(
        train_states,
        jax.tree.map(lambda x: x.squeeze(), last_ppo_step_h_state),
        traj_batch,
        advantages,
        targets,
        rng,
    )
    return update_state, loss_info


# TRAIN LOOP
@jaxtyped(typechecker=beartype)
def ppo_single_update(
        static_variables: StaticVariables,
        update_runner_state: UpdateStepRunnerState,
        unused,
):
    (
        env,
        config,
        actor_network,
        critic_network,
        initial_communication_message,
        is_running_in_viz_mode,
        store_action_field,
    ) = static_variables
    ppo_config = config.training_config.ppo_config
    num_env = config.training_config.num_envs

    # COLLECT TRAJECTORIES
    runner_state, update_steps = update_runner_state

    last_step_h_states = runner_state.hidden_states

    _env_step_with_static_args = partial(
        _env_step,
        StaticVariables(
            env,
            config,
            actor_network,
            critic_network,
            initial_communication_message,
            is_running_in_viz_mode,
            store_action_field,
        ),
    )
    runner_state, traj_batch = jax.lax.scan(
        _env_step_with_static_args, runner_state, None, ppo_config.num_steps_per_update
    )

    # CALCULATE ADVANTAGE
    (
        train_states,
        env_state,
        last_obs,
        last_graph,
        last_done,
        h_states,
        last_communication_message,
        entity_initial_position,
        rng,
    ) = runner_state

    last_world_state = last_obs["world_state"].swapaxes(0, 1)
    last_world_state = last_world_state.reshape((config.derived_values.num_actors, -1))
    graph_batch = batchify_graph(last_graph, env.agent_labels_to_index)
    graph_network_input = jax.tree.map(lambda x: x[None, ...], graph_batch)
    cr_in = (
        last_world_state[None, :],
        graph_network_input,
        last_done[np.newaxis, :],
    )
    _, last_val = critic_apply_cast(
        critic_network.apply(
            train_states.critic_train_state.params, h_states.critic_hidden_state, cr_in
        )
    )
    last_val = last_val.squeeze()

    def _calculate_gae(traj_batch, last_val):
        def _get_advantages(gae_and_next_value, transition):
            nonlocal config, ppo_config
            gae, next_value = gae_and_next_value
            done, value, reward = (
                transition.global_done,
                transition.value,
                transition.reward,
            )
            delta = (
                    reward + config.training_config.gamma * next_value * (1 - done) - value
            )
            gae = (
                    delta
                    + config.training_config.gamma
                    * ppo_config.gae_lambda
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

    update_epoch_with_static_variables = partial(_update_epoch, static_variables)
    update_state = UpdateEpochState(
        train_states,
        last_step_h_states,
        traj_batch,
        advantages,
        targets,
        rng,
    )
    update_state, loss_info = jax.lax.scan(
        update_epoch_with_static_variables, update_state, None, ppo_config.update_epochs
    )
    loss_info["ratio_0"] = loss_info["ratio"].at[0, 0].get()
    loss_info = jax.tree.map(lambda x: x.mean(), loss_info)

    train_states = update_state.network_train_states
    metric = traj_batch.info
    metric["loss"] = loss_info
    rng = update_state.rng_keys

    def callback(metric):
        out = metric["actor_network"]
        progress = round(
            (metric["update_steps"] / config.derived_values.num_updates) * 100,
            4,
        )
        update_steps = metric["update_steps"]
        if (
                config.wandb_config.save_model
                and update_steps % config.wandb_config.checkpoint_model_every_update_steps
                == 0
        ):
            dict_config = config_to_dict(config)

            model_artifact = wandb.Artifact(
                "PPO_RNN_Runner_State",
                type="model",
                metadata=dict_config,
            )
            running_script_path = os.path.abspath(".")
            checkpoint_dir = os.path.join(
                running_script_path,
                f"saved_actor/{wandb.run.name}/PPO_Runner_Checkpoint_{progress}",
            )
            orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
            save_args = orbax_utils.save_args_from_target(out)
            orbax_checkpointer.save(checkpoint_dir, out, save_args=save_args)
            model_artifact.add_dir(checkpoint_dir)

            wandb.log_artifact(model_artifact)
        print(
            f"progress: {progress:.4f}% ; update step: {update_steps}/{config.derived_values.num_updates}"
        )
        wandb.log(
            {
                "returns": metric["returned_episode_returns"][-1, :].mean(),
                "env_step": update_steps
                            * config.training_config.num_envs
                            * ppo_config.num_steps_per_update,
                **metric["loss"],
            }
        )

    metric["update_steps"] = update_steps
    metric["actor_network"] = {
        "actor_train_params": train_states.actor_train_state.params,
    }
    jax.experimental.io_callback(callback, None, metric)
    update_steps += 1
    actor_critic_train_states = ActorAndCriticTrainStates(*train_states)
    actor_critic_hidden_states = ActorAndCriticHiddenStates(*h_states)
    runner_state = EnvStepRunnerState(
        actor_critic_train_states,
        env_state,
        last_obs,
        last_graph,
        last_done,
        actor_critic_hidden_states,
        last_communication_message,
        entity_initial_position,
        rng,
    )
    return UpdateStepRunnerState(runner_state, update_steps), metric


def make_train(config: MAPPOConfig):
    env = make_env_from_config(config)

    ppo_config = config.training_config.ppo_config

    def linear_schedule(count):
        nonlocal config, ppo_config
        frac = (
                1.0
                - (count // (ppo_config.num_minibatches_actors * ppo_config.update_epochs))
                / config.derived_values.num_updates
        )
        return config.training_config.lr * frac

    def train(rng):
        nonlocal env, config, ppo_config, linear_schedule
        num_env = config.training_config.num_envs
        # INIT NETWORK
        actor_network = GraphAttentionActorRNN(
            env.action_space_for_agent(env.agent_labels[0]).n, config=config
        )

        critic_network = CriticRNN(config=config)

        rng, _rng_actor, _rng_critic = jax.random.split(rng, 3)

        num_actors = config.derived_values.num_actors
        ac_init_x, ac_init_h_state, graph_init = get_actor_init_input(config, env)
        actor_network_params = actor_network.init(
            _rng_actor, ac_init_h_state, ac_init_x
        )
        cr_init_x, cr_init_h_state = get_critic_init_input(config, env, graph_init)
        critic_network_params = critic_network.init(
            _rng_critic, cr_init_h_state, cr_init_x
        )

        max_grad_norm = ppo_config.max_grad_norm
        lr = config.training_config.lr
        if config.training_config.anneal_lr:
            actor_tx = optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
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
            params=actor_network_params,
            tx=actor_tx,
        )
        critic_train_state = TrainState.create(
            apply_fn=critic_network.apply,
            params=critic_network_params,
            tx=critic_tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, num_env)
        ac_init_h_state = ScannedRNN.initialize_carry(
            config.derived_values.num_actors, config.network_config.gru_hidden_dim
        )
        cr_init_h_state = ScannedRNN.initialize_carry(
            config.derived_values.num_actors, config.network_config.gru_hidden_dim
        )

        initial_communication_message, initial_communication_message_env_input = (
            get_init_communication_message(config, env, ac_init_h_state)
        )
        initial_entity_position = jnp.asarray([])
        obs_v, graph_v, env_state = jax.vmap(
            env.reset,
            in_axes=(
                0,
                0 if initial_communication_message_env_input.size != 0 else None,
                None,
            ),
        )(reset_rng, initial_communication_message_env_input, initial_entity_position)

        rng, _rng = jax.random.split(rng)
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
            initial_entity_position,
            _rng,
        )
        update_step_static_args = StaticVariables(
            env,
            config,
            actor_network,
            critic_network,
            initial_communication_message_env_input,
            False,
            False,
        )
        ppo_single_update_with_static_args = partial(
            ppo_single_update, update_step_static_args
        )
        runner_state, metric = jax.lax.scan(
            ppo_single_update_with_static_args,
            UpdateStepRunnerState(env_step_runner_state, 0),
            None,
            config.derived_values.num_updates,
        )
        return {"runner_state": runner_state}

    return train


def main():
    config: MAPPOConfig = MAPPOConfig.create()
    assert (
            config.training_config.num_envs > 1
    ), "Number of environments must be greater than 1 for training"
    dict_config = config_to_dict(config)
    wandb.init(
        entity=config.wandb_config.entity,
        project=config.wandb_config.project,
        mode=config.wandb_config.mode,
        config=dict_config,
    )
    rng = jax.random.PRNGKey(config.training_config.seed)
    with jax.disable_jit(False):
        train_jit = jax.jit(make_train(config))
        out = train_jit(rng)
        block_until_ready(out)

    runner_state: UpdateStepRunnerState = out["runner_state"]
    out = {
        "actor_train_params": runner_state.update_step_runner_state.network_train_states.actor_train_state.params,
        # "critic_train_state": out["runner_state"][0][0][1].params,
    }

    if config.wandb_config.save_model:
        model_artifact = wandb.Artifact(
            "PPO_RNN_Runner_State", type="model", metadata=dict_config
        )
        running_script_path = os.path.abspath(".")
        checkpoint_dir = os.path.join(
            running_script_path,
            f"saved_actor/{wandb.run.name}/PPO_Runner_Checkpoint_final",
        )
        orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        save_args = orbax_utils.save_args_from_target(out)
        orbax_checkpointer.save(checkpoint_dir, out, save_args=save_args)
        model_artifact.add_dir(checkpoint_dir)

        wandb.log_artifact(model_artifact)

    wandb.finish()


if __name__ == "__main__":
    main()

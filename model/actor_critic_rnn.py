import functools

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import jraph
from flax.linen.initializers import constant, orthogonal
from jaxtyping import Float, Array
from jraph._src import utils

from config.mappo_config import MAPPOConfig


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
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


# noinspection DuplicatedCode
class ActorRNN(nn.Module):
    action_dim: list[int]
    config: MAPPOConfig

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        embedding = nn.Dense(
            self.config.network.fc_dim_size,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(
            self.config.network.gru_hidden_dim,
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        actor_mean = nn.relu(actor_mean)
        action_logits = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        pi = distrax.Categorical(logits=action_logits)

        return hidden, pi


# This is built off of the GAT implementation in jraph
class GraphMultiHeadAttentionLayer(nn.Module):
    config: MAPPOConfig

    @nn.compact
    def __call__(self, graph: jraph.GraphsTuple, avg_multi_head):
        # Assumes that given graph is in into jraph compatible format
        nodes, edges, receivers, senders, _, _, _ = graph

        # Equivalent to the sum of n_node, but statically known.
        sum_n_node = nodes.shape[0]

        def linear_layer(x):
            return nn.Dense(
                self.config.network.fc_dim_size,
                kernel_init=orthogonal(jnp.sqrt(2)),
                bias_init=constant(0.0),
            )(x)

        # embed node features
        nodes = linear_layer(nodes)
        # embed edge features
        edge_features = linear_layer(edges)

        sent_attributes = nodes[senders]
        received_attributes = nodes[receivers]

        nodes_seg_sum_from_each_attn_head = []

        for _ in range(self.config.network.num_heads_per_attn_layer):
            # extract key for node feature to be used in attention
            key_sent_attributes = linear_layer(sent_attributes)
            key_received_attributes = linear_layer(received_attributes)

            key_received_attributes = key_received_attributes + edge_features

            softmax_logits: Float[Array, "edge_id, edge_id"] = jnp.sum(
                key_sent_attributes * key_received_attributes, axis=1
            ) / jnp.sqrt(self.config.network.fc_dim_size)

            # Compute the softmax weights on the entire tree.
            weights = utils.segment_softmax(
                softmax_logits, segment_ids=receivers, num_segments=sum_n_node
            )
            # Apply weights
            messages = weights[..., None] * (sent_attributes + edge_features)
            # Aggregate messages to nodes.
            nodes_seg_sum = jax.ops.segment_sum(
                messages, receivers, num_segments=sum_n_node
            )
            nodes_seg_sum_from_each_attn_head.append(nodes_seg_sum)

        if avg_multi_head:
            nodes_seg_sum = jnp.mean(
                jnp.stack(nodes_seg_sum_from_each_attn_head), axis=0
            )
        else:
            nodes_seg_sum = jnp.concatenate(nodes_seg_sum_from_each_attn_head, axis=1)
            nodes_seg_sum = linear_layer(nodes_seg_sum)

        nodes += nodes_seg_sum

        nodes = nn.relu(nodes)

        return graph._replace(nodes=nodes)


class GraphTransformer(nn.Module):
    config: MAPPOConfig

    @nn.compact
    def __call__(self, graph: jraph.GraphsTuple):
        """Applies a Graph Attention layer."""

        # nodes: Float[Array, "graph_id entity_id features"]
        # edges: Float[Array, "graph_id edge_id features"]
        # receivers/senders: Int[Array, "graph_id edge_id"]
        # n_node/n_edge: Int[Array, "graph_id"]
        # here graph_id is like batch_id. One different environment/batch has different graphs
        nodes, edges, receivers, senders, _, _, _ = graph
        num_graph, n_node, _ = nodes.shape
        _, n_edge, _ = edges.shape

        # reshape into jraph compatible format
        nodes = nodes.reshape((num_graph * n_node, -1))
        edges = edges.reshape((num_graph * n_edge, -1))
        index_offset = jnp.arange(num_graph)[..., None]
        receivers = receivers + index_offset * n_node
        senders = senders + index_offset * n_node
        receivers = receivers.flatten().astype(jnp.int32)
        senders = senders.flatten().astype(jnp.int32)

        # Embed and extract node features.
        entity_type = nodes[..., -1].astype(jnp.int32)
        entity_emb = nn.Embed(2, self.config.network.embedding_dim)(entity_type)
        nodes = jnp.concatenate([nodes[..., :-1], entity_emb], axis=-1)

        graph = graph._replace(
            nodes=nodes, edges=edges, receivers=receivers, senders=senders
        )

        for _ in range(self.config.network.num_graph_attn_layers - 1):
            graph = GraphMultiHeadAttentionLayer(self.config)(
                graph, avg_multi_head=False
            )
        # Average the multi-head attention for the last layer
        graph = GraphMultiHeadAttentionLayer(self.config)(graph, avg_multi_head=True)

        return graph


class GraphTransformerActorRNN(nn.Module):
    action_dim: list[int]
    config: MAPPOConfig

    @nn.compact
    def __call__(self, hidden, x):

        obs, graph, dones = x

        # graph_transformer = GraphTransformer(self.config)(graph)

        # nodes: Float[Array, "graph_id entity_id features"]
        # edges: Float[Array, "graph_id edge_id features"]
        # receivers/senders: Int[Array, "graph_id edge_id"]
        # n_node/n_edge: Int[Array, "graph_id"]
        # here graph_id is like batch_id. One different environment/batch has different graphs

        hidden, pi = ActorRNN(self.action_dim, self.config)(hidden, (obs, dones))

        return hidden, pi


# noinspection DuplicatedCode
class CriticRNN(nn.Module):
    config: MAPPOConfig

    @nn.compact
    def __call__(self, hidden, x):
        world_state, dones = x
        embedding = nn.Dense(
            self.config.network.fc_dim_size,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(world_state)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        critic = nn.Dense(
            self.config.network.gru_hidden_dim,
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return hidden, jnp.squeeze(critic, axis=-1)

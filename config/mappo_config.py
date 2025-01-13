from __future__ import annotations

from enum import Enum

from beartype import beartype
from flax import struct


class CommunicationType(Enum):
    HIDDEN_STATE = 1
    PAST_ACTION = 2
    CURRENT_ACTION = 3


@beartype
class MAPPOConfig(struct.PyTreeNode):
    class EnvConfig(struct.PyTreeNode):
        class EnvKwArgs(struct.PyTreeNode):
            """
            Attributes:
                num_agents: Number of agents in a single environment.
                max_steps: Rollout length in a single environment.
                agent_max_speed: Negative value means no maximum speed.
            """

            num_agents = 8
            max_steps = 50
            dist_to_goal_reward_ratio = 0.30
            agent_max_speed = -1
            agent_visibility_radius = (1.5,)
            entities_initial_coord_radius = (2.5,)
            entity_acceleration = 5
            one_time_death_reward = 15
            agent_communication_type = CommunicationType.CURRENT_ACTION
            agent_control_noise_std = 0.2

        env_cls_name = "TargetMPEEnvironment"
        kwargs = EnvKwArgs()

    class TrainingConfig(struct.PyTreeNode):
        class PPOConfig(struct.PyTreeNode):
            """
            Attributes:
                clip_eps: clip_param for PPO to make sure the policy being updated via SGD is close to the policy
                            used in the rollout phase.
                num_steps_per_update: number of samples collected from all environment during the rollout phase
                                    before the model is updated . This is batch_size per actor
                num_minibatches_actors: Number of actors to use in one epoch.
                update_epochs: Number of epochs to update the policy per update. One epoch is NUM_MINIBATCHES updates.
            """

            clip_eps = 0.2
            is_clip_eps_per_env = False
            max_grad_norm = 0.5
            num_steps_per_update = 256
            num_minibatches_actors = 4
            update_epochs = 4

            gae_lambda = 0.95
            entropy_coefficient = 0.01
            value_coefficient = 0.5

        """
        Attributes:
            total_timesteps: Total env time steps to collect. This will be distributed across num_envs environments.
            gamma: discount factor.
        """

        seed = 0
        num_seeds = 2
        lr = 2e-3
        anneal_lr = True
        num_envs = 4
        gamma = 0.99
        total_timesteps = 1e4
        ppo_config = PPOConfig()

    class NetworkConfig(struct.PyTreeNode):
        fc_dim_size = 64
        gru_hidden_dim = 64

        actor_num_hidden_linear_layer = 4
        critic_num_hidden_linear_layer = 4

        entity_type_embedding_dim = 4

        num_graph_attn_layers = 3
        num_heads_per_attn_layer = 4
        graph_attention_key_dim = 8

        graph_num_linear_layer = 4
        graph_hidden_feature_dim = 32

    class WandbConfig(struct.PyTreeNode):
        entity = "josssdan"
        project = "JaxInforMARL"
        mode = "disabled"
        save_model = False
        checkpoint_model_every_update_steps = 1e2

    class DerivedValues(struct.PyTreeNode):
        num_actors: int
        num_updates: int
        minibatch_size: int
        scaled_clip_eps: float

    env_config = EnvConfig()
    training_config = TrainingConfig()
    network = NetworkConfig()
    wandb = WandbConfig()
    derived_values: DerivedValues = struct.field()

    @classmethod
    def create(cls) -> MAPPOConfig:
        env_config = cls.env_config
        train_config = cls.training_config
        num_actors = env_config.kwargs.num_agents * train_config.num_envs
        batch_size = num_actors * train_config.ppo_config.num_steps_per_update
        _derived_values = cls.DerivedValues(
            num_actors=num_actors,
            num_updates=int(
                train_config.total_timesteps
                // train_config.num_envs
                // train_config.ppo_config.num_steps_per_update
            ),
            minibatch_size=(
                batch_size // train_config.ppo_config.num_minibatches_actors
            ),
            scaled_clip_eps=(
                train_config.ppo_config.clip_eps / env_config.kwargs.num_agents
                if train_config.ppo_config.is_clip_eps_per_env
                else train_config.ppo_config.clip_eps
            ),
        )
        assert (
            _derived_values.num_updates > 0
        ), "Number of updates per environment must be greater than 0."

        return cls(derived_values=_derived_values)


def config_to_dict(config):
    is_primitive_type = lambda obj: isinstance(
        obj, (int, float, str, bool, type(None), CommunicationType, tuple)
    )
    return {
        attr: (
            getattr(config, attr)
            if is_primitive_type(getattr(config, attr))
            else config_to_dict(getattr(config, attr))
        )
        for attr in dir(config)
        if not callable(getattr(config, attr))
        and not (attr.startswith("__") or attr.startswith("_"))
    }

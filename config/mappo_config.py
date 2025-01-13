from __future__ import annotations

from enum import Enum
from typing import NamedTuple

from beartype import beartype


class CommunicationType(Enum):
    HIDDEN_STATE = 1
    PAST_ACTION = 2
    CURRENT_ACTION = 3


class EnvConfig(NamedTuple):
    class EnvKwArgs(NamedTuple):
        """
        Attributes:
            num_agents: Number of agents in a single environment.
            max_steps: Rollout length in a single environment.
            agent_max_speed: Negative value means no maximum speed.
        """

        num_agents = 3
        max_steps = 50
        dist_to_goal_reward_ratio = 0.30
        agent_max_speed = -1
        agent_visibility_radius = [
            1.5,
        ]
        entities_initial_coord_radius = [
            2.5,
        ]
        entity_acceleration = 5
        one_time_death_reward = 15
        agent_communication_type = CommunicationType.CURRENT_ACTION
        agent_control_noise_std = 0.2

    env_cls_name = "TargetMPEEnvironment"
    kwargs = EnvKwArgs()


class TrainingConfig(NamedTuple):
    class PPOConfig(NamedTuple):
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
        num_steps_per_update = 128
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


class NetworkConfig(NamedTuple):
    fc_dim_size = 8
    gru_hidden_dim = 8

    actor_num_hidden_linear_layer = 2
    critic_num_hidden_linear_layer = 2

    entity_type_embedding_dim = 4

    num_graph_attn_layers = 1
    num_heads_per_attn_layer = 2
    graph_attention_key_dim = 8

    graph_num_linear_layer = 2
    graph_hidden_feature_dim = 8


class WandbConfig(NamedTuple):
    entity = "josssdan"
    project = "JaxInforMARL"
    mode = "disabled"
    save_model = False
    checkpoint_model_every_update_steps = 1e2


class DerivedValues(NamedTuple):
    num_actors: int
    num_updates: int
    minibatch_size: int
    scaled_clip_eps: float


@beartype
class MAPPOConfig(NamedTuple):
    env_config: EnvConfig
    training_config: TrainingConfig
    network: NetworkConfig
    wandb: WandbConfig
    derived_values: DerivedValues

    @classmethod
    def create(cls) -> MAPPOConfig:
        env_config = EnvConfig()
        training_config = TrainingConfig()
        network = NetworkConfig()
        wandb = WandbConfig()

        num_actors = env_config.kwargs.num_agents * training_config.num_envs
        batch_size = num_actors * training_config.ppo_config.num_steps_per_update
        _derived_values = DerivedValues(
            num_actors=num_actors,
            num_updates=int(
                training_config.total_timesteps
                // training_config.num_envs
                // training_config.ppo_config.num_steps_per_update
            ),
            minibatch_size=(
                batch_size // training_config.ppo_config.num_minibatches_actors
            ),
            scaled_clip_eps=(
                training_config.ppo_config.clip_eps / env_config.kwargs.num_agents
                if training_config.ppo_config.is_clip_eps_per_env
                else training_config.ppo_config.clip_eps
            ),
        )
        assert (
            _derived_values.num_updates > 0
        ), "Number of updates per environment must be greater than 0."

        return cls(
            env_config=env_config,
            training_config=training_config,
            network=network,
            wandb=wandb,
            derived_values=_derived_values,
        )

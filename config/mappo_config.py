from __future__ import annotations

from enum import Enum
from typing import Literal, NamedTuple


class CommunicationType(Enum):
    HIDDEN_STATE = "HIDDEN_STATE"
    PAST_ACTION = "PAST_ACTION"
    CURRENT_ACTION = "CURRENT_ACTION"


class EnvKwArgs(NamedTuple):
    """
    Attributes:
        num_agents: Number of agents in a single environment.
        max_steps: Rollout length in a single environment.
        agent_max_speed: Negative value means no maximum speed.
    """

    num_agents: int = 3
    max_steps: int = 50
    collision_reward_coefficient: float = -10
    one_time_death_reward: int = 10
    distance_to_goal_reward_coefficient: int = 1
    entity_acceleration: int = 5

    agent_max_speed: int = -1
    agent_visibility_radius: list[float] = [
        0.25,
    ]
    entities_initial_coord_radius: list[float] = [
        1,
    ]
    agent_communication_type: CommunicationType.value = None
    agent_control_noise_std: float = 0.0
    add_self_edges_to_nodes: bool = True

    add_target_goal_to_nodes: bool = True
    heterogeneous_agents: bool = False

    agent_previous_obs_stack_size: int = 1


class EnvConfig(NamedTuple):
    env_cls_name: str = "StackedTargetMPEEnvironment"
    env_kwargs: EnvKwArgs = EnvKwArgs()


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

    clip_eps: float = 0.2
    is_clip_eps_per_env: bool = False
    max_grad_norm: float = 10
    num_steps_per_update: int = 128
    num_minibatches_actors: int = 4
    update_epochs: int = 4

    gae_lambda: float = 0.95
    entropy_coefficient: float = 0.01
    value_coefficient: float = 0.5


class TrainingConfig(NamedTuple):
    """
    Attributes:
        total_timesteps: Total env time steps to collect. This will be distributed across num_envs environments.
        gamma: discount factor.
    """

    seed: int = 1
    num_seeds: int = 2
    lr: float = 5e-4
    anneal_lr: bool = True
    num_envs: int = 4
    gamma: float = 0.99
    total_timesteps: float = 1e4
    ppo_config: PPOConfig = PPOConfig()


class NeuralODEConfig(NamedTuple):
    ode_hidden_dim: int = 8
    ode_num_layers: int = 2
    dt: float = 0.2
    steps: int = 10


class NetworkConfig(NamedTuple):
    use_rnn: bool = False
    use_graph_attention_in_actor: bool = True
    use_graph_attention_in_critic: bool = False

    fc_dim_size: int = 8
    gru_hidden_dim: int = 8

    actor_num_hidden_linear_layer: int = 2
    critic_num_hidden_linear_layer: int = 2

    entity_type_embedding_dim: int = 4

    num_graph_attn_layers: int = 2
    num_heads_per_attn_layer: int = 3
    graph_attention_key_dim: int = 8

    graph_num_linear_layer: int = 2
    graph_hidden_feature_dim: int = 8

    neural_ODE_config: NeuralODEConfig = NeuralODEConfig()


class WandbConfig(NamedTuple):
    entity: str = "josssdan"
    project: str = "JaxInforMARL"
    mode: Literal["online", "offline", "disabled"] = "disabled"
    save_model: bool = False
    checkpoint_model_every_update_steps: float = 1e2


class DerivedValues(NamedTuple):
    num_actors: int
    num_updates: int
    minibatch_size: int
    scaled_clip_eps: float
    num_entity_types: int


class MAPPOConfig(NamedTuple):
    env_config: EnvConfig
    training_config: TrainingConfig
    network_config: NetworkConfig
    wandb_config: WandbConfig
    derived_values: DerivedValues

    @classmethod
    def create(
            cls,
            env_config=EnvConfig(),
            training_config=TrainingConfig(),
            network_config=NetworkConfig(),
            wandb_config=WandbConfig(),
            testing=False,
    ) -> MAPPOConfig:
        num_actors = env_config.env_kwargs.num_agents * training_config.num_envs
        batch_size = num_actors * training_config.ppo_config.num_steps_per_update
        num_entity_types = env_config.env_kwargs.num_agents * 2
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
                training_config.ppo_config.clip_eps / env_config.env_kwargs.num_agents
                if training_config.ppo_config.is_clip_eps_per_env
                else training_config.ppo_config.clip_eps
            ),
            num_entity_types=num_entity_types
        )
        if not testing:
            assert (
                    _derived_values.num_updates > 0
            ), "Number of updates per environment must be greater than 0."

        return cls(
            env_config=env_config,
            training_config=training_config,
            network_config=network_config,
            wandb_config=wandb_config,
            derived_values=_derived_values,
        )

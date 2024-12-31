from __future__ import annotations

from beartype import beartype
from flax import struct
from flax.struct import dataclass


@beartype
@dataclass
class Config:
    @dataclass
    class EnvConfig:
        """
        Attributes:
            gamma: discount factor.
        """

        @dataclass
        class EnvKwArgs:
            """
            Attributes:
                num_agents: Number of agents in a single environment.
                max_steps: Rollout length in a single environment.
            """

            num_agents = 3
            max_steps = 25

        cls_name = "TargetMPEEnvironment"
        gamma = 0.99
        kwargs = EnvKwArgs()

    @dataclass
    class TrainingConfig:
        @dataclass
        class PPOConfig:
            """
            Attributes:
                clip_eps: clip_param for PPO to make sure the policy being updated via SGD is close to the policy
                            used in the rollout phase.
                num_steps_per_update_per_env: number of samples collected from each environment before the model is
                                                updated during the rollout phase. This is batch_size per actor
                num_minibatches: Number of mini batch in a single batch.
                                mini batch size =  num_steps_per_update_per_env (which is train batch size) / num_minibatches
                update_epochs: Number of epochs to update the policy per update. One epoch is NUM_MINIBATCHES updates.
            """

            clip_eps = 0.2
            is_clip_eps_per_env = False
            max_grad_norm = 0.5
            num_steps_per_update_per_env = 128
            num_minibatches = 4
            update_epochs = 4

            gae_lambda = 0.95
            entropy_coefficient = 0.01
            value_coefficient = 0.5

        """
        Attributes:
            total_timesteps: Total time steps across all parallel environments.
        """

        seed = 1
        num_seeds = 2
        lr = 2e-3
        anneal_lr = True
        num_envs = 16
        total_timesteps = 2e7
        ppo_config = PPOConfig()

    @dataclass
    class NetworkConfig:
        fc_dim_size = 128
        gru_hidden_dim = 128

    @dataclass
    class DerivedValues:
        num_actors: int
        num_updates_per_env: int
        minibatch_size: int
        scaled_clip_eps: float

    env_config = EnvConfig()
    training_config = TrainingConfig()
    network = NetworkConfig()
    derived_values: DerivedValues = struct.field()

    @classmethod
    def create(cls) -> Config:
        env_config = cls.env_config
        train_config = cls.training_config
        num_actors = env_config.kwargs.num_agents * train_config.num_envs
        batch_size = num_actors * train_config.ppo_config.num_steps_per_update_per_env
        _derived_values = cls.DerivedValues(
            num_actors=num_actors,
            num_updates_per_env=int(
                train_config.total_timesteps
                // train_config.num_envs
                // train_config.ppo_config.num_steps_per_update_per_env
            ),
            minibatch_size=(batch_size // train_config.ppo_config.num_minibatches),
            scaled_clip_eps=(
                train_config.ppo_config.clip_eps / env_config.kwargs.num_agents
                if train_config.ppo_config.is_clip_eps_per_env
                else train_config.ppo_config.clip_eps
            ),
        )
        assert (
            _derived_values.num_updates_per_env > 0
        ), "Number of updates per environment must be greater than 0."
        # assert (
        #     batch_size % _derived_values.minibatch_size == 0
        # ), f"Minibatch size {_derived_values.minibatch_size} must divide batch size {batch_size}."

        return cls(derived_values=_derived_values)  # type: ignore

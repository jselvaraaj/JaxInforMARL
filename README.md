# Main scripts:

1. `algorithm/marl_ppo.py` for training Multi agent PPO on target MPE environment.
    - Note run this script as python module with `python -m algorithm/marl_ppo.py` for imports to work properly.
2. `envs/target_mpe_env.py`. This is the main class that defines the target MPE environment.
    - Also look at `envs/wrapper.py` for env wrappers.
3. `config/mappo_config.py`. This is the one and only file for changing config values to run experiments.
   Used python classes instead of yaml file to get auto complete and type checking and easier refactor when accessing
   and changing the structure of config.
4. `download_artifacts.py` for downloading trained actor weights from wandb.
5. `visualize_actor.py` for visualizing the trained actor in a local environment.

# Typical train and test flow:

1. Run the `train_with_gpu.ipynb` notebook in a colab with gpu.
    - Remember to set up the config in `WandbConfig` in `config/mappo_config.py` and change mode `online` to get wandb
      logging.
    - The artifacts are saved under the name "PPO_RNN_Runner_State"
2. Find the latest artifact name in wandb and download the artifact in local machine with `download_artifacts.py`
3. Visualize the actor with `visualize_actor.py` after changing the `artifact_name` variable in the block.
   `if __name__ == "__main__"`

# Note

It is recommended to first install either `requirements_jax_cpu.txt` or `requirements_jax_cuda.txt` before
`requirements.txt` since the packages in `requirements` will install a jax version for you.
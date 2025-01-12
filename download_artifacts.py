import wandb

# Change me
artifact_name = "josssdan/JaxInforMARL/PPO_RNN_Runner_State:v139"

api = wandb.Api()
artifact = api.artifact(artifact_name, type="model")
artifact_dir = artifact.download()  # 33

import wandb

api = wandb.Api()
artifact = api.artifact("josssdan/JaxInforMARL/PPO_RNN_Runner_State:v0", type="model")
artifact_dir = artifact.download()

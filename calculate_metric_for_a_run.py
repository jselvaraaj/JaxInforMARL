from functools import partial

import jax
import jax.numpy as jnp

from calculate_metric import get_stats_for_state
from visualize_actor import get_state_traj

artifact_version = "693"
num_episodes = 100
model_artifact_remote_name = (
    f"josssdan/JaxInforMARL/PPO_RNN_Runner_State:v{artifact_version}"
)

traj_batch, config, env = get_state_traj(
    model_artifact_remote_name, artifact_version, num_episodes=num_episodes
)

num_envs = config.training_config.num_envs
num_agents = config.env_config.env_kwargs.num_agents
num_steps = config.env_config.env_kwargs.max_steps

# reshaping so that the axis becomes num_env, num_steps, num_agents...

traj_batch = jax.tree.map(
    lambda x: x.reshape(num_steps, num_agents, num_envs, *x.shape[2:]), traj_batch
)
traj_batch = jax.tree.map(
    lambda x: jnp.swapaxes(x, 1, 2),
    traj_batch,
)
traj_batch = jax.tree.map(
    lambda x: jnp.swapaxes(x, 0, 1),
    traj_batch,
)

jax.tree.map(lambda x: x.shape, traj_batch)

# summing across all steps in episode and across all agents
total_reward = jnp.sum(traj_batch.reward, axis=(1, 2))
avg_reward_per_episode = jnp.average(total_reward).item()

done = jnp.swapaxes(
    traj_batch.done, 1, 2
)  # so that it becomes num_env, num_agents, num_steps
avg_goal_reach_time_in_episode_fraction = (jnp.argmax(done, axis=-1) + 1) / num_steps
agents_that_didnt_reach_goal = jnp.all(~done, axis=-1)
avg_goal_reach_time_in_episode_fraction = avg_goal_reach_time_in_episode_fraction.at[
    agents_that_didnt_reach_goal
].set(1)
avg_goal_reach_time_in_episode_fraction = jnp.average(
    avg_goal_reach_time_in_episode_fraction
).item()

reached_goal = jnp.any(done, axis=-1)
all_agents_reached_goal = jnp.all(reached_goal, axis=-1)

episode_percent_all_agents_reached_goals = jnp.average(all_agents_reached_goal) * 100
episode_percent_all_agents_reached_goals = (
    episode_percent_all_agents_reached_goals.item()
)


@partial(jax.jit, static_argnums=(0,))
def compute_stats_for_all_episode(env, state):
    compute_stats_for_every_step = jax.vmap(get_stats_for_state, in_axes=(None, 0))
    compute_all_stats = jax.vmap(compute_stats_for_every_step, in_axes=(None, 0))
    return compute_all_stats(env, state)


env_state = traj_batch.env_state.env_state
env_state = jax.tree.map(
    lambda x: x[:, :, 0], env_state
)  # take state from one agent since it will be the same for all agents

num_collisions, num_agent_died = compute_stats_for_all_episode(env, env_state)

avg_num_collision_across_all_episodes = jnp.average(num_collisions).item()
avg_num_deaths_across_all_episodes = jnp.average(num_agent_died).item()

print(
    avg_reward_per_episode,
    avg_goal_reach_time_in_episode_fraction,
    f"{episode_percent_all_agents_reached_goals} %",
    avg_num_collision_across_all_episodes,
)

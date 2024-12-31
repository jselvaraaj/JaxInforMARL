import jax
from jaxmarl.environments.mpe.simple import SimpleMPE

from envs.mpe_env import TargetMPEEnvironment
from envs.mpe_visualizer import MPEVisualizer

# Parameters + random keys
max_steps = 1000
key = jax.random.PRNGKey(0)
key, key_r = jax.random.split(key, 2)

num_agents = 3

# Instantiate environment
env = TargetMPEEnvironment(num_agents=num_agents)
MARL_env = SimpleMPE(num_agents=num_agents, num_landmarks=num_agents)
observation, state = env.reset(key_r)

state_seq = []
print("state", state)
print("action spaces", env.action_spaces)

for _ in range(max_steps):
    state_seq.append(state)
    key, key_act = jax.random.split(key)
    key_act = jax.random.split(key_act, env.num_agents)
    actions = {
        agent_label: env.action_space_for_agent(agent_label).sample(key_act[i])
        for i, agent_label in enumerate(env.agent_labels)
    }

    err, (obs, state, rew, dones, _) = env.step(key, state, actions)

viz = MPEVisualizer(env, state_seq)
viz.animate(None, view=True)

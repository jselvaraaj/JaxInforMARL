import jax
from jaxmarl.environments import SimpleMPE
from jaxmarl.environments.mpe import MPEVisualizer

# Parameters + random keys
max_steps = 25
key = jax.random.PRNGKey(0)
key, key_r = jax.random.split(key, 2)

num_agents = 3

# Instantiate environment
# env = TargetMPEEnvironment(num_agents=num_agents)
env = SimpleMPE(num_agents=num_agents, num_landmarks=num_agents)
# MARL_env = SimpleMPE(num_agents=num_agents, num_landmarks=num_agents)
observation, state = env.reset(key_r)

state_seq = []
print("state", state)
print("action spaces", env.action_spaces)

for _ in range(25):
    state_seq.append(state)
    key, key_act = jax.random.split(key)
    key_act = jax.random.split(key_act, env.num_agents)
    actions = {
        agent: env.action_space(agent).sample(key_act[i])
        for i, agent in enumerate(env.agents)
    }

    obs, state, rew, dones, _ = env.step_env(key, state, actions)

viz = MPEVisualizer(env, state_seq)
viz.animate(None, view=True)

# for _ in range(25):
#     state_seq.append(state)
#     key, key_act = jax.random.split(key)
#     key_act = jax.random.split(key_act, env.num_agents)
#     actions = {
#         agent_label: env.action_space_for_agent(agent_label).sample(key_act[i])
#         for i, agent_label in enumerate(env.agent_labels)
#     }
#
#     err, (obs, state, rew, dones, _) = env.step(key, state, actions)

# def to_JAXMARL_state(state: MPEState):
#     return State(
#         p_pos=state.entity_positions,
#         p_vel=state.entity_velocities,
#         c=None,
#         done=None,
#         step=state.step,
#     )
# state_seq = [to_JAXMARL_state(state) for state in state_seq]
# viz = MPEVisualizer(MARL_env, state_seq)
# viz.animate(None, view=True)

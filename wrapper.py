from envs.multiagent_env import MultiAgentEnv


class MARLEnvWrapper:
    """Base class for all MARL wrappers."""

    def __init__(self, env: MultiAgentEnv):
        self._env = env

    @property
    def num_agents(self):
        return self._env.num_agents

    @property
    def agent_labels(self):
        return self._env.agent_labels

    @property
    def action_spaces(self):
        return self._env.action_spaces

    @property
    def observation_spaces(self):
        return self._env.observation_spaces

    def action_space_for_agent(self, *args, **kwargs):
        return self._env.action_space_for_agent(*args, **kwargs)

    def observation_space_for_agent(self, *args, **kwargs):
        return self._env.observation_space_for_agent(*args, **kwargs)

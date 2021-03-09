from typing import List, Dict

import numpy as np

from coltra.buffers import Observation, Action
from coltra.envs.unity_envs import MultiAgentEnv


class ConstRewardEnv(MultiAgentEnv):
    def __init__(self, num_agents: int = 1):
        super().__init__()
        self.num_agents = num_agents
        self.active_agents = [f"Agent{i}" for i in range(num_agents)]

        self.obs_vector_size = 1
        self.action_vector_size = 1

    def reset(self, *args, **kwargs):
        zero_obs = Observation(vector=np.ones((1,), dtype=np.float32))
        return {agent_id: zero_obs for agent_id in self.active_agents}

    def step(self, actions: Dict[str, Action] = None):
        zero_obs = {agent_id: Observation(vector=np.ones((1,), dtype=np.float32)) for agent_id in self.active_agents}
        reward = {agent_id: np.float32(1.) for agent_id in self.active_agents}
        done = {agent_id: True for agent_id in self.active_agents}
        info = {"m_stat": np.array([1, 2, 3], dtype=np.float32)}
        return zero_obs, reward, done, info

    def render(self, mode='human'):
        return 0


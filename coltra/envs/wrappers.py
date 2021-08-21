from typing import Dict, Any

import numpy as np
import gym

from .base_env import MultiAgentEnv
from .subproc_vec_env import VecEnv, SubprocVecEnv
from coltra.buffers import Observation, Action


class MultiAgentWrapper(MultiAgentEnv):
    """
    A simple wrapper converting any instance of a single agent gym environment, into one compatible with my MultiAgentEnv approach.
    Might be useful for benchmarking.
    """

    def __init__(self, env: gym.Env, name: str = "agent", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.s_env = env
        self.name = name

        self.observation_space = self.s_env.observation_space
        self.action_space = self.s_env.action_space

        self.is_discrete_action = isinstance(self.s_env.action_space, gym.spaces.Discrete)

    def reset(self, *args, **kwargs):
        obs = self.s_env.reset()
        obs = Observation(vector=obs.astype(np.float32))
        return self._dict(obs)

    def step(self, action: Dict[str, Action], *args, **kwargs):
        action = action[self.name]
        if self.is_discrete_action:
            action = action.discrete
        else:
            action = action.continuous

        obs, reward, done, info = self.s_env.step(action, *args, **kwargs)
        obs = Observation(vector=obs.astype(np.float32))
        return self._dict(obs), self._dict(reward), self._dict(done), info

    def render(self, *args, **kwargs):
        return self.s_env.render(*args, **kwargs)

    def _dict(self, val: Any) -> Dict[str, Any]:
        return {self.name: val}


class MultiGymEnv(MultiAgentEnv):
    """
    A wrapper for environments that can be `gym.make`'d
    """

    def __init__(self, env_name: str, name: str = "agent", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.s_env = gym.make(env_name, **kwargs)
        self.name = name

        self.observation_space = self.s_env.observation_space
        self.action_space = self.s_env.action_space

        self.is_discrete_action = isinstance(self.s_env.action_space, gym.spaces.Discrete)

    def reset(self, *args, **kwargs):
        obs = self.s_env.reset()
        obs = Observation(vector=obs.astype(np.float32))
        return self._dict(obs)

    def step(self, action: Dict[str, Action], *args, **kwargs):
        action = action[self.name]
        if self.is_discrete_action:
            action = action.discrete
        else:
            action = action.continuous

        obs, reward, done, info = self.s_env.step(action, *args, **kwargs)
        obs = Observation(vector=obs.astype(np.float32))
        return self._dict(obs), self._dict(reward), self._dict(done), info

    def render(self, *args, **kwargs):
        return self.s_env.render(*args, **kwargs)

    def _dict(self, val: Any) -> Dict[str, Any]:
        return {self.name: val}

    @classmethod
    def get_env_creator(cls, env_name: str, *args, **kwargs):
        def _inner():
            env = gym.make(env_name, **kwargs)
            return MultiAgentWrapper(env)
        return _inner


    @classmethod
    def get_venv(cls, workers: int = 8, *args, **kwargs) -> VecEnv:
        venv = SubprocVecEnv([
            cls.get_env_creator(*args, **kwargs)
            for i in range(workers)
        ])
        return venv

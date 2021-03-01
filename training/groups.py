from typing import Dict

from agents import Agent
from buffers import Observation


class GroupAgent:
    """
    An
    """
    def __init__(self, agents: Dict[str, Agent]):
        self.agents = agents

    def act(self,
            obs_dict: Dict[str, Observation],
            deterministic: bool = False,
            get_value: bool = False):
        pass
import os

import numpy as np

import torch
from torch import nn, Tensor
from torch.distributions import Categorical, Normal

from models import BaseModel

from typing import Tuple, Optional, Dict

from utils import AgentDataBatch



class Agent:
    model: BaseModel

    def act(self, obs_batch: np.ndarray,
            state_batch: Tuple = (),
            deterministic: bool = False,
            get_value: bool = False) -> Tuple[np.ndarray, Tuple, Dict]:
        raise NotImplementedError

    def evaluate(self, obs_batch: Tensor, action_batch: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        raise NotImplementedError

    def cuda(self):
        if self.model is not None:
            self.model.cuda()

    def cpu(self):
        if self.model is not None:
            self.model.cpu()

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)

    def get_initial_state(self, requires_grad=True):
        return getattr(self.model, "get_initial_state", lambda *x, **xx: ())(requires_grad=requires_grad)


class CAgent(Agent):  # Continuous Agent
    model: BaseModel

    def __init__(self, model: BaseModel):
        self.model = model
        self.stateful = model.stateful

    def act(self, obs_batch: np.ndarray,  # [B, obs_size]
            state_batch: Tuple = (),
            deterministic: bool = False,
            get_value: bool = False) -> Tuple[np.ndarray, Tuple, Dict]:
        """Computes the action for an observation,
        passes along the state for recurrent models, and optionally the value"""
        obs_batch = torch.tensor(obs_batch)
        obs_batch.to(self.model.device)
        state_batch = tuple(s.to(self.model.device) for s in state_batch)

        action_distribution: Normal
        states: Tuple
        action: Tensor

        with torch.no_grad():
            action_distribution, states, extra_outputs = self.model(obs_batch, state_batch, get_value=get_value)

            if deterministic:
                actions = action_distribution.loc
            else:
                actions = action_distribution.rsample()

        if get_value:
            value = extra_outputs["value"]
            extra = {"value": value.squeeze(-1).cpu().numpy()}

        return actions.cpu().numpy(), states, extra

    def evaluate(self, obs_batch: Tensor,
                 action_batch: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Computes action logprobs, observation values and policy entropy for each of the (obs, action, hidden_state)
        transitions. Preserves all the necessary gradients.

        Args:
            obs_batch: observations collected with the collector
            action_batch: actions taken by the agent

        Returns:
            action_logprobs: tensor of action logprobs (batch_size, )
            values: tensor of observation values (batch_size, )
            entropies: tensor of entropy values (batch_size, )
        """
        obs_batch = obs_batch.to(self.model.device)
        action_batch = action_batch.to(self.model.device)
        # state_batch = data_batch['states']

        action_distribution, _, extra_outputs = self.model(obs_batch, get_value=True)
        values = extra_outputs["value"].sum(-1)
        action_logprobs = action_distribution.log_prob(action_batch).sum(-1)  # Sum across dimensions of the action
        entropies = action_distribution.entropy().sum(-1)

        return action_logprobs, values, entropies

    @staticmethod
    def load_agent(base_path: str,
                   weight_idx: Optional[int] = None,
                   fname: str = 'base_agent.pt',
                   weight_fname: str = 'weights') -> "CAgent":
        """
        Loads a saved model and wraps it as an Agent.
        The input path must point to a directory holding a pytorch file passed as fname
        """
        model: BaseModel = torch.load(os.path.join(base_path, fname))

        if weight_idx == -1:
            weight_idx = max([int(fname.split('_')[-1])  # Get the last agent
                              for fname in os.listdir(os.path.join(base_path, "saved_weights"))
                              if fname.startswith(weight_fname)])

        if weight_idx is not None:
            weights = torch.load(os.path.join(base_path, "saved_weights", f"{weight_fname}_{weight_idx}"))
            model.load_state_dict(weights)

        return CAgent(model)

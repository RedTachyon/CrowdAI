import os
from typing import Tuple, Optional, Dict

import numpy as np

import torch
from torch import nn, Tensor
from torch.distributions import Categorical, Normal

from coltra.models import BaseModel
from coltra.buffers import Observation, Action


class Agent:
    model: BaseModel

    def act(self, obs_batch: Observation,
            state_batch: Tuple = (),
            deterministic: bool = False,
            get_value: bool = False) -> Tuple[Action, Tuple, Dict]:
        raise NotImplementedError

    def evaluate(self, obs_batch: Observation, action_batch: Action) -> Tuple[Tensor, Tensor, Tensor]:
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

    def act(self, obs_batch: Observation,  # [B, ...]
            state_batch: Tuple = (),
            deterministic: bool = False,
            get_value: bool = False) -> Tuple[Action, Tuple, Dict]:
        """Computes the action for an observation,
        passes along the state for recurrent models, and optionally the value"""
        obs_batch = obs_batch.tensor(self.model.device)
        state_batch = tuple(s.to(self.model.device) for s in state_batch)

        action_distribution: Normal
        states: Tuple
        actions: Tensor

        with torch.no_grad():
            action_distribution, states, extra_outputs = self.model(obs_batch, state_batch, get_value=get_value)

            if deterministic:
                actions = action_distribution.loc
            else:
                actions = action_distribution.rsample()

        extra = {}
        if get_value:
            value = extra_outputs["value"]
            extra["value"] = value.squeeze(-1).cpu().numpy()

        return Action(continuous=actions.cpu().numpy()), states, extra

    def evaluate(self, obs_batch: Observation,
                 action_batch: Action) -> Tuple[Tensor, Tensor, Tensor]:
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
        obs_batch = obs_batch.tensor(self.model.device)
        action_batch = action_batch.tensor(self.model.device)
        # state_batch = data_batch['states']

        action_distribution, _, extra_outputs = self.model(obs_batch, get_value=True)
        values = extra_outputs["value"].sum(-1)
        # Sum across dimensions of the action
        action_logprobs = action_distribution.log_prob(action_batch.continuous).sum(-1)
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


class ConstantAgent(Agent):

    def __init__(self, action: np.array):
        self.action = action

    def act(self, obs_batch: Observation,
            state_batch: Tuple = (),
            deterministic: bool = False,
            get_value: bool = False) -> Tuple[Action, Tuple, Dict]:

        batch_size = obs_batch.batch_size

        return Action(continuous=np.tile(self.action, (batch_size, 1))), (), {"value": np.zeros((batch_size,))}

    def evaluate(self, obs_batch: Tensor, action_batch: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        zero = torch.zeros_like(action_batch)
        return zero, zero, zero

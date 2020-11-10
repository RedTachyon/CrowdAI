import numpy as np

import torch
from torch import nn, Tensor
from torch.distributions import Categorical, Normal

from models import BaseModel

from typing import Tuple

from utils import AgentDataBatch, tanh_norm, atanh_unnorm


class BaseAgent:
    """A base class for an agent, exposing the basic API methods"""

    def __init__(self, model: nn.Module):
        self.model = model
        self.stateful = False

    def compute_actions(self, obs_batch: Tensor,
                        state_batch: Tuple = (),
                        deterministic: bool = False) -> Tuple:

        raise NotImplementedError

    def evaluate_actions(self, data_batch: AgentDataBatch,
                         padded: bool = False):
        raise NotImplementedError

    def compute_single_action(self, obs: np.ndarray,
                              state: Tuple[Tensor, ...] = (),
                              deterministic: bool = False) -> Tuple[np.ndarray, float, Tuple]:
        """
        Computes the action for a single observation with the given hidden state. Breaks gradients.

        Args:
            obs: flat observation array in shape either
            state: tuple of state tensors of shape (1, lstm_nodes)
            deterministic: boolean, whether to always take the best action

        Returns:
            action, logprob of the action, new state vectors
        """
        obs = torch.tensor([obs])

        with torch.no_grad():
            action, logprob, new_state = self.compute_actions(obs, state, deterministic)

        return action.numpy().ravel(), logprob.item(), new_state

    def get_initial_state(self, requires_grad=True):
        return getattr(self.model, "get_initial_state", lambda *x, **xx: ())(requires_grad=requires_grad)

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)

    def cuda(self):
        if self.model is not None:
            self.model.cuda()

    def cpu(self):
        if self.model is not None:
            self.model.cpu()


class Agent(BaseAgent):
    """Agent variant for Continuous (Normal) action distributions"""
    model: BaseModel

    def __init__(self, model: BaseModel,
                 action_range: Tuple[Tensor, Tensor] = None):

        super().__init__(model)
        self.stateful = model.stateful
        self.action_range = tuple((x.view(1, -1) for x in action_range))

    def compute_actions(self, obs_batch: Tensor,
                        state_batch: Tuple = (),
                        deterministic: bool = False) -> Tuple[Tensor, Tensor, Tuple]:
        """
        Computes the action for a batch of observations with given hidden states. Breaks gradients.

        Args:
            obs_batch: observation array in shape either (batch_size, obs_size)
            state_batch: tuple of state tensors of shape (batch_size, lstm_nodes)
            deterministic: whether to always take the best action

        Returns:
            action, logprob of the action, new state vectors
        """
        action_distribution: Normal
        states: Tuple
        with torch.no_grad():
            action_distribution, states, extra_outputs = self.model(obs_batch, state_batch)

        action: Tensor
        if deterministic:
            actions = action_distribution.loc
        else:
            actions = action_distribution.sample()

        logprobs = action_distribution.log_prob(actions).sum(1)

        if self.action_range:
            a, b = self.action_range
            out_actions = tanh_norm(actions, a, b)
        else:
            out_actions = actions

        return out_actions.detach().cpu().numpy(), logprobs.detach().cpu().numpy(), states

    def evaluate_actions(self, data_batch: AgentDataBatch,
                         padded: bool = False) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Computes action logprobs, observation values and policy entropy for each of the (obs, action, hidden_state)
        transitions. Preserves all the necessary gradients.

        Args:
            data_batch: data collected from a Collector for this agent
            padded: whether the data is passed as 1D (not padded; [T*B, *]) or 2D (padded; [T, B, *]) tensor


        Returns:
            action_logprobs: tensor of action logprobs (batch_size, )
            values: tensor of observation values (batch_size, )
            entropies: tensor of entropy values (batch_size, )
        """
        obs_batch = data_batch['observations']
        action_batch = data_batch['actions']
        # state_batch = data_batch['states']

        if not padded:  # BP or non-recurrent
            if self.action_range:
                a, b = self.action_range
                action_batch = atanh_unnorm(action_batch, a, b)
            action_distribution, new_states, extra_outputs = self.model(obs_batch)
            values = extra_outputs["value"]
            action_logprobs = action_distribution.log_prob(action_batch).sum(1)
            values = values.view(-1)
            entropies = action_distribution.entropy().sum(1)

        else:  # padded == True, BPTT
            # TODO: Might not work, copied from DiscreteAgent; useful if I want BPTT, but slower, otherwise useless
            return NotImplementedError
            batch_size = obs_batch.size()[1]  # assume it's padded, so in [L, B, *] format
            state: Tuple[Tensor, ...] = self.get_initial_state()
            state = tuple(_state.repeat(batch_size, 1) for _state in state)
            entropies = []
            action_logprobs = []
            values = []
            # states_cache = [state]
            # breakpoint()

            for (obs, action) in zip(obs_batch, action_batch):
                action_distribution, new_state, extra_outputs = self.model(obs, state)
                value = extra_outputs["value"]
                action_logprob = action_distribution.log_prob(action)
                entropy = action_distribution.entropy()
                action_logprobs.append(action_logprob)
                values.append(value.T)
                entropies.append(entropy)

                state = new_state
                # states_cache.append(state)

            action_logprobs = torch.stack(action_logprobs)
            values = torch.cat(values, dim=0)
            entropies = torch.stack(entropies)

        return action_logprobs, values, entropies


class StillAgent(BaseAgent):
    """DEPRECATED
    might be worth reviving"""

    def __init__(self, model: nn.Module = None, action_value: int = 4):
        super().__init__(model)
        self.action_value = action_value

    def compute_actions(self, obs_batch: Tensor,
                        *args, **kwargs) -> Tuple[Tensor, Tensor, Tuple]:
        batch_size = obs_batch.shape[0]
        actions = torch.ones(batch_size) * self.action_value
        actions = actions.to(torch.int64)

        logprobs = torch.zeros(batch_size)

        states = ()

        return actions, logprobs, states

    def compute_single_action(self, obs: np.ndarray,
                              *args, **kwargs) -> Tuple[int, float, Tuple]:
        return self.action_value, 0., ()

    def evaluate_actions(self, data_batch: AgentDataBatch, padded: bool = False):
        batch_size = data_batch["observations"].shape[0]

        action_logprobs = torch.zeros(batch_size)
        values = torch.zeros(batch_size)
        entropies = torch.zeros(batch_size)

        return action_logprobs, values, entropies


class RandomAgent(BaseAgent):
    """DEPRECATED
    might be worth reviving"""

    def __init__(self, model: nn.Module = None, action_value: int = 4):
        super().__init__(model)
        self.action_value = action_value

    def compute_actions(self, obs_batch: Tensor,
                        *args, **kwargs) -> Tuple[Tensor, Tensor, Tuple]:
        batch_size = obs_batch.shape[0]
        actions = torch.randint(0, self.action_value + 1, (batch_size,))
        actions = actions.to(torch.int64)

        logprobs = torch.ones(batch_size) * np.log(1 / (self.action_value + 1))

        states = ()

        return actions, logprobs, states

    def compute_single_action(self, obs: np.ndarray,
                              *args, **kwargs) -> Tuple[int, float, Tuple]:
        return torch.randint(0, self.action_value + 1, (1,)).item(), 0., ()

    def evaluate_actions(self, data_batch: AgentDataBatch, padded: bool = False):
        batch_size = data_batch["observations"].shape[0]

        action_logprobs = torch.zeros(batch_size)
        values = torch.zeros(batch_size)
        entropies = torch.zeros(batch_size)

        return action_logprobs, values, entropies

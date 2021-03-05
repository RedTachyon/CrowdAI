from typing import Dict, Tuple, Callable, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.distributions import Distribution, Normal
from typarse import BaseConfig

from coltra.buffers import Observation
from coltra.utils import get_activation, get_initializer
from .network_utils import FCNetwork


class BaseModel(nn.Module):
    """
    A base class for any NN-based models, stateful or not, following a common convention:
    Each model in its forward pass takes an input and the previous recurrent state.
    If the state is not used in that specific model, it will just be discarded

    The output of each model is an action distribution, the next recurrent state,
    and a dictionary with any extra outputs like the value
    """

    def __init__(self):
        super().__init__()
        self._stateful = False
        # self.config = config
        self.device = 'cpu'

    # TO IMPLEMENT
    def forward(self, x: Observation,
                state: Tuple,
                get_value: bool) -> Tuple[Distribution, Tuple, Dict[str, Tensor]]:
        # Output: action_dist, state, {value, whatever else}
        raise NotImplementedError

    def value(self, x: Observation,
              state: Tuple) -> Tensor:
        raise NotImplementedError

    # Built-ins
    def get_initial_state(self, requires_grad=True) -> Tuple:
        return ()

    @property
    def stateful(self):
        return self._stateful

    def cuda(self, *args, **kwargs):
        super().cuda(*args, **kwargs)
        self.device = 'cuda'

    def cpu(self):
        super().cpu()
        self.device = 'cpu'


class MLPModel(BaseModel):
    """DEPRECATED (mostly)"""
    def __init__(self, config: Dict):
        super().__init__()

        class Config(BaseConfig):
            input_size: int = 94
            num_actions: int = 2
            activation: str = "leaky_relu"

            hidden_sizes: List[int] = [64, 64]
            separate_value: bool = False

            sigma0: float = 0.3

            initializer: str = "kaiming_uniform"

        Config.update(config)
        self.config = Config

        self.activation: Callable = get_activation(self.config.activation)
        self.separate_value = self.config.separate_value

        if not self.separate_value:
            self.policy_network = FCNetwork(
                input_size=self.config.input_size,
                output_sizes=[self.config.num_actions, 1],
                hidden_sizes=self.config.hidden_sizes,
                activation=self.config.activation,
                initializer=self.config.initializer,
                is_policy=[True, False]
            )

            self.value_network = None
        else:
            self.policy_network = FCNetwork(
                input_size=self.config.input_size,
                output_sizes=[self.config.num_actions],
                hidden_sizes=self.config.hidden_sizes,
                activation=self.config.activation,
                initializer=self.config.initializer,
                is_policy=True
            )

            self.value_network = FCNetwork(
                input_size=self.config.input_size,
                output_sizes=[1],
                hidden_sizes=self.config.hidden_sizes,
                activation=self.config.activation,
                initializer=self.config.initializer,
                is_policy=False
            )

        self.std = nn.Parameter(torch.tensor(self.config.sigma0) * torch.ones(1, self.config.num_actions))

        self.config = self.config.to_dict()  # Convert to a dictionary for pickling

    def forward(self, x: Observation,
                state: Tuple = (),
                get_value: bool = True) -> Tuple[Distribution, Tuple[Tensor, Tensor], Dict[str, Tensor]]:
        if self.separate_value:
            [action_mu] = self.policy_network(x.vector)
            value = None
        else:
            [action_mu, value] = self.policy_network(x.vector)

        action_distribution = Normal(loc=action_mu, scale=self.std)

        extra_outputs = {}

        if get_value:
            if self.separate_value:
                [value] = self.value_network(x.vector)

            extra_outputs["value"] = value

        return action_distribution, state, extra_outputs

    def value(self, x: Observation,
              state: Tuple = ()) -> Tensor:
        if self.separate_value:
            [value] = self.value_network(x.vector)
        else:
            [_, value] = self.policy_network(x.vector)

        return value


class FancyMLPModel(BaseModel):
    def __init__(self, config: Dict):
        super().__init__()

        class Config(BaseConfig):
            input_size: int = 94
            num_actions: int = 2
            activation: str = "leaky_relu"

            hidden_sizes: List[int] = [64, 64]

            initializer: str = "kaiming_uniform"

        Config.update(config)
        self.config = Config

        self.activation: Callable = get_activation(self.config.activation)

        # Create the policy network
        self.policy_network = FCNetwork(
            input_size=self.config.input_size,
            output_sizes=[self.config.num_actions, self.config.num_actions],
            hidden_sizes=self.config.hidden_sizes,
            activation=self.config.activation,
            initializer=self.config.initializer,
            is_policy=True
        )

        self.value_network = FCNetwork(
            input_size=self.config.input_size,
            output_sizes=[1],
            hidden_sizes=self.config.hidden_sizes,
            activation=self.config.activation,
            initializer=self.config.initializer,
            is_policy=False
        )

        self.config = self.config.to_dict()  # Convert to a dictionary for pickling

    def forward(self, x: Observation,
                state: Tuple = (),
                get_value: bool = True) -> Tuple[Distribution, Tuple[Tensor, Tensor], Dict[str, Tensor]]:

        [action_mu, action_std] = self.policy_network(x.vector)
        action_std = F.softplus(action_std - 0.5)

        action_distribution = Normal(loc=action_mu, scale=action_std)

        extra_outputs = {}

        if get_value:
            [value] = self.value_network(x.vector)
            extra_outputs["value"] = value

        return action_distribution, state, extra_outputs

    def value(self, x: Observation,
              state: Tuple = ()) -> Tensor:
        [value] = self.value_network(x.vector)
        return value

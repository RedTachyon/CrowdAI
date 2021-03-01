from typing import Dict, Tuple, Callable, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.distributions import Distribution, Normal
from typarse import BaseConfig

from coltra.buffers import Observation
from coltra.utils import get_activation, get_initializer


class BaseModel(nn.Module):
    """
    A base class for any NN-based models, stateful or not, following a common convention:
    Each model in its forward pass takes an input and the previous recurrent state.
    If the state is not used in that specific model, it will just be discarded

    The output of each model is an action distribution, the next recurrent state,
    and a dictionary with any extra outputs like the value
    """
    def __init__(self, config: Dict):
        super().__init__()
        self._stateful = False
        self.config = config
        self.device = 'cpu'

    def forward(self, x: Observation,
                state: Tuple,
                get_value: bool) -> Tuple[Distribution, Tuple, Dict[str, Tensor]]:
        # Output: action_dist, state, {value, whatever else}
        raise NotImplementedError

    def value(self, x: Observation,
              state: Tuple) -> Tensor:
        raise NotImplementedError

    def get_initial_state(self, requires_grad=True) -> Tuple:
        raise NotImplementedError

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
    def __init__(self, config: Dict):
        super().__init__(config)

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

        layer_sizes = [self.config.input_size] + self.config.hidden_sizes

        self.hidden_layers = nn.ModuleList([
            nn.Linear(in_size, out_size)
            for in_size, out_size in zip(layer_sizes, layer_sizes[1:])
        ])

        self.policy_head = nn.Linear(layer_sizes[-1], self.config.num_actions)

        if self.separate_value:
            self.value_layers = nn.ModuleList([
                nn.Linear(in_size, out_size)
                for in_size, out_size in zip(layer_sizes, layer_sizes[1:])
            ])

        self.value_head = nn.Linear(layer_sizes[-1], 1)

        self.std = nn.Parameter(torch.tensor(self.config.sigma0) * torch.ones(1, self.config.num_actions))

        if self.config.initializer:
            # If given an initializer, initialize all weights using it, and all biases with 0's
            initializer_ = get_initializer(self.config.initializer)

            for layer in self.hidden_layers:
                initializer_(layer.weight)
                nn.init.zeros_(layer.bias)

            if self.separate_value:
                for layer in self.value_layers:
                    initializer_(layer.weight)
                    nn.init.zeros_(layer.bias)

            initializer_(self.policy_head.weight)
            self.policy_head.weight.data /= 100.
            initializer_(self.value_head.weight)

            nn.init.zeros_(self.policy_head.bias)
            nn.init.zeros_(self.value_head.bias)

        self.config = self.config.to_dict()  # Convert to a dictionary for pickling

    def forward(self, x: Observation,
                state: Tuple = (),
                get_value: bool = True) -> Tuple[Distribution, Tuple[Tensor, Tensor], Dict[str, Tensor]]:
        x = x.vector  # Discard buffers etc.
        inp = x
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.activation(x)

        action_mu = self.policy_head(x)

        action_distribution = Normal(loc=action_mu, scale=self.std)

        extra_outputs = {}

        if get_value:
            if self.separate_value:
                x = inp
                for layer in self.value_layers:
                    x = layer(x)
                    x = self.activation(x)

            value = self.value_head(x)
            extra_outputs["value"] = value

        return action_distribution, state, extra_outputs

    def value(self, x: Observation,
              state: Tuple) -> Tensor:
        x = x.vector
        if self.separate_value:
            layers = self.value_layers
        else:
            layers = self.hidden_layers

        for layer in layers:
            x = layer(x)
            x = self.activation(x)

        value = self.value_head(x)

        return value

    def get_initial_state(self, requires_grad=True):
        return ()


class FancyMLPModel(BaseModel):
    def __init__(self, config: Dict):
        super().__init__(config)

        class Config(BaseConfig):
            input_size: int = 94
            num_actions: int = 2
            activation: str = "leaky_relu"

            hidden_sizes: List[int] = [64, 64]

            initializer: str = "kaiming_uniform"

        Config.update(config)
        self.config = Config

        self.activation: Callable = get_activation(self.config.activation)

        pi_layer_sizes = [self.config.input_size] + self.config.hidden_sizes

        self.pi_hidden_layers = nn.ModuleList([
            nn.Linear(in_size, out_size)
            for in_size, out_size in zip(pi_layer_sizes, pi_layer_sizes[1:])
        ])

        self.policy_head = nn.Linear(pi_layer_sizes[-1], self.config.num_actions)

        self.std_head = nn.Linear(pi_layer_sizes[-1], self.config.num_actions)

        v_layer_sizes = [self.config.input_size] + self.config.hidden_sizes  # Use identical policy/value networks

        self.v_hidden_layers = nn.ModuleList([
            nn.Linear(in_size, out_size)
            for in_size, out_size in zip(v_layer_sizes, v_layer_sizes[1:])
        ])

        self.value_head = nn.Linear(v_layer_sizes[-1], 1)

        if self.config.initializer:
            # If given an initializer, initialize all weights using it, and all biases with 0's
            initializer_ = get_initializer(self.config.initializer)

            for layer in self.pi_hidden_layers:
                initializer_(layer.weight)
                nn.init.zeros_(layer.bias)

            for layer in self.v_hidden_layers:
                initializer_(layer.weight)
                nn.init.zeros_(layer.bias)

            initializer_(self.policy_head.weight)
            self.policy_head.weight.data /= 100.
            initializer_(self.value_head.weight)
            initializer_(self.std_head.weight)
            self.std_head.weight.data /= 100.

            nn.init.zeros_(self.policy_head.bias)
            nn.init.zeros_(self.value_head.bias)
            nn.init.zeros_(self.std_head.bias)

        self.config = self.config.to_dict()  # Convert to a dictionary for pickling

    def forward(self, x: Observation,
                state: Tuple = (),
                get_value: bool = True) -> Tuple[Distribution, Tuple[Tensor, Tensor], Dict[str, Tensor]]:
        x = x.vector
        input_ = x

        for layer in self.pi_hidden_layers:
            x = layer(x)
            x = self.activation(x)

        action_mu = self.policy_head(x)

        action_std = self.std_head(x)
        action_std = F.softplus(action_std - 0.5)

        action_distribution = Normal(loc=action_mu, scale=action_std)

        extra_outputs = {}

        if get_value:
            x = input_
            for layer in self.v_hidden_layers:
                x = layer(x)
                x = self.activation(x)

            value = self.value_head(x)
            extra_outputs["value"] = value

        return action_distribution, state, extra_outputs

    def value(self, x: Observation,
              state: Tuple) -> Tensor:
        x = x.vector
        for layer in self.v_hidden_layers:
            x = layer(x)
            x = self.activation(x)

        value = self.value_head(x)

        return value

    def get_initial_state(self, requires_grad=True):
        return ()


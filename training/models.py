from typing import Dict, Tuple, Callable, Optional, Any

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.distributions import Distribution, Categorical, Normal

from utils import with_default_config, get_activation, get_initializer


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

    def forward(self, x: Tensor,
                state: Tuple) -> Tuple[Distribution, Tuple, Dict[str, Tensor]]:
        # Output: action_dist, state, {value, whatever else}
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

        default_config = {
            "input_size": 90,
            "num_actions": 2,
            "activation": "leaky_relu",

            "hidden_sizes": (64, 64),

            "sigma0": 0.1,

            "initializer": "kaiming_uniform",
        }
        self.config = with_default_config(config, default_config)

        input_size: int = self.config.get("input_size")
        num_actions: int = self.config.get("num_actions")
        hidden_sizes: Tuple[int] = self.config.get("hidden_sizes")
        self.activation: Callable = get_activation(self.config.get("activation"))

        layer_sizes = (input_size,) + hidden_sizes

        self.hidden_layers = nn.ModuleList([
            nn.Linear(in_size, out_size)
            for in_size, out_size in zip(layer_sizes, layer_sizes[1:])
        ])

        self.policy_head = nn.Linear(layer_sizes[-1], num_actions)
        self.value_head = nn.Linear(layer_sizes[-1], 1)

        self.std = nn.Parameter(torch.ones(1, num_actions) * self.config["sigma0"])

        if self.config["initializer"]:
            # If given an initializer, initialize all weights using it, and all biases with 0's
            initializer_ = get_initializer(self.config["initializer"])

            for layer in self.hidden_layers:
                initializer_(layer.weight)
                nn.init.zeros_(layer.bias)

            initializer_(self.policy_head.weight)
            self.policy_head.weight.data /= 100.
            initializer_(self.value_head.weight)

            nn.init.zeros_(self.policy_head.bias)
            nn.init.zeros_(self.value_head.bias)

    def forward(self, x: Tensor,
                state: Tuple = ()) -> Tuple[Distribution, Tuple[Tensor, Tensor], Dict[str, Tensor]]:
        
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.activation(x)

        action_mu = self.policy_head(x)
        value = self.value_head(x)

        action_distribution = Normal(loc=action_mu, scale=self.std)

        extra_outputs = {
            "value": value,
        }

        return action_distribution, state, extra_outputs

    def get_initial_state(self, requires_grad=True):
        return ()


class FancyMLPModel(BaseModel):
    def __init__(self, config: Dict):
        super().__init__(config)

        default_config = {
            "input_size": 90,
            "num_actions": 2,
            "activation": "leaky_relu",

            "pi_hidden_sizes": (64, 64),
            "v_hidden_sizes": (64, 64),

            "initializer": "kaiming_uniform",
        }
        self.config = with_default_config(config, default_config)

        input_size: int = self.config.get("input_size")
        num_actions: int = self.config.get("num_actions")
        pi_hidden_sizes: Tuple[int] = self.config.get("pi_hidden_sizes")
        v_hidden_sizes: Tuple[int] = self.config.get("v_hidden_sizes")
        self.activation: Callable = get_activation(self.config.get("activation"))

        pi_layer_sizes = (input_size,) + pi_hidden_sizes

        self.pi_hidden_layers = nn.ModuleList([
            nn.Linear(in_size, out_size)
            for in_size, out_size in zip(pi_layer_sizes, pi_layer_sizes[1:])
        ])

        self.policy_head = nn.Linear(pi_layer_sizes[-1], num_actions)

        self.std_head = nn.Linear(pi_layer_sizes[-1], num_actions)

        v_layer_sizes = (input_size,) + v_hidden_sizes

        self.v_hidden_layers = nn.ModuleList([
            nn.Linear(in_size, out_size)
            for in_size, out_size in zip(v_layer_sizes, v_layer_sizes[1:])
        ])

        self.value_head = nn.Linear(v_layer_sizes[-1], 1)

        if self.config["initializer"]:
            # If given an initializer, initialize all weights using it, and all biases with 0's
            initializer_ = get_initializer(self.config["initializer"])

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

            nn.init.zeros_(self.policy_head.bias)
            nn.init.zeros_(self.value_head.bias)
            nn.init.zeros_(self.std_head.bias)

    def forward(self, x: Tensor,
                state: Tuple = ()) -> Tuple[Distribution, Tuple[Tensor, Tensor], Dict[str, Tensor]]:

        input_ = x

        for layer in self.pi_hidden_layers:
            x = layer(x)
            x = self.activation(x)

        action_mu = self.policy_head(x)

        action_std = self.std_head(x)
        action_std = F.softplus(action_std)  # TODO: add a negative offset here?

        x = input_
        for layer in self.v_hidden_layers:
            x = layer(x)
            x = self.activation(x)

        value = self.value_head(x)

        action_distribution = Normal(loc=action_mu, scale=action_std)

        extra_outputs = {
            "value": value,
        }

        return action_distribution, state, extra_outputs

    def get_initial_state(self, requires_grad=True):
        return ()


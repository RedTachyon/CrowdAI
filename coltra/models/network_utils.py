from typing import List, Callable, Union

import torch
from torch import nn, Tensor

from coltra.utils import get_activation, get_initializer


class FCNetwork(nn.Module):
    def __init__(self,
                 input_size: int,
                 output_sizes: List[int],
                 hidden_sizes: List[int],
                 activation: str,
                 initializer: str = "kaiming_uniform",
                 is_policy: Union[bool, List[bool]] = False):
        super().__init__()

        self.activation: Callable = get_activation(activation)
        layer_sizes = [input_size] + hidden_sizes

        self.hidden_layers = nn.ModuleList([
            nn.Linear(in_size, out_size)
            for in_size, out_size in zip(layer_sizes, layer_sizes[1:])
        ])

        self.heads = [
            nn.Linear(layer_sizes[-1], output_size)
            for output_size in output_sizes
        ]

        if initializer:
            # If given an initializer, initialize all weights using it, and all biases with 0's
            initializer_ = get_initializer(initializer)

            for layer in self.hidden_layers:
                initializer_(layer.weight)
                nn.init.zeros_(layer.bias)

            for i, head in enumerate(self.heads):
                initializer_(head.weight)
                if isinstance(is_policy, list):
                    divide = is_policy[i]
                else:
                    divide = is_policy
                if divide: head.weight.data /= 100.
                nn.init.zeros_(head.bias)

    def forward(self, x: Tensor) -> List[Tensor]:
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.activation(x)

        return [head(x) for head in self.heads]

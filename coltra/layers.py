from typing import Dict, Any, List, Type

import torch
from torch import Tensor
from torch import nn
from typarse import BaseConfig

from coltra.utils import get_activation_module, get_initializer


class RelationLayer(nn.Module):
    """DEPRECATED  TODO: Fix this to work with crowds
    Will need rewriting to accommodate crowds
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        class Config(BaseConfig):
            emb_size: int = 4
            rel_hiddens: List[int] = [16, 16]
            mlp_hiddens: List[int] = [16, ]
            activation: str = "leaky_relu"
            initializer: str = "xavier_uniform"

        Config.update(config)
        self.config = Config

        self.activation: Type[nn.Module] = get_activation_module(self.config["activation"])
        self.goals = self.config["num_goals"]

        # Needs to be generalized
        self.own_embedding = nn.Parameter(torch.randn(self.config["emb_size"]) / 10., requires_grad=True)
        self.agent_embedding = nn.Parameter(torch.randn(self.config["emb_size"]) / 10., requires_grad=True)
        self.subgoal_embedding = nn.Parameter(torch.randn(self.config["emb_size"]) / 10., requires_grad=True)
        self.goal_embedding = nn.Parameter(torch.randn(self.config["emb_size"]) / 10., requires_grad=True)
        self.goal2_embedding = nn.Parameter(torch.randn(self.config["emb_size"]) / 10., requires_grad=True)

        rel_sizes = (2 * (self.config["emb_size"] + 3),) + self.config["rel_hiddens"]
        mlp_sizes = (self.config["rel_hiddens"][-1],) + self.config["mlp_hiddens"]

        _relation_layers = tuple(
            (nn.Linear(in_size, out_size), self.activation())
            for in_size, out_size in zip(rel_sizes, rel_sizes[1:])
        )
        self.relation_layers = nn.Sequential(*sum(_relation_layers, ()))  # flatten the tuple of tuples with sum

        _mlp_layers = tuple(
            (nn.Linear(in_size, out_size), self.activation())
            for in_size, out_size in zip(mlp_sizes, mlp_sizes[1:])
        )

        self.mlp_layers = nn.Sequential(*sum(_mlp_layers, ()))

        if self.config["initializer"]:
            initializer_ = get_initializer(self.config["initializer"])
            for layer in self.relation_layers:
                if hasattr(layer, "weight"):
                    initializer_(layer.weight)
                if hasattr(layer, "bias"):
                    nn.init.zeros_(layer.bias)

            for layer in self.mlp_layers:
                if hasattr(layer, "weight"):
                    initializer_(layer.weight)
                if hasattr(layer, "bias"):
                    nn.init.zeros_(layer.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the model"""

        object_size = 3  # (x, y, flag)
        input_size = x.size()  # = (batch, [seq,] num_obj*3)
        num_objects = input_size[-1] // object_size
        x = x.view(input_size[:-1] + (num_objects, object_size))  # (batch, [seq,] num_obj, 3)

        non_subgoals = 2 + self.config["num_goals"]

        goal_embeddings = () + \
                          ((self.goal_embedding,) if self.goals > 0 else ()) + \
                          ((self.goal2_embedding,) if self.goals > 1 else ())

        # num_objects - non_subgoals is the number of subgoals in the input, so all objects except agents and goals
        subgoals_tuple = tuple(self.subgoal_embedding for _ in range(num_objects - non_subgoals))
        embeddings = torch.stack((self.own_embedding, self.agent_embedding)
                                 + subgoals_tuple
                                 + goal_embeddings, dim=0)  # (num_obj, emb_size)

        # (1, [1, ] num_obj, emb_size)
        embeddings = embeddings.view(tuple(1 for _ in input_size[:-1]) + embeddings.size())

        # (batch, [seq, ] num_obj, emb_size)
        embeddings = embeddings.repeat(input_size[:-1] + (1, 1))

        # (batch, [seq, ] num_obj, emb_size+3)
        inputs = torch.cat((x, embeddings), dim=-1)

        # (batch, [seq, ] emb_size+3); agent's own embedding
        own_input = inputs[..., 0:1, :]

        # (batch, [seq, ] num_obj, emb_size+3); agent's own embedding repeated so there's one for each object
        own_input = own_input.repeat(tuple(1 for _ in input_size[:-1]) + (num_objects, 1))

        # (batch, [seq, ] num_obj, 2*(emb_size+3) )
        full_input = torch.cat((own_input, inputs), dim=-1)

        rel_outputs = self.relation_layers(full_input)
        rel_outputs = torch.sum(rel_outputs, dim=-2)

        final_output = self.mlp_layers(rel_outputs)
        return final_output

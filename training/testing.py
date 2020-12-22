import time
from typing import Dict

import torch
from torch import nn, tensor, Tensor
import torch.multiprocessing as mp
import numpy as np

from agents import Agent
from collectors import collect_crowd_data, collect_parallel_unity, _collection_worker
from environments import UnitySimpleCrowdEnv
from models import MLPModel, FancyMLPModel
from policy_optimization import CrowdPPOptimizer

from utils import concat_crowd_batch, concat_batches, DataBatch, concat_metrics, discount_td_rewards

from torch.distributions import Normal

import matplotlib.pyplot as plt

from parallel import SubprocVecEnv


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(tensor([0.]))

    def forward(self, x: Tensor):
        return self.w + x


def _test_worker(model: Model):
    model.w.data += 1.

def get_env_creator(*args, **kwargs):
    def _inner():
        env = UnitySimpleCrowdEnv(*args, **kwargs)
        env.engine_channel.set_configuration_parameters(time_scale=100)
        return env
    return _inner



if __name__ == '__main__':

    venv = SubprocVecEnv(
        [get_env_creator(file_name="builds/1-random-16-mac.app", no_graphics=True, worker_id=i, seed=i)
         for i in range(8)]
    )

    agent = Agent(MLPModel({
        "input_size": 72,
    }))

    data, metrics = collect_crowd_data(agent, venv, 500, disable_tqdm=False)

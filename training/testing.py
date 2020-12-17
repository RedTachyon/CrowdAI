import time
from typing import Dict

import torch
import torch.multiprocessing as mp

from agents import Agent
from collectors import collect_crowd_data, collect_parallel_unity, _collection_worker
from environments import UnitySimpleCrowdEnv
from models import MLPModel, FancyMLPModel
from policy_optimization import CrowdPPOptimizer

from utils import concat_crowd_batch, concat_batches, DataBatch, concat_metrics, discount_td_rewards

from torch.distributions import Normal

import matplotlib.pyplot as plt


if __name__ == '__main__':

    agent = Agent(FancyMLPModel({
        "input_size": 8,
        "separate_value": True
    }))

    # batch, metrics = collect_parallel_unity(8, 8, agent=agent, env_path="builds/1-random-16-mac.app", num_steps=500)
    batch, metrics = _collection_worker(agent, 0, "builds/1-random-16-mac.app", 500)
    batch = concat_crowd_batch(batch)
    # batch = discount_td_rewards(batch, gamma=1)

    # ppo = CrowdPPOptimizer(agent, {})
    # ppo.train_on_data(batch)
from typing import Sequence

import torch
import numpy as np
from models import MLPModel, FancyMLPModel
from agents import Agent
from collectors import Memory, CrowdCollector
from environments import UnityCrowdEnv, UnitySimpleCrowdEnv
from policy_optimization import CrowdPPOptimizer
from trainers import PPOCrowdTrainer

from tqdm import tqdm, trange
from mlagents_envs.environment import UnityEnvironment

from utils import transpose_batch, concat_batches, concat_crowd_batch, tanh_norm, atanh_unnorm

from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

import seaborn as sns
import matplotlib.pyplot as plt

from utils import discount_rewards_to_go, get_episode_lens


print("Load the environment, please")
env = UnitySimpleCrowdEnv(file_name=None)
print("Environment loaded")

env.engine_channel.set_configuration_parameters(time_scale=100)


action_range = (
    torch.tensor([-.3, -1.]),
    torch.tensor([ 1.,  1.])
)

model = MLPModel({
    "input_size": 94,
})

agent = Agent(model, action_range=action_range)

collector = CrowdCollector(agent, env)

data, metrics = collector.collect_data(500, disable_tqdm=False)

data = concat_crowd_batch(data)

logprobs, values, entropies = agent.evaluate_actions(data)

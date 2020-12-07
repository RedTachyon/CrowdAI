from typing import Sequence

import torch
import numpy as np
from models import MLPModel, FancyMLPModel
from agents import Agent
from collectors import Memory, CrowdCollector, collect_crowd_data
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


# TODO: Observation - value loss seems to overtake policy loss later in the training; need to make it so that they don't
#  bother one another, probably by using separate policy/value networks. Maybe just decrease value coeff?

# print("Load the environment, please")
# env = UnitySimpleCrowdEnv(file_name=None)
# print("Environment loaded")
env = UnitySimpleCrowdEnv(file_name="Test.app", no_graphics=True, worker_id=0)

env2 = UnitySimpleCrowdEnv(file_name="Test.app", no_graphics=True, worker_id=1)

env.engine_channel.set_configuration_parameters(time_scale=100)


action_range = (
    torch.tensor([-.3, -1.]),
    torch.tensor([ 1.,  1.])
)

model = MLPModel({
    "input_size": 94,
})

agent = Agent(model, action_range=action_range)

data, metrics = collect_crowd_data(agent, env, 500, disable_tqdm=False)

data = concat_crowd_batch(data)

logprobs, values, entropies = agent.evaluate_actions(data)

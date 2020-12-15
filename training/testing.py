import time
from typing import Dict

import torch
import torch.multiprocessing as mp

from agents import Agent
from collectors import collect_crowd_data, collect_parallel_unity
from environments import UnitySimpleCrowdEnv
from models import MLPModel

from utils import concat_crowd_batch, concat_batches, DataBatch, concat_metrics, discount_td_rewards


rewards_batch = torch.rand(10)
values_batch = torch.rand(10)
dones_batch = torch.tensor([True if (i+1) % 5 == 0 else False for i in range(10)])

rets, advs = discount_td_rewards(rewards_batch, values_batch, dones_batch)
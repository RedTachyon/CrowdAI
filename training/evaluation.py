import copy
import os
from collections import defaultdict
from typing import List, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import sem
from torch import Tensor
from tqdm import tqdm

from agents import Agent
from environments import MultiAgentEnv
from models import BaseModel
from collectors import CrowdCollector
from utils import get_episode_lens


def load_agent(base_path: str,
               fname: str = 'base_agent.pt',
               weight_idx: Optional[int] = None,
               weight_fname: str = 'weights') -> Agent:
    """
    Loads a saved model and wraps it as an Agent.
    The input path must point to a directory holding a pytorch file passed as fname
    """
    model: BaseModel = torch.load(os.path.join(base_path, fname))

    if weight_idx == -1:
        weight_idx = max([int(fname.split('_')[-1])  # Get the last agent
                          for fname in os.listdir(os.path.join(base_path, "saved_weights"))
                          if fname.startswith(weight_fname)])

    if weight_idx is not None:
        weights = torch.load(os.path.join(base_path, "saved_weights", f"{weight_fname}_{weight_idx}"))
        model.load_state_dict(weights)

    return Agent(model)


def load_weights(base_path: str,
                 base_name: str = 'weights',
                 start: int = 0,
                 end: Optional[int] = None) -> List[Dict[str, Tensor]]:
    """
    Loads a list of model weights from a certain path
    """
    base_path = os.path.join(base_path, "saved_weights")
    weight_paths = sorted(
        [
            os.path.join(base_path, fname)
            for fname in os.listdir(base_path)
            if fname.startswith(base_name)
        ],
        key=lambda x: int(x.split('_')[-1])
    )
    weight_paths = weight_paths[start:end]

    return [torch.load(path) for path in weight_paths]


def load_agent_population(base_path: str,
                          agent_fname: str = 'base_agent.pt',
                          weight_fname: str = 'weights',
                          start: int = 0,
                          end: Optional[int] = None) -> Tuple[Agent, List[Dict[str, Tensor]], np.ndarray]:
    """
    Convenience function to load an agent along with its historic weights.
    The files in an appropriate format are generated in the sampling trainer, in the tensorboard log directory.

    Args:
        base_path: path to the directory holding the saved agent and weights; usually tensorboard logdir
        agent_fname: filename of the agent file
        weight_fname: beginning of the weight filenames
        start: starting weight index that should be loaded; assumes
        end: last weight index that should be loaded

    Returns:

    """
    base_agent, _, returns = load_agent(base_path=base_path, fname=agent_fname)
    weights = load_weights(base_path=base_path, base_name=weight_fname, start=start, end=end)

    return base_agent, weights, returns

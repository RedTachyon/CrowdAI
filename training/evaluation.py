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
from collectors import Collector
from utils import get_episode_lens


def load_agent(base_path: str,
               fname: str = 'base_agent.pt',
               weight_idx: Optional[int] = None,
               weight_fname: str = 'weights') -> Tuple[Agent, Dict[str, Tensor], np.ndarray]:
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
    else:
        weights = model.state_dict()

    returns = np.array(torch.load(os.path.join(base_path, "returns.pt")))

    return Agent(model), copy.deepcopy(weights), returns


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


def evaluate_training(env: MultiAgentEnv,
                      main_agent: Agent,
                      main_weights: Dict[str, Tensor],
                      other_agent: Agent,
                      other_population: List[Dict[str, Tensor]],
                      det: Optional[Dict[str, bool]] = None,
                      num_runs: int = 100,
                      use_tom_iter: bool = False,
                      last_step: int = 100,
                      step_normalization: int = 100,
                      step_freq: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Performs ad-hoc evaluation of an agent with a population of other agents.
    For every 5th agent from the other population, it's evaluated with itself giving the x-value,
    and evaluated with the main agent, giving the y-value."""
    if det is None:
        det = defaultdict(lambda: False)

    x_all = []
    y_all = []

    self_agents = {
        "Agent0": other_agent,
        "Agent1": other_agent
    }

    mix_agents = {
        "Agent0": main_agent,
        "Agent1": other_agent,
    }

    # Two collectors since the two agents might have different architectures.
    self_collector = Collector(self_agents, env, {})
    mix_collector = Collector(mix_agents, env, {})

    LAST_STEP = np.array([last_step], dtype=np.float32) / step_normalization

    for i, weights in tqdm(enumerate(other_population), total=len(other_population)):
        current_step = np.array([i * step_freq], dtype=np.float32) / step_normalization
        # Evaluate with itself
        weights_self = {
            "Agent0": weights,
            "Agent1": weights
        }
        self_collector.update_agent_state_dict(weights_self)

        if use_tom_iter:
            tom_dict = {"Agent0": current_step, "Agent1": current_step}
        else:
            tom_dict = None

        self_batch = self_collector.collect_data(num_episodes=num_runs,
                                                 deterministic=det,
                                                 tom=tom_dict)

        reward_batch = self_batch["rewards"]["Agent0"]
        done_batch = self_batch["dones"]["Agent0"]

        ep_lens = get_episode_lens(done_batch)

        rewards_x = torch.tensor([torch.sum(rewards) for rewards in torch.split(reward_batch, ep_lens)]).numpy()

        x_all.append(rewards_x)

        weights_main = {
            "Agent0": main_weights,
            "Agent1": weights
        }

        if use_tom_iter:
            tom_dict = {"Agent0": current_step, "Agent1": LAST_STEP}
        else:
            tom_dict = None

        mix_collector.update_agent_state_dict(weights_main)
        other_batch = mix_collector.collect_data(num_episodes=num_runs,
                                                 deterministic=det,
                                                 tom=tom_dict)

        reward_batch = other_batch["rewards"]["Agent0"]
        done_batch = other_batch["dones"]["Agent0"]

        ep_lens = get_episode_lens(done_batch)

        rewards_y = torch.tensor([torch.sum(rewards) for rewards in torch.split(reward_batch, ep_lens)]).numpy()

        y_all.append(rewards_y)

    x_all = np.array(x_all)
    y_all = np.array(y_all)

    return x_all, y_all


def evaluate_populations(main_population_path: str,
                         other_population_path: str,
                         env: MultiAgentEnv,
                         main_idx: int = -1,
                         main_name: str = "base_agent.pt",
                         other_start: int = 0,
                         other_end: int = 100,
                         other_freq: int = 5,
                         other_name: str = "base_agent.pt",
                         other_wname: str = "weights",
                         num_runs: int = 1000,
                         use_tom_iter: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function for the population evaluation, handling the loading of agents with settable names etc.
    """
    main_agent, main_weights, _ = load_agent(main_population_path, main_name, weight_idx=main_idx)

    other_agent, old_weights, _ = load_agent_population(other_population_path,
                                                        other_name,
                                                        other_wname,
                                                        start=other_start, end=other_end)

    x_all, y_all = evaluate_training(env,
                                     main_agent=main_agent,
                                     main_weights=main_weights,
                                     other_agent=other_agent,
                                     other_population=old_weights[::other_freq],
                                     det=defaultdict(lambda: False),
                                     num_runs=num_runs,
                                     use_tom_iter=use_tom_iter,
                                     step_freq=other_freq)

    return x_all, y_all


def plot_adhoc_line(fig: plt.Figure,
                    x: np.ndarray,
                    y: np.ndarray,
                    label: str,
                    **plt_kwargs) -> plt.Figure:
    """
    Draws a single ad-hoc evaluation line.

    Args:
        fig: matplotlib figure with one axes to be drawn on
        x: self-evaluation results, shape (agents, trials)
        y: cross-evaluation results, shape (agents, trials)
        label: label for the legend

    Returns:

    """
    ax: plt.Axes = fig.gca()

    ax.errorbar(x=np.mean(x, axis=1),
                y=np.mean(y, axis=1),
                xerr=sem(x, axis=1),
                yerr=sem(y, axis=1),
                label=label,
                **plt_kwargs)

    return fig


def plot_lines(xs: List[np.ndarray],
               ys: List[np.ndarray],
               labels: List[str]) -> plt.Figure:
    """
    Plots multiple adhoc lines with proper labels

    Args:
        xs: list of self-evaluation results
        ys: list of cross-evaluation results
        labels: list of labels

    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots()
    for (x, y, label) in zip(xs, ys, labels):
        plot_adhoc_line(fig, x, y, label)
    ax.legend()
    return fig

from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard import SummaryWriter
from typarse import BaseConfig

from agents import Agent
from preprocessors import simple_padder
from utils import with_default_config, get_optimizer, DataBatch, Timer, DataBatchT, transpose_batch, AgentDataBatch, \
    discount_rewards_to_go, masked_mean, get_episode_lens, write_dict, batch_to_gpu, concat_crowd_batch, \
    discount_td_rewards


class CrowdPPOptimizer:  # TODO: rewrite this with minibatches
    """
    An optimizer for a single homogeneous crowd agent. Estimates the gradient from the whole batch (no SGD).
    """

    def __init__(self, agent: Agent,
                 config: Dict[str, Any]):

        self.agent = agent

        class Config(BaseConfig):
            gamma: float = 0.95
            ppo_steps: int = 5
            eps: float = 0.1
            target_kl: float = 0.01

            value_steps: int = 10

            entropy_coeff: float = 0.1
            entropy_decay_time: float = 100.
            min_entropy: float = 0.0001

            use_gpu: bool = False

            optimizer: str = "adam"

            class OptimizerKwargs(BaseConfig):
                lr: float = 1e-4
                betas: Tuple[float, float] = (0.9, 0.999)
                eps: float = 1e-7
                weight_decay: float = 0.0
                amsgrad: bool = False

        Config.update(config)
        self.config = Config

        # default_config = {
        #     # GD settings
        #     "optimizer": "adam",
        #     "optimizer_kwargs": {
        #         "lr": 1e-3,
        #         "betas": (0.9, 0.999),
        #         "eps": 1e-7,
        #         "weight_decay": 0,
        #         "amsgrad": False
        #     },
        #     "gamma": 0.95,  # Discount factor
        #
        #     # PPO settings
        #     "ppo_steps": 5,
        #     "eps": 0.1,  # PPO clip parameter
        #     "target_kl": 0.01,  # KL divergence limit
        #
        #     # Value estimator settings
        #     "value_steps": 10,
        #
        #     "entropy_coeff": 0.1,
        #     "entropy_decay_time": 100,  # How many steps to decrease entropy to 0.1 of the original value
        #     "min_entropy": 0.0001,  # Minimum value of the entropy bonus - use this to disable decay
        #
        #     # GPU
        #     "use_gpu": False,
        # }
        # self.config = with_default_config(config, default_config)

        self.policy_optimizer = get_optimizer(self.config.optimizer)(agent.model.parameters(),
                                                                     **self.config.OptimizerKwargs.to_dict())

        self.value_optimizer = get_optimizer(self.config.optimizer)(agent.model.parameters(),
                                                                    **self.config.OptimizerKwargs.to_dict())

        self.gamma: float = self.config.gamma
        self.eps: float = self.config.eps

    def train_on_data(self, data_batch: DataBatch,
                      step: int = 0,
                      writer: Optional[SummaryWriter] = None) -> Dict[str, float]:
        """
        Performs a single update step with PPO on the given batch of data.

        Args:
            data_batch: DataBatch, dictionary
            step:
            writer:

        Returns:

        """
        metrics = {}
        timer = Timer()

        # data_batch: DataBatchT = transpose_batch(data_batch)

        entropy_coeff = max(
            self.config.entropy_coeff * 0.1 ** (step / self.config.entropy_decay_time),
            self.config.min_entropy
        )

        agent_id = "crowd"
        agent = self.agent

        ####################################### Unpack and prepare the data #######################################
        # agent_batch: AgentDataBatch = concat_crowd_batch(data_batch)
        agent_batch = data_batch

        if self.config.use_gpu:
            agent_batch = batch_to_gpu(agent_batch)
            agent.cuda()

        # if self.config["pad_sequences"]:
        #     agent_batch, mask = simple_padder(agent_batch)
        #     ep_lens = tuple(mask.sum(0).cpu().numpy())
        # else:
        # mask = torch.ones_like(agent_batch['rewards'])
        # ep_lens = None

        # Unpacking the data for convenience

        # Compute discounted rewards to go
        # add the 'returns' and 'advantages' keys, and removes last position from other fields
        agent_batch = discount_td_rewards(agent_batch,
                                          gamma=self.gamma,
                                          lam=0.95)

        # obs_batch = agent_batch['observations']
        # action_batch = agent_batch['actions']  # actions taken
        reward_batch = agent_batch['rewards']  # rewards obtained
        old_logprobs_batch = agent_batch['logprobs']  # logprobs of taken actions
        done_batch = agent_batch['dones']  # whether the step is the end of an episode
        # state_batch = agent_batch['states']  # hidden LSTM state
        discounted_batch = agent_batch['returns']
        advantages_batch = agent_batch['advantages']

        # Evaluate actions to have values that require gradients
        # logprob_batch, value_batch, entropy_batch = agent.evaluate_actions(agent_batch)

        # Move data to GPU if applicable
        if self.config.use_gpu:
            discounted_batch = discounted_batch.cuda()

        # breakpoint()
        # Compute the normalized advantage
        # advantages_batch = (discounted_batch - value_batch).detach()
        advantages_batch = (advantages_batch - advantages_batch.mean())
        advantages_batch = advantages_batch / (torch.sqrt(torch.mean(advantages_batch ** 2) + 1e-8))

        # Initialize metrics
        kl_divergence = 0.
        ppo_step = -1
        value_loss = torch.tensor(0)
        policy_loss = torch.tensor(0)
        loss = torch.tensor(0)
        entropy_batch = torch.tensor([0])

        # Start a timer
        timer.checkpoint()

        for ppo_step in range(self.config.ppo_steps):
            # Evaluate again after the PPO step, for new values and gradients
            logprob_batch, value_batch, entropy_batch = agent.evaluate_actions(agent_batch)
            # Compute the KL divergence for early stopping
            kl_divergence = torch.mean(old_logprobs_batch - logprob_batch).item()
            if np.isnan(kl_divergence): breakpoint()
            if kl_divergence > self.config.target_kl:
                break

            ######################################### Compute the loss #############################################
            # Surrogate loss
            prob_ratio = torch.exp(logprob_batch - old_logprobs_batch)
            surr1 = prob_ratio * advantages_batch
            # surr2 = torch.clamp(prob_ratio, 1. - self.eps, 1 + self.eps) * advantages_batch
            surr2 = torch.where(advantages_batch > 0,
                                (1 + self.eps) * advantages_batch,
                                (1 - self.eps) * advantages_batch)

            policy_loss = -torch.min(surr1, surr2)

            loss = (
                    policy_loss.mean()
                    - (entropy_coeff * entropy_batch.mean())
            )

            ############################################# Update step ##############################################
            self.policy_optimizer.zero_grad()
            loss.backward()

            self.policy_optimizer.step()

        for value_step in range(self.config.value_steps):
            _, value_batch, _ = agent.evaluate_actions(agent_batch)

            value_loss = (value_batch - discounted_batch) ** 2

            loss = value_loss.mean()

            self.value_optimizer.zero_grad()
            loss.backward()
            self.value_optimizer.step()

        ############################################## Collect metrics #############################################

        agent.cpu()

        # Training-related metrics
        metrics[f"{agent_id}/kl_divergence"] = kl_divergence
        metrics[f"{agent_id}/ppo_steps_made"] = ppo_step + 1
        metrics[f"{agent_id}/policy_loss"] = policy_loss.mean().cpu().item()
        metrics[f"{agent_id}/value_loss"] = value_loss.mean().cpu().item()
        # metrics[f"{agent_id}/total_loss"] = loss.detach().cpu().item()
        metrics[f"{agent_id}/total_steps"] = len(reward_batch.view(-1))

        # ep_lens = ep_lens if self.config["pad_sequences"] else get_episode_lens(done_batch.cpu())
        # ep_lens = get_episode_lens(done_batch.cpu())

        # Group rewards by episode and sum them up to get full episode returns
        # if self.config["pad_sequences"]:
        #     ep_rewards = reward_batch.sum(0)
        # else:
        # ep_rewards = torch.tensor([torch.sum(rewards) for rewards in torch.split(reward_batch, ep_lens)])

        # # Episode length metrics
        # metrics[f"{agent_id}/episode_len_mean"] = np.mean(ep_lens)
        # metrics[f"{agent_id}/episode_len_median"] = np.median(ep_lens)
        # metrics[f"{agent_id}/episode_len_min"] = np.min(ep_lens)
        # metrics[f"{agent_id}/episode_len_max"] = np.max(ep_lens)
        # metrics[f"{agent_id}/episode_len_std"] = np.std(ep_lens)

        ep_rewards = reward_batch.sum(0)

        # Episode reward metrics
        metrics[f"{agent_id}/episode_reward_mean"] = torch.mean(ep_rewards).item()
        metrics[f"{agent_id}/episode_reward_median"] = torch.median(ep_rewards).item()
        metrics[f"{agent_id}/episode_reward_min"] = torch.min(ep_rewards).item()
        metrics[f"{agent_id}/episode_reward_max"] = torch.max(ep_rewards).item()
        metrics[f"{agent_id}/episode_reward_std"] = torch.std(ep_rewards).item()

        # Other metrics
        metrics[f"{agent_id}/episodes_this_iter"] = done_batch.shape[1]
        metrics[f"{agent_id}/mean_entropy"] = torch.mean(entropy_batch).item()

        metrics[f"{agent_id}/entropy_bonus"] = entropy_coeff

        metrics[f"{agent_id}/time_update"] = timer.checkpoint()

        # Write the metrics to tensorboard
        write_dict(metrics, step, writer)

        return metrics

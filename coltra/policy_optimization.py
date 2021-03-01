from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from typarse import BaseConfig

from coltra.agents import Agent
from coltra.utils import get_optimizer, DataBatch, Timer, AgentDataBatch, \
    write_dict, batch_to_gpu


def discount_td_rewards(data_batch: AgentDataBatch,  # TODO: replace with bGAE
                        gamma: float = 0.99,
                        lam: float = 0.95) -> Tuple[Tensor, Tensor]:
    """An alternative TD-based method of return-to-go and advantage estimation via GAE"""

    rewards_batch = data_batch['rewards']  # (T, E)
    values_batch = data_batch['values']  # (T, E)

    returns_batch = []
    advantages_batch = []
    returns = values_batch[-1]
    advantages = 0

    for i in reversed(range(len(rewards_batch) - 1)):
        rewards = rewards_batch[i]

        returns = rewards + gamma * returns  # v(s) = r + y*v(s+1)
        returns_batch.insert(0, returns)

        value = values_batch[i]
        next_value = values_batch[i + 1]

        # calc. of discounted advantage = A(s,a) + y^1*A(s+1,a+1) + ...
        delta = rewards + gamma * next_value.detach() - value.detach()  # td_err=q(s,a) - v(s)
        advantages = advantages * lam * gamma + delta
        advantages_batch.insert(0, advantages)

    for key in data_batch:
        data_batch[key] = data_batch[key][:-1]

    advantages_batch = torch.stack(advantages_batch)
    returns_batch = torch.stack(returns_batch)

    return advantages_batch, returns_batch


def minibatches(*tensors: Tensor, batch_size: int = 32, shuffle: bool = True):
    full_size = tensors[0].shape[0]
    for tensor in tensors:
        assert tensor.shape[0] == full_size, "One of the tensors has a different batch size"

    if shuffle:
        indices = np.random.permutation(full_size)
    else:
        indices = np.arange(full_size)

    for i in range(0, full_size, batch_size):
        idx = indices[slice(i, i + batch_size)]

        yield [tensor[idx, ...] for tensor in tensors]


class CrowdPPOptimizer:  # TODO: rewrite this with minibatches
    """
    An optimizer for a single homogeneous crowd agent. Estimates the gradient from the whole batch (no SGD).
    """

    def __init__(self, agent: Agent,
                 config: Dict[str, Any]):

        self.agent = agent

        class Config(BaseConfig):
            gamma: float = 0.99
            gae_lambda: float = 0.95
            ppo_steps: int = 5
            eps: float = 0.1
            target_kl: float = 0.01

            # value_steps: int = 10
            value_coeff: float = 1.0

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

        self.policy_optimizer = get_optimizer(self.config.optimizer)(agent.model.parameters(),
                                                                     **self.config.OptimizerKwargs.to_dict())

        self.value_optimizer = get_optimizer(self.config.optimizer)(agent.model.parameters(),
                                                                    **self.config.OptimizerKwargs.to_dict())

        self.gamma: float = self.config.gamma
        self.eps: float = self.config.eps
        self.gae_lambda: float = self.config.gae_lambda

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

        entropy_coeff = max(
            self.config.entropy_coeff * 0.1 ** (step / self.config.entropy_decay_time),
            self.config.min_entropy
        )

        agent_id = "crowd"
        agent = self.agent

        ####################################### Unpack and prepare the data #######################################
        agent_batch = data_batch

        if self.config.use_gpu:
            agent_batch = batch_to_gpu(agent_batch)
            agent.cuda()

        # Unpacking the data for convenience

        # Compute discounted rewards to go
        # add the 'returns' and 'advantages' keys, and removes last position from other fields
        # advantages_batch, returns_batch = discount_td_rewards(agent_batch, gamma=self.gamma, lam=self.gae_lambda)

        obs_batch: Tensor = agent_batch['observations']
        action_batch: Tensor = agent_batch['actions']  # actions taken
        reward_batch: Tensor = agent_batch['rewards']  # rewards obtained
        done_batch: Tensor = agent_batch['dones']  # whether the step is the end of an episode
        # state_batch = agent_batch['states']  # hidden LSTM state
        # discounted_batch: Tensor = agent_batch['returns']
        # advantages_batch: Tensor = agent_batch['advantages']

        # Evaluate actions to have values that require gradients
        logprobs_batch, _, _ = agent.evaluate(obs_batch, action_batch)
        old_logprobs_batch = logprobs_batch.detach()

        # Move data to GPU if applicable
        # if self.config.use_gpu:
        #     returns_batch = returns_batch.cuda()

        # breakpoint()
        # Compute the normalized advantage
        # advantages_batch = (discounted_batch - value_batch).detach()
        # advantages_batch = (advantages_batch - advantages_batch.mean())
        # advantages_batch = advantages_batch / (torch.sqrt(torch.mean(advantages_batch ** 2) + 1e-8))

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
            advantages_batch, returns_batch = discount_td_rewards(agent_batch, gamma=self.gamma, lam=self.gae_lambda)

            for obs_mini, action_mini, old_logprob_mini, advantage_mini, return_mini in minibatches(obs_batch,
                                                                                                    action_batch,
                                                                                                    old_logprobs_batch,
                                                                                                    advantages_batch,
                                                                                                    returns_batch,
                                                                                                    batch_size=32,
                                                                                                    shuffle=True):
                # TODO: complete converting this into minibatches
                # Evaluate again after the PPO step, for new values and gradients
                logprob_batch, value_batch, entropy_batch = agent.evaluate(obs_batch, action_batch)
                # Compute the KL divergence for early stopping
                kl_divergence = torch.mean(old_logprobs_batch - logprob_batch).item()
                if np.isnan(kl_divergence): breakpoint()
                if kl_divergence > self.config.target_kl:
                    break

                ######################################### Compute the loss #############################################
                # Surrogate loss
                prob_ratio = torch.exp(logprob_batch - old_logprobs_batch)
                surr1 = prob_ratio * advantages_batch

                surr2 = torch.where(torch.gt(advantages_batch, 0),
                                    (1 + self.eps) * advantages_batch,
                                    (1 - self.eps) * advantages_batch)

                policy_loss = -torch.min(surr1, surr2)

                value_loss = (value_batch - returns_batch) ** 2

                loss = (
                        policy_loss.mean()
                        + value_loss.mean()
                        - (entropy_coeff * entropy_batch.mean())
                )

                ############################################# Update step ##############################################
                self.policy_optimizer.zero_grad()
                loss.backward()

                self.policy_optimizer.step()

        # for value_step in range(self.config.value_steps):
        #     _, value_batch, _ = agent.evaluate(obs_batch, action_batch)
        #
        #     value_loss = (value_batch - discounted_batch) ** 2
        #
        #     loss = value_loss.mean()
        #
        #     self.value_optimizer.zero_grad()
        #     loss.backward()
        #     self.value_optimizer.step()

        ############################################## Collect metrics #############################################

        # Training-related metrics
        metrics[f"meta/kl_divergence"] = kl_divergence
        metrics[f"meta/ppo_steps_made"] = ppo_step + 1
        metrics[f"meta/policy_loss"] = policy_loss.mean().cpu().item()
        metrics[f"meta/value_loss"] = value_loss.mean().cpu().item()
        # metrics[f"{agent_id}/total_loss"] = loss.detach().cpu().item()
        metrics[f"meta/total_steps"] = len(reward_batch.view(-1))

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
        metrics[f"meta/episodes_this_iter"] = done_batch.shape[1]
        metrics[f"meta/mean_entropy"] = torch.mean(entropy_batch).item()

        metrics[f"meta/entropy_bonus"] = entropy_coeff

        metrics[f"meta/time_update"] = timer.checkpoint()

        # Write the metrics to tensorboard
        write_dict(metrics, step, writer)

        return metrics

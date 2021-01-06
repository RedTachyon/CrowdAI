import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch
import yaml

from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from agents import Agent
from parallel import SubprocVecEnv
from utils import Timer, with_default_config, write_dict, concat_subproc_batch
from collectors import collect_crowd_data
from policy_optimization import CrowdPPOptimizer

class Trainer:
    def __init__(self,
                 agent: Agent,
                 env: SubprocVecEnv,
                 config: Dict[str, Any]):
        pass

    def train(self, num_iterations: int,
              disable_tqdm: bool = False,
              save_path: Optional[str] = None,
              **collect_kwargs):
        raise NotImplementedError


class PPOCrowdTrainer(Trainer):
    """This performs training in a sampling paradigm, where each agent is stored, and during data collection,
    some part of the dataset is collected with randomly sampled old agents"""

    def __init__(self,
                 agent: Agent,
                 env: SubprocVecEnv,
                 config: Dict[str, Any]):
        super().__init__(agent, env, config)

        default_config = {
            "steps": 500,  # number of steps we want in one episode
            "workers": 8,

            # Tensorboard settings
            "tensorboard_name": None,  # str, set explicitly

            "save_freq": 10,

            # PPO
            "ppo_config": {
                # GD settings
                "optimizer": "adam",
                "optimizer_kwargs": {
                    "lr": 1e-4,
                    "betas": (0.9, 0.999),
                    "eps": 1e-7,
                    "weight_decay": 0,
                    "amsgrad": False
                },
                "gamma": 1.,  # Discount factor

                # PPO settings
                "ppo_steps": 25,  # How many max. gradient updates in one iterations
                "eps": 0.1,  # PPO clip parameter
                "target_kl": 0.01,  # KL divergence limit
                "value_loss_coeff": 0.1,
                "entropy_coeff": 0.1,
                "max_grad_norm": 0.5,

                # Backpropagation settings
                "use_gpu": False,
            }
        }

        self.agent = agent
        self.config = config

        self.env = env

        self.config = with_default_config(config["trainer"], default_config)

        self.ppo = CrowdPPOptimizer(self.agent, config=self.config["ppo_config"])

        # Setup tensorboard
        self.writer: SummaryWriter
        if self.config["tensorboard_name"]:
            dt_string = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.path = Path.home() / "tb_logs" / f"{self.config['tensorboard_name']}_{dt_string}"

            self.writer = SummaryWriter(str(self.path))
            os.mkdir(str(self.path / "saved_weights"))

            # Log the configs
            with open(str(self.path / "trainer_config.yaml"), "w") as f:
                yaml.dump(self.config, f)
            with open(str(self.path / f"full_config.yaml"), "w") as f:
                yaml.dump(config, f)

            self.path = str(self.path)
        else:
            self.path = None
            self.writer = None

    def train(self, num_iterations: int,
              save_path: Optional[str] = None,
              disable_tqdm: bool = False,
              **collect_kwargs):

        if save_path is None:
            save_path = self.path  # Can still be None

        print(f"Begin training, logged in {self.path}")
        timer = Timer()
        step_timer = Timer()

        if save_path:
            torch.save(self.agent.model, os.path.join(save_path, "base_agent.pt"))

        for step in trange(1, num_iterations+1, disable=disable_tqdm):
            ########################################### Collect the data ###############################################
            timer.checkpoint()

            full_batch, collector_metrics = collect_crowd_data(agent=self.agent,
                                                               env=self.env,
                                                               num_steps=self.config["steps"])
            # breakpoint()
            full_batch = concat_subproc_batch(full_batch)
            # collector_metrics = concat_metrics(collector_metrics)

            # full_batch, collector_metrics = collect_parallel_unity(num_workers=self.config["workers"],
            #                                                        num_runs=self.config["workers"],
            #                                                        agent=self.agent,
            #                                                        env_path=self.env_path,
            #                                                        num_steps=self.config["steps"],
            #                                                        base_seed=step)

            data_time = timer.checkpoint()

            ############################################## Update policy ##############################################
            # Perform the PPO update
            metrics = self.ppo.train_on_data(full_batch, step, writer=self.writer)

            end_time = step_timer.checkpoint()

            ########################################## Save the updated agent ##########################################

            # Save the agent to disk
            if save_path and (step % self.config["save_freq"] == 0):
                # torch.save(old_returns, os.path.join(save_path, "returns.pt"))
                torch.save(self.agent.model.state_dict(),
                           os.path.join(save_path, "saved_weights", f"weights_{step}"))

            # Write remaining metrics to tensorboard
            extra_metric = {f"crowd/time_data_collection": data_time,
                            f"crowd/total_time": end_time,
                            f"crowd/mean_distance": np.mean(collector_metrics["mean_distance"]),
                            f"crowd/mean_speed": np.mean(collector_metrics["mean_speed"]),
                            f"crowd/mean_speed_100": np.mean(collector_metrics["mean_speed"][:100]),
                            f"crowd/mean_speed_l100": np.mean(collector_metrics["mean_speed"][-100:]),
                            f"crowd/mean_finish": np.mean(collector_metrics["mean_finish"]),
                            f"crowd/mean_finish_l1": np.mean(collector_metrics["mean_finish"][-1]),
                            f"crowd/mean_distance_l100": np.mean(collector_metrics["mean_distance"][-100:]),
                            f"crowd/collisions_per_capita": np.sum(collector_metrics["mean_collision"].mean(1))}

            write_dict(extra_metric, step, self.writer)


if __name__ == '__main__':
    pass
    # from rollout import Collector

    # env_ = foraging_env_creator({})

    # agent_ids = ["Agent0", "Agent1"]
    # agents_: Dict[str, Agent] = {
    #     agent_id: Agent(LSTMModel({}), name=agent_id)
    #     for agent_id in agent_ids
    # }
    #
    # runner = Collector(agents_, env_)
    # data_batch = runner.rollout_steps(num_episodes=10, disable_tqdm=True)
    # obs_batch = data_batch['observations']['Agent0']
    # action_batch = data_batch['actions']['Agent0']
    # reward_batch = data_batch['rewards']['Agent0']
    # done_batch = data_batch['dones']['Agent0']
    #
    # logprob_batch, value_batch, entropy_batch = agents_['Agent0'].evaluate_actions(obs_batch,
    #                                                                                action_batch,
    #                                                                                done_batch)

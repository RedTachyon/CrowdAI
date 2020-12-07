from collections import OrderedDict
from typing import Dict, Callable, List, Tuple, Optional, TypeVar, Any

import gym
import numpy as np
import torch
from torch import Tensor
from tqdm import trange

from agents import BaseAgent, Agent, StillAgent, RandomAgent
from preprocessors import simple_padder
from utils import DataBatch, with_default_config, np_float, transpose_batch, concat_batches, pack, unpack
from environments import MultiAgentEnv, UnityCrowdEnv

T = TypeVar('T')


def append_dict(var: Dict[str, T], data_dict: Dict[str, List[T]]):
    """
    Works like append, but operates on dictionaries of lists and dictionaries of values (as opposed to lists and values)

    Args:
        var: values to be appended
        data_dict: lists to be appended to
    """
    for key, value in var.items():
        data_dict.setdefault(key, []).append(value)  # variant to allow for new agents to be created in real time
        # data_dict[key].append(value)


class Memory:
    """
    Holds the rollout data in a nested dictionary structure as follows:
    {
        "observations":
            {
                "Agent0": [obs1, obs2, ...],
                "Agent1": [obs1, obs2, ...]
            },
        "actions":
            {
                "Agent0": [act1, act2, ...],
                "Agent1": [act1, act2, ...]
            },
        ...,
        "states":
            {
                "Agent0": [(h1, c1), (h2, c2), ...]
                "Agent1": [(h1, c1), (h2, c2), ...]
            }
    }
    """

    def __init__(self, fields: List[str] = None):
        """
        Creates the memory container. The only argument is a list of agent names to set up the dictionaries.

        Args:
            fields: names of fields to store in memory
        """
        if fields is None:
            self.fields = ['observations', 'actions', 'rewards', 'logprobs', 'dones']
        else:
            self.fields = fields

        self.data = {  # TODO: Optimize by allocating some memory beforehand?
            field: {}
            for field in self.fields
        }

    def store(self, *args):
        update = args
        for key, var in zip(self.data, update):
            append_dict(var, self.data[key])
            # append_dict(var, self.data.setdefault(key, {}))

    def reset(self):
        for key in self.data:
            self.data[key] = {}

    def apply_to_dict(self, func: Callable, d: Dict):
        return {
            key: func(key) for key in d
        }

    def get_torch_data(self) -> DataBatch:
        """
        Gather all the recorded data into torch tensors (still keeping the dictionary structure)
        """

        # The first version is old and worked, the second one is shorter and should work too
        torch_data = {
            # field_name: self.apply_to_dict(lambda agent: torch.tensor(self.data[field_name][agent]), field)
            field_name: {agent: torch.tensor(field[agent]) for agent in field}
            for field_name, field in self.data.items()
        }

        return torch_data

    def set_done(self):
        assert 'dones' in self.data, "Must collect dones in the memory"
        for data in self.data['dones'].values():
            data[-1] = True

    def __getitem__(self, item):
        return self.data[item]

    def __str__(self):
        return self.data.__str__()


def collect_crowd_data(agent: Agent,
                       env: MultiAgentEnv,
                       num_steps: Optional[int] = None,
                       deterministic: bool = False,
                       disable_tqdm: bool = True,
                       include_last: bool = False,
                       reset_start: bool = True
                       ) -> Tuple[DataBatch, Dict]:

    """
            Performs a rollout of the agents in the environment, for an indicated number of steps or episodes.

            Args:
                agent: Agent with which to collect the data
                env: Environment in which the agent will act
                num_steps: number of steps to take; either this or num_episodes has to be passed (not both)
                deterministic: whether each agent should use the greedy policy; False by default
                disable_tqdm: whether a live progress bar should be (not) displayed
                include_last: whether to include the last observation in episodic mode - useful for visualizations
                reset_start: whether the environment should be reset at the beginning of collection

            Returns: dictionary with the gathered data in the following format:

            {
                "observations":
                    {
                        "Agent0": tensor([obs1, obs2, ...]),

                        "Agent1": tensor([obs1, obs2, ...])
                    },
                "actions":
                    {
                        "Agent0": tensor([act1, act2, ...]),

                        "Agent1": tensor([act1, act2, ...])
                    },
                ...,

                "states":
                    {
                        "Agent0": (tensor([h1, h2, ...]), tensor([c1, c2, ...])),

                        "Agent1": (tensor([h1, h2, ...]), tensor([c1, c2, ...]))
                    }
            }
            """
    memory = Memory(['observations', 'actions', 'rewards', 'logprobs', 'dones'])

    if reset_start:
        obs_dict = env.reset()
    else:
        obs_dict = env.current_obs

    # start_metrics = self.env.current_info['metrics']

    # state = {
    #     agent_id: self.agents[agent_id].get_initial_state(requires_grad=False) for agent_id in self.agent_ids
    # }

    metrics = {
        "mean_distance": [],  # [start_metrics[0]],
        "mean_speed": [],  # [start_metrics[1]],
        "mean_finish": []  # [start_metrics[2]]
    }

    end_flag = False
    for step in trange(num_steps, disable=disable_tqdm):
        # Compute the action for each agent
        # action_info = {  # action, logprob, entropy, state, sm
        #     agent_id: self.agents[agent_id].compute_single_action(obs[agent_id],
        #                                                           # state[agent_id],
        #                                                           deterministic[agent_id])
        #     for agent_id in obs
        # }

        # TODO: have a separate agent for each behavior in the environment
        # TODO: reintroduce recurrent state management

        obs_array, agent_keys = pack(obs_dict)
        obs_tensor = torch.tensor(obs_array)

        # Centralize the action computation for better parallelization
        actions, logprobs, _ = agent.compute_actions(obs_tensor, (), deterministic)

        action_dict = unpack(actions, agent_keys)
        logprob_dict = unpack(logprobs, agent_keys)

        # Actual step in the environment
        next_obs, reward_dict, done_dict, info_dict = env.step(action_dict)

        # Collect the metrics passed by the environment
        mean_distance, mean_speed, mean_finish = info_dict["metrics"].T
        metrics["mean_distance"].append(mean_distance)
        metrics["mean_speed"].append(mean_speed)
        metrics["mean_finish"].append(mean_finish)

        # Saving to memory
        memory.store(obs_dict, action_dict, reward_dict, logprob_dict, done_dict)

        # Update the current obs and state - either reset, or keep going
        if done_dict["__all__"]:  # episode is over
            if include_last:  # record the last observation along with placeholder action/reward/logprob
                memory.store(next_obs, action_dict, reward_dict, logprob_dict, done_dict)

            # Step mode with episode finish handling
            if end_flag:
                break

            # If we didn't end, create a new environment
            obs_dict = env.reset()

        else:  # keep going
            obs_dict = next_obs
            # obs_dict = {key: obs for key, obs in next_obs.items() if key in env.active_agents}

    memory.set_done()
    metrics = {key: np.array(value) for key, value in metrics.items()}

    # TODO: make sure metrics work with parallel environments
    return memory.get_torch_data(), metrics


class CrowdCollector:
    """DEPRECATED
    Class to perform data collection from two agents.
    """

    def __init__(self, agent: Agent, env: MultiAgentEnv):
        self.agent = agent
        self.env = env
        self.memory = Memory(['observations', 'actions', 'rewards', 'logprobs', 'dones'])

    def collect_data(self,
                     num_steps: Optional[int] = None,
                     deterministic: bool = False,
                     disable_tqdm: bool = True,
                     reset_memory: bool = True,
                     include_last: bool = False,
                     reset_start: bool = True) -> Tuple[DataBatch, Dict]:
        """
        Performs a rollout of the agents in the environment, for an indicated number of steps or episodes.

        Args:
            num_steps: number of steps to take; either this or num_episodes has to be passed (not both)
            deterministic: whether each agent should use the greedy policy; False by default
            disable_tqdm: whether a live progress bar should be (not) displayed
            reset_memory: whether to reset the memory before generating data
            include_last: whether to include the last observation in episodic mode - useful for visualizations
            reset_start: whether the environment should be reset at the beginning of collection

        Returns: dictionary with the gathered data in the following format:

        {
            "observations":
                {
                    "Agent0": tensor([obs1, obs2, ...]),

                    "Agent1": tensor([obs1, obs2, ...])
                },
            "actions":
                {
                    "Agent0": tensor([act1, act2, ...]),

                    "Agent1": tensor([act1, act2, ...])
                },
            ...,

            "states":
                {
                    "Agent0": (tensor([h1, h2, ...]), tensor([c1, c2, ...])),

                    "Agent1": (tensor([h1, h2, ...]), tensor([c1, c2, ...]))
                }
        }
        """

        if reset_memory:  # clears the memory cache
            self.reset()

        if reset_start:
            obs_dict = self.env.reset()
        else:
            obs_dict = self.env.current_obs

        # start_metrics = self.env.current_info['metrics']

        # state = {
        #     agent_id: self.agents[agent_id].get_initial_state(requires_grad=False) for agent_id in self.agent_ids
        # }

        metrics = {
            "mean_distance": [],#[start_metrics[0]],
            "mean_speed": [],#[start_metrics[1]],
            "mean_finish": []#[start_metrics[2]]
        }

        end_flag = False
        for step in trange(num_steps, disable=disable_tqdm):
            # Compute the action for each agent
            # action_info = {  # action, logprob, entropy, state, sm
            #     agent_id: self.agents[agent_id].compute_single_action(obs[agent_id],
            #                                                           # state[agent_id],
            #                                                           deterministic[agent_id])
            #     for agent_id in obs
            # }

            # TODO: have a separate agent for each behavior in the environment
            # TODO: reintroduce recurrent state management

            obs_array, agent_keys = pack(obs_dict)
            obs_tensor = torch.tensor(obs_array)

            # Centralize the action computation for better parallelization
            actions, logprobs, _ = self.agent.compute_actions(obs_tensor, (), deterministic)

            action_dict = unpack(actions, agent_keys)
            logprob_dict = unpack(logprobs, agent_keys)

            # Actual step in the environment
            next_obs, reward_dict, done_dict, info_dict = self.env.step(action_dict)

            # Collect the metrics passed by the environment
            mean_distance, mean_speed, mean_finish = info_dict["metrics"].T
            metrics["mean_distance"].append(mean_distance)
            metrics["mean_speed"].append(mean_speed)
            metrics["mean_finish"].append(mean_finish)

            # Saving to memory
            self.memory.store(obs_dict, action_dict, reward_dict, logprob_dict, done_dict)

            # Update the current obs and state - either reset, or keep going
            if done_dict["__all__"]:  # episode is over
                if include_last:  # record the last observation along with placeholder action/reward/logprob
                    self.memory.store(next_obs, action_dict, reward_dict, logprob_dict, done_dict)

                # Step mode with episode finish handling
                if end_flag:
                    break

                # If we didn't end, create a new environment
                obs_dict = self.env.reset()


            else:  # keep going
                obs_dict = {key: obs for key, obs in next_obs.items() if key in self.env.active_agents}

        self.memory.set_done()
        metrics = {key: np.array(value) for key, value in metrics.items()}

        # TODO: make sure metrics work with parallel environments
        return self.memory.get_torch_data(), metrics

    def reset(self):
        self.memory.reset()

    def change_agent(self, new_agent: Agent):
        """Replace the agents in the collector"""
        self.agent = new_agent

    def update_agent_state_dict(self, state_dict: OrderedDict):
        """Update the state dict of the agents in the collector"""
        self.agent.model.load_state_dict(state_dict)


if __name__ == '__main__':
    pass

    # env = foraging_env_creator({})
    #
    # agent_ids = ["Agent0", "Agent1"]
    #
    # agents: Dict[str, Agent] = {
    #     agent_id: Agent(LSTMModel({}), name=agent_id)
    #     for agent_id in agent_ids
    # }
    #
    # runner = Collector(agents, env, {})
    #
    # data_steps = runner.collect_data(num_steps=1000, disable_tqdm=False)
    # data_episodes = runner.collect_data(num_episodes=2, disable_tqdm=False)
    # print(data_episodes['observations']['Agent0'].shape)
    # generate_video(data_episodes['observations']['Agent0'], 'vids/video.mp4')

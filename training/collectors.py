from dataclasses import dataclass, field
from typing import Dict, Callable, List, Tuple, Optional, TypeVar, Union

import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import trange

from agents import Agent
from parallel import SubprocVecEnv
from utils import DataBatch, concat_batches, \
    concat_crowd_batch, concat_metrics
from envs.unity_envs import MultiAgentEnv, UnitySimpleCrowdEnv

from buffers import MemoryRecord, MemoryBuffer, AgentMemoryBuffer


def collect_crowd_data(agent: Agent,
                       env: Union[MultiAgentEnv, SubprocVecEnv],
                       num_steps: Optional[int] = None,
                       deterministic: bool = False,
                       disable_tqdm: bool = True,
                       reset_start: bool = True) -> Tuple[MemoryRecord, Dict]:
    """
        Performs a rollout of the agents in the environment, for an indicated number of steps or episodes.

        Args:
            agent: Agent with which to collect the data
            env: Environment in which the agent will act
            num_steps: number of steps to take; either this or num_episodes has to be passed (not both)
            deterministic: whether each agent should use the greedy policy; False by default
            disable_tqdm: whether a live progress bar should be (not) displayed
            reset_start: whether the environment should be reset at the beginning of collection

        Returns:
            data: a nested dictionary with the data
            metrics: a dictionary of metrics passed by the environment
    """
    memory = MemoryBuffer()

    if reset_start:
        obs_dict = env.reset()
    else:
        obs_dict = env.current_obs

    # state = {
    #     agent_id: self.agents[agent_id].get_initial_state(requires_grad=False) for agent_id in self.agent_ids
    # }
    metrics = {}

    for step in trange(num_steps, disable=disable_tqdm):
        # Compute the action for each agent
        # action_info = {  # action, logprob, entropy, state, sm
        #     agent_id: self.agents[agent_id].compute_single_action(obs[agent_id],
        #                                                           # state[agent_id],
        #                                                           deterministic[agent_id])
        #     for agent_id in obs
        # }

        # Converts a dict to a compact array which will be fed to the network - needs rethinking
        # TODO: Change this to use a GroupAgent instead?
        obs_array, agent_keys = env.pack(obs_dict)

        # Centralize the action computation for better parallelization
        actions, states, extra = agent.act(obs_array, (), deterministic, get_value=True)
        # actions, logprobs, values, _ = agent.compute_actions(obs_tensor, (), deterministic)

        values = extra["value"]

        action_dict = env.unpack(actions, agent_keys)  # Convert an array to a agent-indexed dictionary
        # logprob_dict = env.unpack(logprobs, agent_keys)
        values_dict = env.unpack(values, agent_keys)

        # Actual step in the environment
        next_obs, reward_dict, done_dict, info_dict = env.step(action_dict)
        # breakpoint()

        # Collect the metrics passed by the environment
        if isinstance(info_dict, tuple):
            # all_metrics = np.concatenate([info["metrics"] for info in info_dict])
            all_metrics = {}
            for key in info_dict[0].keys():
                if key.startswith("m_"):
                    all_metrics[key] = np.concatenate([val[key] for val in info_dict])
        else:
            # all_metrics = info_dict["metrics"]
            all_metrics = {k: v for k, v in info_dict.items() if k.startswith("m_")}

        if not all_metrics:
            breakpoint()

        for key in all_metrics:
            metrics.setdefault(key[2:], []).append(all_metrics[key])

        memory.append(obs_dict, action_dict, reward_dict, values_dict, done_dict)

        obs_dict = next_obs

        # \/ Unused if the episode can't end by itself, interrupts vectorized env collection
        # If I want to reintroduce it, probably move it back one line
        # Update the current obs and state - either reset, or keep going
        # if done_dict["__all__"]:  # episode is over
        #
        #     # Step mode with episode finish handling
        #     if end_flag:
        #         break
        #
        #     # If we didn't end, create a new environment
        #     obs_dict = env.reset()
        #
        # else:  # keep going
        #     obs_dict = next_obs
        #     # obs_dict = {key: obs for key, obs in next_obs.items() if key in env.active_agents}
        #

    metrics = {key: np.array(value) for key, value in metrics.items()}

    data = memory.crowd_tensorify()
    return data, metrics


# def _collection_worker(agent: Agent, i: int, env_path: str, num_steps: int, base_seed: int) -> Tuple[DataBatch, Dict]:
#     # seed = round(time.time() % 100000) + i  # Ensure it's different every time
#     seed = base_seed * 100 + i
#     env = UnitySimpleCrowdEnv(file_name=env_path, no_graphics=True, worker_id=i, timeout_wait=5, seed=seed)
#     env.engine_channel.set_configuration_parameters(time_scale=100)
#     data, metrics = collect_crowd_data(agent, env, num_steps)
#     env.close()
#     return data, metrics


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

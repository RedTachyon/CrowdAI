import numpy as np
import gym
from typing import Dict, Any, Tuple, Callable, List

import torch
from mlagents_envs.base_env import BehaviorSpec, ActionTuple
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

from coltra.envs.side_channels import StatsChannel, parse_side_message
from coltra.buffers import Observation, Action, Reward, Done, TensorArray

StateDict = Dict[str, Observation]
ActionDict = Dict[str, Action]
RewardDict = Dict[str, Reward]
DoneDict = Dict[str, Done]
InfoDict = Dict[str, Any]


class MultiAgentEnv(gym.Env):
    """
    Base class for a gym-like environment for multiple agents. An agent is identified with its id (string),
    and most interactions are communicated through that API (actions, states, etc)
    """

    obs_vector_size: int
    action_vector_size: int

    def __init__(self):
        self.config = {}
        self.active_agents: List = []

    def reset(self, *args, **kwargs) -> StateDict:
        """
        Resets the environment and returns the state.
        Returns:
            A dictionary holding the state visible to each agent.
        """
        raise NotImplementedError

    def step(self, action_dict: ActionDict) -> Tuple[StateDict, RewardDict, DoneDict, InfoDict]:
        """
        Executes the chosen actions for each agent and returns information about the new state.

        Args:
            action_dict: dictionary holding each agent's action

        Returns:
            states: new state for each agent
            rewards: reward obtained by each agent
            dones: whether the environment is done for each agent
            infos: any additional information
        """
        raise NotImplementedError

    def render(self, mode='human'):
        raise NotImplementedError

    @staticmethod
    def pack(dict_: Dict[str, Observation]) -> Tuple[Observation, List[str]]:
        keys = list(dict_.keys())
        values = Observation.stack_tensor([dict_[key] for key in keys])

        return values, keys

    @staticmethod
    def unpack(arrays: Any, keys: List[str]) -> Dict[str, Any]:
        value_dict = {key: arrays[i] for i, key in enumerate(keys)}
        return value_dict


class UnitySimpleCrowdEnv(MultiAgentEnv):

    def __init__(self, *args, **kwargs):

        super().__init__()
        self.engine_channel = EngineConfigurationChannel()
        self.stats_channel = StatsChannel()

        self.active_agents: List[str] = []

        kwargs.setdefault("side_channels", []).append(self.engine_channel)
        kwargs["side_channels"].append(self.stats_channel)

        self.unity = UnityEnvironment(*args, **kwargs)
        self.behaviors = {}
        self.manager = ""

        # semi-hardcoded computation of osb/action spaces, slightly different api than gym
        self.obs_vector_size = next(iter(self.reset().values())).vector.shape[0]
        self.action_vector_size = 2

    def _get_step_info(self, step: bool = False) -> Tuple[StateDict, RewardDict, DoneDict, InfoDict]:
        names = self.behaviors.keys()
        obs_dict: StateDict = {}
        reward_dict: RewardDict = {}
        done_dict: DoneDict = {}
        info_dict: InfoDict = {}

        ter_obs_dict = {}
        ter_reward_dict = {}
        # has_decision = False
        for name in names:
            decisions, terminals = self.unity.get_steps(name)

            dec_obs, dec_ids = decisions.obs, list(decisions.agent_id)
            for idx in dec_ids:
                agent_name = f"{name}&id={idx}"
                if len(dec_obs) == 1:
                    obs = Observation(vector=dec_obs[0][dec_ids.index(idx)])
                elif len(dec_obs) == 2:
                    obs = Observation(vector=dec_obs[1][dec_ids.index(idx)],
                                      buffer=dec_obs[0][dec_ids.index(idx)])
                else:
                    raise ValueError("Too many observations (vector, buffer and what else?")

                obs_dict[agent_name] = obs
                # obs_dict[agent_name] = np.concatenate([o[dec_ids.index(idx)] for o in dec_obs])
                reward_dict[agent_name] = decisions.reward[dec_ids.index(idx)]
                done_dict[agent_name] = False

            ter_obs, ter_ids = terminals.obs, list(terminals.agent_id)

            for idx in terminals.agent_id:
                agent_name = f"{name}&id={idx}"
                if len(dec_obs) == 1:
                    obs = Observation(vector=ter_obs[0][ter_ids.index(idx)])
                elif len(dec_obs) == 2:
                    obs = Observation(vector=ter_obs[1][ter_ids.index(idx)],
                                      buffer=ter_obs[0][ter_ids.index(idx)])
                else:
                    raise ValueError("Too many observations (vector, buffer and what else?")

                ter_obs_dict[agent_name] = obs
                ter_reward_dict[agent_name] = terminals.reward[ter_ids.index(idx)]
                done_dict[agent_name] = True

        done_dict["__all__"] = len(self.active_agents) == 0

        info_dict["final_obs"] = ter_obs_dict
        info_dict["final_rewards"] = ter_reward_dict

        stats = self.stats_channel.parse_info(clear=step)
        # stats = parse_side_message(self.stats_channel.last_msg)
        for key in stats:
            info_dict["m_" + key] = stats[key]

        return obs_dict, reward_dict, done_dict, info_dict

    def step(self, action: ActionDict) -> Tuple[StateDict, RewardDict, DoneDict, InfoDict]:

        for name in self.behaviors.keys():
            decisions, terminals = self.unity.get_steps(name)
            action_shape = self.behaviors[name].action_spec.continuous_size
            dec_ids = list(decisions.agent_id)

            all_actions = []
            for id_ in dec_ids:
                single_action = action.get(f"{name}&id={id_}",  # Get the appropriate action
                                           Action(continuous=np.zeros(action_shape))  # Default value
                                           )

                cont_action = single_action.apply(np.asarray).continuous.ravel()

                all_actions.append(cont_action)

            # all_actions = np.array([action.get(f"{name}&id={id_}", np.zeros(action_shape)).ravel()
            #                         for id_ in dec_ids])
            #
            if len(all_actions) == 0:
                all_actions = np.zeros((0, action_shape))
            else:
                all_actions = np.array(all_actions)

            self.unity.set_actions(name, ActionTuple(continuous=all_actions))

        # The terminal step handling has been removed as episodes are only reset from here

        self.unity.step()
        obs_dict, reward_dict, done_dict, info_dict = self._get_step_info(step=True)

        return obs_dict, reward_dict, done_dict, info_dict

    def reset(self) -> StateDict:
        self.unity.reset()

        # All behavior names, except for Manager agents which do not take actions but manage the environment
        behaviors = dict(self.unity.behavior_specs)
        self.behaviors = {key: value for key, value in behaviors.items() if not key.startswith("Manager")}

        # ...but manager is used to collect stats
        self.manager = [key for key in behaviors if key.startswith("Manager")][0]

        obs_dict, _, _, _ = self._get_step_info(step=True)

        self.active_agents = list(obs_dict.keys())

        return obs_dict

    @property
    def current_obs(self) -> StateDict:
        obs_dict, _, _, info_dict = self._get_step_info()
        return obs_dict

    @property
    def current_info(self) -> InfoDict:
        _, _, _, info_dict = self._get_step_info()
        return info_dict

    def close(self):
        self.unity.close()

    def render(self, mode='human'):
        raise NotImplementedError

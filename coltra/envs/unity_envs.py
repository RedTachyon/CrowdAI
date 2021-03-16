import numpy as np
from typing import Tuple, List
from enum import Enum

from mlagents_envs.base_env import ActionTuple
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel

from .side_channels import StatsChannel
from coltra.buffers import Observation, Action
from .subproc_vec_env import SubprocVecEnv
from .base_env import MultiAgentEnv, ObsDict, ActionDict, RewardDict, DoneDict, InfoDict


class Mode(Enum):
    Random = 0.0
    Circle = 1.0
    Hallway = 2.0

    @staticmethod
    def from_string(name: str):
        if name.lower() == "random":
            return Mode.Random
        elif name.lower() == "circle":
            return Mode.Circle
        elif name.lower() == "hallway":
            return Mode.Hallway
        else:
            raise ValueError(f"{name} is not a valid mode identifier")


class UnitySimpleCrowdEnv(MultiAgentEnv):

    def __init__(self, file_name: str = None, **kwargs):

        super().__init__()
        self.engine_channel = EngineConfigurationChannel()
        self.stats_channel = StatsChannel()
        self.param_channel = EnvironmentParametersChannel()

        self.active_agents: List[str] = []

        kwargs.setdefault("side_channels", []).append(self.engine_channel)
        kwargs["side_channels"].append(self.stats_channel)
        kwargs["side_channels"].append(self.param_channel)

        self.unity = UnityEnvironment(file_name=file_name, **kwargs)
        self.behaviors = {}
        self.manager = ""

        # semi-hardcoded computation of obs/action spaces, slightly different api than gym
        self.obs_vector_size = next(iter(self.reset().values())).vector.shape[0]
        self.action_vector_size = 2

    def _get_step_info(self, step: bool = False) -> Tuple[ObsDict, RewardDict, DoneDict, InfoDict]:
        names = self.behaviors.keys()
        obs_dict: ObsDict = {}
        reward_dict: RewardDict = {}
        done_dict: DoneDict = {}
        info_dict: InfoDict = {}

        ter_obs_dict = {}
        ter_reward_dict = {}
        # has_decision = False
        for name in names:
            decisions, terminals = self.unity.get_steps(name)

            dec_obs, dec_ids = decisions.obs, list(decisions.agent_id)
            # TODO: Fix this to avoid repetition
            for idx in dec_ids:
                agent_name = f"{name}&id={idx}"
                if len(dec_obs) == 1:
                    obs = Observation(vector=dec_obs[0][dec_ids.index(idx)])
                elif len(dec_obs) == 2:
                    obs = Observation(vector=dec_obs[1][dec_ids.index(idx)],
                                      buffer=dec_obs[0][dec_ids.index(idx)])
                elif len(dec_obs) == 3:
                    obs = Observation(vector=dec_obs[2][dec_ids.index(idx)],
                                      rays=dec_obs[1][dec_ids.index(idx)],
                                      buffer=dec_obs[0][dec_ids.index(idx)])
                else:
                    raise ValueError(f"Too many observations, got {len(dec_obs)}")

                obs_dict[agent_name] = obs
                # obs_dict[agent_name] = np.concatenate([o[dec_ids.index(idx)] for o in dec_obs])
                reward_dict[agent_name] = decisions.reward[dec_ids.index(idx)]
                done_dict[agent_name] = False

            ter_obs, ter_ids = terminals.obs, list(terminals.agent_id)

            for idx in terminals.agent_id:
                agent_name = f"{name}&id={idx}"
                if len(ter_obs) == 1:
                    obs = Observation(vector=ter_obs[0][ter_ids.index(idx)])
                elif len(ter_obs) == 2:
                    obs = Observation(vector=ter_obs[1][ter_ids.index(idx)],
                                      buffer=ter_obs[0][ter_ids.index(idx)])
                elif len(ter_obs) == 3:
                    obs = Observation(vector=ter_obs[2][ter_ids.index(idx)],
                                      rays=ter_obs[1][ter_ids.index(idx)],
                                      buffer=ter_obs[0][ter_ids.index(idx)])
                else:
                    raise ValueError(f"Too many observations, got {len(ter_obs)}")

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

    def step(self, action: ActionDict) -> Tuple[ObsDict, RewardDict, DoneDict, InfoDict]:

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

    def reset(self, mode: Mode = None, num_agents: int = None, **kwargs) -> ObsDict:
        if mode:
            self.param_channel.set_float_parameter("mode", mode.value)
        if num_agents:
            self.param_channel.set_float_parameter("agents", num_agents)

        for (name, value) in kwargs.items():
            self.param_channel.set_float_parameter(name, value)

        self.unity.reset()

        # All behavior names, except for Manager agents which do not take actions but manage the environment
        behaviors = dict(self.unity.behavior_specs)
        self.behaviors = {key: value for key, value in behaviors.items() if not key.startswith("Manager")}

        # ...but manager is used to collect stats
        self.manager = [key for key in behaviors if key.startswith("Manager")][0]

        obs_dict, _, _, _ = self._get_step_info(step=True)
        if len(obs_dict) == 0:
            # Dealing with some terminal steps due to reducing the number of agents
            self.unity.step()
            obs_dict, _, _, _ = self._get_step_info(step=True)

        self.active_agents = list(obs_dict.keys())
        # if len(self.active_agents) == 0:
        #     self.reset(mode, num_agents)

        return obs_dict

    @property
    def current_obs(self) -> ObsDict:
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

    @classmethod
    def get_env_creator(cls, *args, **kwargs):
        def _inner():
            env = cls(*args, **kwargs)
            env.engine_channel.set_configuration_parameters(time_scale=100)
            return env

        return _inner

    @classmethod
    def get_venv(cls, workers: int = 8, file_name: str = None, *args, **kwargs) -> SubprocVecEnv:
        venv = SubprocVecEnv([
            cls.get_env_creator(file_name=file_name, no_graphics=False, worker_id=i, seed=i, *args, **kwargs)
            for i in range(workers)
        ])
        return venv
from typing import Optional

import gym
import torch
import yaml
from typarse import BaseParser

from coltra.agents import CAgent, DAgent
from coltra.envs.unity_envs import UnitySimpleCrowdEnv
from coltra.envs.probe_envs import ConstRewardEnv
from coltra.models.mlp_models import FancyMLPModel
from coltra.models.relational_models import RelationModel
from coltra.trainers import PPOCrowdTrainer
from coltra.models.raycast_models import LeeModel
from coltra.envs import MultiGymEnv


class Parser(BaseParser):
    config: str = "configs/base_config.yaml"
    iters: int = 500
    env_name: str
    name: str
    workers: int = 8
    start_dir: Optional[str]
    start_idx: Optional[int] = -1

    _help = {
        "config": "Config file for the coltra",
        "iters": "Number of coltra iterations",
        "env_name": "Environment gym name",
        "name": "Name of the tb directory to store the logs",
        "workers": "Number of parallel collection envs to use",
        "start_dir": "Name of the tb directory containing the run from which we want to (re)start the coltra",
        "start_idx": "From which iteration we should start (only if start_dir is set)",
    }

    _abbrev = {
        "config": "c",
        "iters": "i",
        "env_name": "e",
        "name": "n",
        "workers": "w",
        "start_dir": "sd",
        "start_idx": "si",
    }


if __name__ == '__main__':
    CUDA = torch.cuda.is_available()

    args = Parser()

    with open(args.config, "r") as f:
        config = yaml.load(f.read(), yaml.Loader)

    trainer_config = config["trainer"]
    model_config = config["model"]

    trainer_config["tensorboard_name"] = args.name
    trainer_config["ppo_config"]["use_gpu"] = CUDA

    workers = trainer_config.get("workers") or 8  # default value

    # Initialize the environment
    env = MultiGymEnv.get_venv(workers=workers, env_name=args.env_name)
    obs_space = env.observation_space

    # Initialize the agent
    sample_obs = next(iter(env.reset().values()))
    obs_size = sample_obs.vector.shape[0]
    ray_size = sample_obs.rays.shape[0] if sample_obs.rays is not None else None

    model_config["input_size"] = obs_size
    model_config["rays_input_size"] = ray_size

    model_cls = FancyMLPModel
    agent_cls = CAgent if isinstance(obs_space, gym.spaces.Box) else DAgent

    if args.start_dir:
        agent = CAgent.load_agent(args.start_dir, weight_idx=args.start_idx)
    else:
        model = model_cls(model_config)
        agent = CAgent(model)

    if CUDA:
        agent.cuda()

    trainer = PPOCrowdTrainer(agent, env, config)
    trainer.train(args.iters, disable_tqdm=False, save_path=trainer.path)

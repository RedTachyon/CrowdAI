from typing import Optional

import torch
import yaml
from typarse import BaseParser

from coltra.agents import CAgent
from coltra.envs.unity_envs import UnitySimpleCrowdEnv
from coltra.envs.probe_envs import ConstRewardEnv
from coltra.models.mlp_models import FancyMLPModel
from coltra.trainers import PPOCrowdTrainer
from coltra.models.raycast_models import LeeModel


class Parser(BaseParser):
    config: str = "configs/base_config.yaml"
    iters: int = 1000
    env: str
    name: str
    ray_model: bool
    start_dir: Optional[str]
    start_idx: Optional[int] = -1

    _help = {
        "config": "Config file for the coltra",
        "iters": "Number of coltra iterations",
        "env": "Path to the Unity environment binary",
        "name": "Name of the tb directory to store the logs",
        "ray_model": "Whether we should train a model that uses ray perception",
        "start_dir": "Name of the tb directory containing the run from which we want to (re)start the coltra",
        "start_idx": "From which iteration we should start (only if start_dir is set)",
    }

    _abbrev = {
        "config": "c",
        "iters": "i",
        "env": "e",
        "name": "n",
        "ray_model": "r",
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
    if args.env == "probe":
        env = ConstRewardEnv(20)
    else:
        env = UnitySimpleCrowdEnv(args.env)
        env.engine_channel.set_configuration_parameters(time_scale=100)

    # Initialize the agent
    sample_obs = next(iter(env.reset().values()))
    obs_size = sample_obs.vector.shape[0]
    ray_size = sample_obs.rays.shape[0] if sample_obs.rays is not None else None

    model_config["input_size"] = obs_size
    model_config["rays_input_size"] = ray_size

    if args.ray_model:
        model_cls = LeeModel
    else:
        model_cls = FancyMLPModel

    if args.start_dir:
        agent = CAgent.load_agent(args.start_dir, weight_idx=args.start_idx)
    else:
        model = model_cls(model_config)
        agent = CAgent(model)

    if CUDA:
        agent.cuda()

    # env = SubprocVecEnv([
    #     get_env_creator(file_name=args.env, no_graphics=True, worker_id=i, seed=i)
    #     for i in range(workers)
    # ])

    trainer = PPOCrowdTrainer(agent, env, config)
    trainer.train(args.iters, disable_tqdm=False, save_path=trainer.path)

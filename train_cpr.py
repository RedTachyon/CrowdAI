from typing import Optional

import torch
import yaml
from typarse import BaseParser

from coltra.agents import CAgent, DAgent
from coltra.envs.unity_envs import UnitySimpleCrowdEnv
from coltra.envs.harvest_env import HarvestEnv
from coltra.envs.probe_envs import ConstRewardEnv
from coltra.models.mlp_models import FancyMLPModel
from coltra.models.relational_models import RelationModel
from coltra.trainers import PPOCrowdTrainer
from coltra.models.raycast_models import LeeModel


class Parser(BaseParser):
    config: str = "configs/cpr_config.yaml"
    iters: int = 500
    name: str
    workers: int = 8
    num_agents: Optional[int]
    start_dir: Optional[str]
    start_idx: Optional[int] = -1

    _help = {
        "config": "Config file for the coltra",
        "iters": "Number of coltra iterations",
        "name": "Name of the tb directory to store the logs",
        "workers": "Number of parallel collection envs to use",
        "start_dir": "Name of the tb directory containing the run from which we want to (re)start the coltra",
        "start_idx": "From which iteration we should start (only if start_dir is set)",
    }

    _abbrev = {
        "config": "c",
        "iters": "i",
        "name": "n",
        "workers": "w",
        "num_agents": "na",
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

    if args.num_agents:
        trainer_config["num_agents"] = args.num_agents

    workers = trainer_config.get("workers") or 8  # default value

    # Initialize the environment
    env = HarvestEnv.get_venv(workers=args.workers, config={}, num_agents=9, size=(20, 40), num_crosses=30)


    # Initialize the agent
    sample_obs = next(iter(env.reset().values()))
    obs_size = sample_obs.vector.shape[0]

    model_config["input_size"] = obs_size

    model_cls = FancyMLPModel

    if args.start_dir:
        agent = CAgent.load_agent(args.start_dir, weight_idx=args.start_idx)
    else:
        model = model_cls(model_config)
        agent = DAgent(model)

    if CUDA:
        agent.cuda()

    # env = SubprocVecEnv([
    #     get_env_creator(file_name=args.env, no_graphics=True, worker_id=i, seed=i)
    #     for i in range(workers)
    # ])

    trainer = PPOCrowdTrainer(agent, env, config)
    trainer.train(args.iters, disable_tqdm=False, save_path=trainer.path)

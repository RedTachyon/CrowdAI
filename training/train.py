from typing import Dict
import argparse

import torch

from agents import Agent
from models import BaseModel, MLPModel, FancyMLPModel
from trainers import PPOCrowdTrainer


if __name__ == '__main__':
    CUDA = torch.cuda.is_available()

    parser = argparse.ArgumentParser(description='Read training arguments')
    parser.add_argument("--iters", "-i", action="store", type=int, default=1000,
                        help="Number of training iterations")
    parser.add_argument("--steps", "-s", action="store", type=int, default=500,
                        help="Number of steps agent should collect in one episode")
    parser.add_argument("--workers", "-w", action="store", type=int, default=8,
                        help="Number of parallel workers to use")
    parser.add_argument("--env", "-e", action="store", type=str, default="Test.app",
                        help="Path to the Unity environment binary")
    parser.add_argument("--name", "-n", action="store", type=str, default=None,
                        help="Name of the tb directory to store the logs")
    parser.add_argument("--start_dir", "-sd", action="store", type=str, default=None,
                        help="Name of the tb directory containing the run from which we want to (re)start the training")
    parser.add_argument("--start_idx", "-si", action="store", type=int, default=-1,
                        help="From which iteration we should start (only if start_dir is set)")
    parser.add_argument("--learning_rate", "-lr", action="store", type=float, default=1e-4,
                        help="Initial Adam learning rate")
    parser.add_argument("--entropy_bonus", "-eb", action="store", type=float, default=1e-2,
                        help="Entropy bonus coefficient")
    args = parser.parse_args()

    print(args)

    trainer_config = {
        "steps": args.steps,  # number of steps we want in one episode
        "workers": args.workers,

        # Tensorboard settings
        "tensorboard_name": args.name,  # str, set explicitly

        "save_freq": 10,

        # PPO
        "ppo_config": {
            # GD settings
            "optimizer": "adam",
            "optimizer_kwargs": {
                "lr": args.learning_rate,
                "betas": (0.9, 0.999),
                "eps": 1e-7,
                "weight_decay": 0,
                "amsgrad": False
            },
            "gamma": 0.95,  # Discount factor

            # PPO settings
            "ppo_steps": 25,  # How many max. gradient updates in one iterations
            "eps": 0.1,  # PPO clip parameter
            "target_kl": 0.01,  # KL divergence limit
            "value_loss_coeff": 0.1,
            "entropy_coeff": 0.1,
            "max_grad_norm": 0.5,
            "ep_len": args.steps,

            # Backpropagation settings
            "use_gpu": CUDA,
        }
    }

    # action_range = (
    #     torch.tensor([-.3, -1.]),
    #     torch.tensor([1., 1.])
    # )

    action_range = None

    if args.start_dir:
        agent = Agent.load_agent(args.start_dir, action_range=action_range, weight_idx=args.start_idx)
    else:
        model_config = {
            "input_size": 8,
            "num_actions": 2,
            "activation": "tanh",

            "hidden_sizes": (64, 64),
            "separate_value": True,

            "sigma0": 0.3,

            "initializer": "kaiming_uniform",
        }

        model = MLPModel(model_config)
        agent = Agent(model, action_range=action_range)

    agent.model.share_memory()


    trainer = PPOCrowdTrainer(agent, args.env, trainer_config)
    trainer.train(args.iters, disable_tqdm=False, save_path=trainer.path)

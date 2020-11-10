from typing import Dict
import argparse

import torch

from agents import Agent
from models import BaseModel, RelationModel, MLPModel
from trainers import PPOSamplingTrainer

CUDA = torch.cuda.is_available()

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("--iters", "-i", action="store", type=int, default=500,
                    help="Number of training iterations")
parser.add_argument("--subgoals", "-s", action="store", type=int, default=2,
                    help="Number of subgoals")
parser.add_argument("--mode", "-m", action="store", type=str, default="simple",
                    help="Mode of the agent: simple, gt or sm")
parser.add_argument("--name", "-n", action="store", type=str, default=None,
                    help="Name of the tb directory")
parser.add_argument("--half_steps", "-hs", action="store", type=int, default=2500,
                    help="Half of the number of steps agent should collect with itself")
parser.add_argument("--old_samples", "-osamp", action="store", type=int, default=50,
                    help="How many old agents should be samples")
parser.add_argument("--old_steps", "-ostep", action="store", type=int, default=100,
                    help="How many steps should be collected with each of the sampled old agents")
parser.add_argument("--learning_rate", "-lr", action="store", type=float, default=1e-3,
                    help="Initial Adam learning rate")
parser.add_argument("--entropy_bonus", "-eb", action="store", type=float, default=1e-2,
                    help="Entropy bonus coefficient")

ENV_NAME = "action"
GOALS = 1

if __name__ == '__main__':
    args = parser.parse_args()

    print(args)

    env_config = {
        "rows": 7,
        "cols": 7,
        "subgoals": args.subgoals,
        "goals": GOALS,
        "random_positions": True,
        "max_steps": 100,
        # "seed": 8
    }
    env = foraging_env_creator(env_config, env_name=ENV_NAME)
    env.render()

    agent_config = {
            "num_actions": 5,
            "activation": "leaky_relu",
            "initializer": "xavier_uniform",

            # Relation layer
            "emb_size": 4,
            "rel_hiddens": (64, 64, ),
            "mlp_hiddens": (64, 32),
            "goals": GOALS,

            # SM LSTM
            "sm_lstm_nodes": 32,
            "sm_post_lstm_sizes": (32, 32, 1),
            "break_grad": False,

            # ACT LSTM
            "act_lstm_nodes": 32,
            "act_post_lstm_sizes": (32,),

            "mode": args.mode
        }

    model = UnifiedLSTMModel

    agent_ids = ["Agent0", "Agent1"]
    agents: Dict[str, Agent] = {
        agent_id: Agent(model(agent_config))
        for agent_id in agent_ids
    }

    trainer_config = {
        "agents_to_optimize": ["Agent0"],  # ids of agents that should be optimized
        "half_steps_self": args.half_steps,
        "other_samples": args.old_samples,
        "other_steps": args.old_steps,
        "use_tom_iter": True,
        "binary": False,

        # Tensorboard settings
        "tensorboard_name": args.name,  # str, set explicitly

        # Collector
        "collector_config": {
            "finish_episode": True,
        },

        # PPO
        "ppo_config": {
            # GD settings
            "optimizer": "adam",
            "optimizer_kwargs": {
                "lr": args.learning_rate,
                "betas": (0.9, 0.999),
                "eps": 1e-7,
                "weight_decay": 0,
                "amsgrad": False,
            },
            "separate_optimizers": False,
            "gamma": 1.,  # Discount factor

            # PPO settings
            "ppo_steps": 25,
            "eps": 0.1,  # PPO clip parameter
            "target_kl": 0.01,  # KL divergence limit
            "value_loss_coeff": 0.1,

            "entropy_coeff": 0.001,
            "entropy_decay_time": 200,  # How many steps to decrease entropy to 0.1 of the original value
            "min_entropy": 0.001,  # Minimum value of the entropy bonus - use this to disable decay

            "max_grad_norm": 0.5,

            # Backpropagation settings
            "pad_sequences": True if agents['Agent0'].stateful else False,  # BPTT toggle

            # GPU
            "use_gpu": CUDA,
            "use_sm": args.mode == "sm",
            "sm_coeff": 1.,
        }
    }

    trainer = PPOSamplingTrainer(agents, env, config=trainer_config)
    trainer.train(args.iters, disable_tqdm=False, save_path=trainer.path)

import time
import argparse

from coltra.agents import Agent
from coltra.collectors import collect_crowd_data
from coltra.envs.unity_envs import UnitySimpleCrowdEnv

from typarse import BaseParser

class Parser(BaseParser):
    steps: int = 500
    env: str
    start_dir: str
    start_idx: int = -1
    wait: int = 1
    seed: int = 0

    _help = {
        "steps": "Number of steps agent should collect in one episode",
        "env": "Path to the Unity environment binary",
        "start_dir": "Name of the tb directory containing the run from which we want to (re)start the coltra",
        "start_idx": "From which iteration we should start (only if start_dir is set)",
        "wait": "How many seconds to sleep before running",
        "seed": "Random seed for Unity"
    }

    _abbrev = {
        "steps": "s",
        "env": "e",
        "start_dir": "sd",
        "start_idx": "si",
        "wait": "w",
        "seed": "seed"
    }

if __name__ == '__main__':

    # parser = argparse.ArgumentParser(description='Read coltra arguments')
    # parser.add_argument("--steps", "-s", action="store", type=int, default=500,
    #                     help="Number of steps agent should collect in one episode")
    # parser.add_argument("--env", "-e", action="store", type=str, default=None,
    #                     help="Path to the Unity environment binary")
    # parser.add_argument("--start_dir", "-sd", action="store", type=str, default=None,
    #                     help="Name of the tb directory containing the run from which we want to (re)start the coltra")
    # parser.add_argument("--start_idx", "-si", action="store", type=int, default=-1,
    #                     help="From which iteration we should start (only if start_dir is set)")
    # parser.add_argument("--wait", "-w", action="store", type=int, default=1,
    #                     help="How many seconds to sleep before running")
    # parser.add_argument("--seed", "-seed", action="store", type=int, default=0,
    #                     help="Random seed for Unity")
    # args = parser.parse_args()

    args = Parser()

    # action_range = (
    #     torch.tensor([-.3, -1.]),
    #     torch.tensor([1., 1.])
    # )

    action_range = None

    agent = Agent.load_agent(args.start_dir, action_range=action_range, weight_idx=args.start_idx)

    time.sleep(args.wait)

    env = UnitySimpleCrowdEnv(args.env, seed=args.seed)
    env.engine_channel.set_configuration_parameters(width=1000, height=1000, time_scale=1)

    for _ in range(5):
        env.reset()

        data, metrics = collect_crowd_data(agent, env, args.steps)

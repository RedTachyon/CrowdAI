import os
import time

from coltra.agents import Agent, CAgent, ConstantAgent
from coltra.collectors import collect_crowd_data
from coltra.envs.unity_envs import UnitySimpleCrowdEnv, Mode

from typarse import BaseParser

from coltra.utils import Timer


class Parser(BaseParser):
    steps: int = 500
    workers: int = 1
    env: str
    mode: str = "random"
    num_agents: int = 20

    _help = {
        "steps": "Number of steps agent should collect in one episode",
        "workers": "How many subproc parallel envs to use",
        "env": "Path to the Unity environment binary",
        "mode": "Which board mode should be used",
        "num_agents": "How many agents should be in the environment"
    }

    _abbrev = {
        "steps": "s",
        "workers": "w",
        "env": "e",
        "mode": "m",
        "num_agents": "na",
    }


if __name__ == '__main__':

    args = Parser()

    agent = ConstantAgent([1., 1.])

    if args.workers == 1:
        env = UnitySimpleCrowdEnv(file_name=args.env, no_graphics=True)
        env.engine_channel.set_configuration_parameters(time_scale=100)
    else:
        env = UnitySimpleCrowdEnv.get_venv(workers=args.workers, file_name=args.env)

    timer = Timer()
    data, metrics = collect_crowd_data(agent, env, args.steps,
                                       mode=Mode.from_string(args.mode),
                                       num_agents=args.num_agents)
    time = timer.checkpoint()
    print("---------------------------------------------------------------------------------------------------------")
    print("---------------------------------------------------------------------------------------------------------")
    print(f"Time for {args.steps} steps in {os.path.basename(args.env)} with {args.workers} workers: {time:.1f} seconds")
    print("---------------------------------------------------------------------------------------------------------")
    print("---------------------------------------------------------------------------------------------------------")

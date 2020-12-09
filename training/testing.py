import time
from typing import Dict

import torch
import torch.multiprocessing as mp

from agents import Agent
from collectors import collect_crowd_data, collect_parallel_unity
from environments import UnitySimpleCrowdEnv
from models import MLPModel

from utils import concat_crowd_batch, concat_batches, DataBatch, concat_metrics


# TODO: Observation - value loss seems to overtake policy loss later in the training; need to make it so that they don't
#  bother one another, probably by using separate policy/value networks. Maybe just decrease value coeff?

# def collect_parallel(agent: Agent, i: int) -> [DataBatch, Dict]:
#     env = UnitySimpleCrowdEnv(file_name="Test.app", no_graphics=True, worker_id=i, timeout_wait=1)
#     env.engine_channel.set_configuration_parameters(time_scale=100)
#
#     # print(f"Env {i} created")
#     data = collect_crowd_data(agent, env, 500)
#     # print(f"Data {i} collected")
#     # e = time.time()
#     # print(f"Process {i} duration: {e-s_time}")
#     env.close()
#     # print(f"Env {i} closed")
#     return data

if __name__ == '__main__':
    env = UnitySimpleCrowdEnv(file_name=None, no_graphics=True, worker_id=0)
    env.engine_channel.set_configuration_parameters(time_scale=100)
    #
    # env2 = UnitySimpleCrowdEnv(file_name="Test.app", no_graphics=True, worker_id=1)
    # env2.engine_channel.set_configuration_parameters(time_scale=100)

    action_range = (
        torch.tensor([-.3, -1.]),
        torch.tensor([ 1.,  1.])
    )

    model = MLPModel({
        "input_size": 94,
    })

    model.share_memory()

    agent = Agent(model, action_range=action_range)
    s = time.time()

    # data, metrics = collect_parallel_unity(8, 8, agent, "builds/9x9-90deg-1-mac.app", 500)

    data, metrics = collect_crowd_data(agent, env, 500)


    # for p in processes:
    #     p.terminate()
    e = time.time()

    print(f"Total time: {e-s}")

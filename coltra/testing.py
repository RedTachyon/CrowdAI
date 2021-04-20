from torch import nn, tensor, Tensor
from typarse import BaseParser

from coltra.agents import CAgent, ConstantAgent
from coltra.collectors import collect_crowd_data
from coltra.envs.unity_envs import UnitySimpleCrowdEnv

from coltra.envs import SubprocVecEnv


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(tensor([0.]))

    def forward(self, x: Tensor):
        return self.w + x


def _test_worker(model: Model):
    model.w.data += 1.

def get_env_creator(*args, **kwargs):
    def _inner():
        env = UnitySimpleCrowdEnv(*args, **kwargs)
        env.engine_channel.set_configuration_parameters(time_scale=100)
        return env
    return _inner

class Parser(BaseParser):
    env: str

    _help = {
        "env": "Path to the environment executable"
    }

    _abbrev = {
        "env": "e"
    }

if __name__ == '__main__':

    file_name = "builds/1-random-20-mac.app"
    # file_name = None
    venv = SubprocVecEnv(
        [get_env_creator(file_name=file_name, no_graphics=False, worker_id=i, seed=i)
         for i in range(8)]
    )

    agent = ConstantAgent([1, 1])

    data, metrics = collect_crowd_data(agent, venv, 500, disable_tqdm=False)
    breakpoint()

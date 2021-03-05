from torch import Tensor

from coltra.agents import CAgent
from coltra.buffers import Observation
from coltra.models import FCNetwork
import torch

from coltra.models.raycast_models import LeeNetwork, LeeModel


def test_fc():
    torch.manual_seed(0)

    network = FCNetwork(input_size=10,
                        output_sizes=[2, 2],
                        hidden_sizes=[64, 64],
                        activation='tanh',
                        initializer='kaiming_uniform',
                        is_policy=True)

    inp = torch.zeros(5, 10)
    [out1, out2] = network(inp)

    assert isinstance(out1, Tensor)
    assert isinstance(out2, Tensor)
    assert torch.allclose(out1, out2)
    assert torch.allclose(out1, torch.zeros((5, 2)))

    inp = torch.randn(5, 10)
    [out1, out2] = network(inp)

    assert isinstance(out1, Tensor)
    assert isinstance(out2, Tensor)
    assert not torch.allclose(out1, out2)
    assert not torch.allclose(out1, torch.zeros((5, 2)))


def test_empty_fc():
    network = FCNetwork(input_size=10,
                        output_sizes=[32],
                        hidden_sizes=[],
                        activation='elu',
                        initializer='kaiming_uniform',
                        is_policy=False)

    inp = torch.randn(5, 10)
    [out] = network(inp)

    assert isinstance(out, Tensor)
    assert not torch.allclose(out, torch.zeros_like(out))


def test_lee():
    network = LeeNetwork(input_size=4,
                         output_sizes=[2, 4],
                         rays_input_size=126,
                         conv_filters=2)

    obs = Observation(vector=torch.randn(10, 4), rays=torch.randn(10, 126))

    [out1, out2] = network(obs)

    assert out1.shape == (10, 2)
    assert out2.shape == (10, 4)

    model = LeeModel({})
    agent = CAgent(model)

    action, state, extra = agent.act(obs, get_value=True)

    assert action.continuous.shape == (10, 2)
    assert state == ()
    assert extra["value"].shape == (10,)


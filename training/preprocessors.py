from typing import Tuple

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from utils import get_episode_lens, AgentDataBatch


def simple_padder(data_batch: AgentDataBatch) -> Tuple[AgentDataBatch, Tensor]:
    """
    Based on the done values, extracts episode lengths and pads them to a constant length equal to the maximum length
    occurring in the dataset.

    Args:
        data_batch: dictionary of observations, actions etc. for a given agent

    Returns:
        batch of padded data  [T, B, *]
        mask indicating which entries are real and which are padded [T, B]
    """
    ep_lens = get_episode_lens(data_batch['dones'])

    new_batch = {}

    for key, tensor in data_batch.items():
        if key == 'states':  # State is a tuple, so requires different handling
            padded_states = tuple(pad_sequence(torch.split(state, ep_lens)) for state in data_batch[key])
            new_batch[key] = padded_states
        else:
            # Split each tensor, pad each part.
            split = torch.split(data_batch[key], ep_lens)
            new_batch[key] = pad_sequence(split, padding_value=0)

    # Get a mask of 1's where there is actual data, and 0's where it's just the padding
    padded_mask = pad_sequence(torch.split(torch.ones_like(data_batch['actions']), ep_lens))

    return new_batch, padded_mask


def chopping_padder(data_batch: AgentDataBatch) -> Tuple[AgentDataBatch, Tensor]:
    pass

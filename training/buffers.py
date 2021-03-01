from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import List, Dict, Union, get_type_hints, Callable

import numpy as np

import torch

TensorArray = Union[np.array, torch.Tensor]


@dataclass
class Multitype:

    @classmethod
    def stack_tensor(cls, value_list: List[Multitype], dim: int = 0) -> Multitype:
        res = cls()
        for field_ in get_type_hints(cls):
            tensors = [torch.as_tensor(getattr(value, field_))
                       for value in value_list
                       if getattr(value, field_) is not None]

            value = torch.stack(tensors, dim=dim) if tensors else None
            setattr(res, field_, value)

        return res

    @classmethod
    def cat_tensor(cls, value_list: List[Multitype], dim: int = 0) -> Multitype:
        res = cls()
        for field_ in get_type_hints(cls):
            tensors = [torch.as_tensor(getattr(value, field_))
                       for value in value_list
                       if getattr(value, field_) is not None]

            value = torch.cat(tensors, dim=dim) if tensors else None
            setattr(res, field_, value)

        return res

    @property
    def batch_size(self):
        value = None
        for field_ in fields(self):
            field_value = getattr(self, field_.name)
            if value is None and field_value is not None:
                value = field_value.shape[0]
            elif value is not None and field_value is not None:
                assert value == field_value.shape[0], "Different types have different batch sizes"

        return value

    def tensor(self, device: str = 'cpu'):
        res = type(self)()
        for field_ in fields(self):
            value = getattr(self, field_.name)
            tensor = torch.as_tensor(value).to(device) if value is not None else None
            setattr(res, field_.name, tensor)
        return res

    def __getitem__(self, item):
        res = type(self)()
        for field_ in fields(self):
            value = getattr(self, field_.name)
            part = value[item] if value is not None else None
            setattr(res, field_.name, part)
        return res

    def apply(self, func: Callable[[TensorArray], TensorArray]):
        res = type(self)()
        for field_ in fields(self):
            value = getattr(self, field_.name)
            new_field = func(value) if value is not None else None
            setattr(res, field_.name, new_field)
        return res


@dataclass
class Observation(Multitype):
    vector: TensorArray = None
    rays: TensorArray = None
    buffer: TensorArray = None
    image: TensorArray = None


@dataclass
class Action(Multitype):
    continuous: TensorArray = None
    discrete: TensorArray = None


Reward = TensorArray
LogProb = TensorArray
Value = TensorArray
Done = TensorArray


@dataclass
class MemoryRecord:
    obs: Observation
    action: Action
    reward: Reward
    # logprob: LogProb
    value: Value
    done: Done


@dataclass
class AgentMemoryBuffer:  # TODO: change to a list of records?
    observations: List[Observation] = field(default_factory=list)
    actions: List[Action] = field(default_factory=list)
    rewards: List[Reward] = field(default_factory=list)
    # logprobs: List[LogProb] = field(default_factory=list)
    values: List[Value] = field(default_factory=list)
    dones: List[Done] = field(default_factory=list)

    def append(self, record):
        self.observations.append(record.obs)
        self.actions.append(record.action)
        self.rewards.append(record.reward)
        # self.logprobs.append(record.logprob)
        self.values.append(record.value)
        self.dones.append(record.done)


@dataclass
class MemoryBuffer:
    data: Dict[str, AgentMemoryBuffer] = field(default_factory=dict)

    def append(self,
               obs: Dict[str, Observation],
               action: Dict[str, Action],
               reward: Dict[str, Reward],
               # logprobs: Dict[str, LogProb],
               value: Dict[str, Value],
               done: Dict[str, Done]):

        for agent_id in obs:  # Assume the keys are identical
            record = MemoryRecord(obs[agent_id],
                                  action[agent_id],
                                  reward[agent_id],
                                  # logprobs[agent_id],
                                  value[agent_id],
                                  done[agent_id])

            self.data.setdefault(agent_id, AgentMemoryBuffer()).append(record)

    def tensorify(self) -> Dict[str, MemoryRecord]:
        result = {}
        for agent_id, agent_buffer in self.data.items():  # str -> AgentMemoryBuffer
            result[agent_id] = MemoryRecord(
                obs=Observation.stack_tensor(agent_buffer.observations),
                action=Action.stack_tensor(agent_buffer.actions),
                reward=torch.tensor(agent_buffer.rewards),
                # logprob=torch.tensor(agent_buffer.logprobs),
                value=torch.tensor(agent_buffer.values),
                done=torch.tensor(agent_buffer.dones)
            )
        return result

    def crowd_tensorify(self) -> MemoryRecord:
        tensor_data = self.tensorify().values()
        return MemoryRecord(
            obs=Observation.cat_tensor([agent_buffer.obs for agent_buffer in tensor_data]),
            action=Action.cat_tensor([agent_buffer.action for agent_buffer in tensor_data]),
            reward=torch.cat([agent_buffer.reward for agent_buffer in tensor_data]),
            # logprob=torch.cat([agent_buffer.logprob for agent_buffer in tensor_data]),
            value=torch.cat([agent_buffer.value for agent_buffer in tensor_data]),
            done=torch.cat([agent_buffer.done for agent_buffer in tensor_data])
        )

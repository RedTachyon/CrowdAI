from typing import Dict, Sequence

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.side_channel import (
    SideChannel,
    IncomingMessage,
    OutgoingMessage,
)
import numpy as np
import uuid
from utils import np_float, concat_dicts


def parse_side_message(msg: str) -> Dict[str, np.ndarray]:
    """Parses a message from StatsChannel"""
    if msg == "": return {}
    lines = msg.split('\n')
    out = {line.split(' ')[0]: np_float(float(line.split(' ')[1])) for line in lines}
    return out

class StatsChannel(SideChannel):

    def __init__(self) -> None:
        super().__init__(uuid.UUID("621f0a70-4f87-11ea-a6bf-784f4387d1f7"))
        # self.last_msg = ""
        self.msg_buffer = []

    def on_message_received(self, msg: IncomingMessage) -> None:
        """
        Note: We must implement this method of the SideChannel interface to
        receive messages from Unity
        """
        # We simply read a string from the message and print it.
        text = msg.read_string()
        # self.last_msg = text
        self.msg_buffer.append(text)
        # print(self.last_msg)

    def send_string(self, data: str) -> None:
        # Unused, mostly just pro forma
        # Add the string to an OutgoingMessage
        msg = OutgoingMessage()
        msg.write_string(data)
        # We call this method to queue the data we want to send
        super().queue_message_to_send(msg)

    def parse_info(self, clear: bool = True) -> Dict[str, np.ndarray]:
        dicts: Sequence = [parse_side_message(msg) for msg in self.msg_buffer]
        result = concat_dicts(dicts)
        if clear:
            self.msg_buffer = []
        return result

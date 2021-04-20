from __future__ import annotations

import random
from collections import namedtuple
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional

import numpy as np
from gym.spaces import Box, Discrete
from numba import vectorize
from scipy.ndimage import convolve

import matplotlib as mpl
import matplotlib.pyplot as plt

from coltra.buffers import Observation, Action
from coltra.envs.base_env import MultiAgentEnv
from coltra.envs.subproc_vec_env import SubprocVecEnv

GO_FORWARD = 0
GO_BACKWARD = 1
GO_LEFT = 2
GO_RIGHT = 3
ROT_LEFT = 4
ROT_RIGHT = 5
SHOOT = 6
NOOP = 7

# Kernel to compute the number of surrounding apples - not the same as in DM paper
NEIGHBOR_KERNEL = np.array(
    [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 0, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
    ]
)

DIRECTIONS = [
    (-1, 0),  # UP
    (0, 1),  # RIGHT
    (1, 0),  # DOWN
    (0, -1),  # LEFT
]

# noinspection PyUnresolvedReferences
cmap = mpl.colors.ListedColormap(["white", "green", "red", "gray"])

Board = np.ndarray


class Position(namedtuple("Position", ["i", "j"])):
    def __add__(self, other):
        if isinstance(other, Position):
            return Position(i=self.i + other.i, j=self.j + other.j)
        elif isinstance(other, int):
            return Position(i=self.i + other, j=self.j + other)
        elif isinstance(other, tuple):
            return Position(i=self.i + other[0], j=self.j + other[1])
        else:
            raise ValueError(
                "A Position can only be added to an int or another Position"
            )

    def __mul__(self, other):
        if isinstance(other, int):
            return Position(i=self.i * other, j=self.j * other)
        else:
            raise ValueError("A Position can only be multiplied by an int")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        if isinstance(other, Position):
            return Position(i=self.i - other.i, j=self.j - other.j)
        elif isinstance(other, int):
            return Position(i=self.i - other, j=self.j - other)
        elif isinstance(other, tuple):
            return Position(i=self.i - other[0], j=self.j - other[1])
        else:
            raise ValueError(
                "A Position can only be added to an int or another Position"
            )

    def __neg__(self):
        return Position(-self.i, -self.j)

    def __eq__(self, other) -> bool:
        if isinstance(other, Position):
            return self.i == other.i and self.j == other.j
        if isinstance(other, (tuple, list)):
            assert (
                len(other) == 2
            ), "Position equality comparison must be with a length-2 sequence"
            return self.i == other[0] and self.j == other[1]
        raise ValueError("A Position can only be compared with a Position-like item.")

    def is_between(self, pos1: Position, pos2: Position):
        """Checks whether a position is between two antipodal bounding box coordinates.
        Inclusive on all sides
        """
        min_i, max_i = sorted((pos1.i, pos2.i))
        min_j, max_j = sorted((pos1.j, pos2.j))
        return (min_i <= self.i <= max_i) and (min_j <= self.j <= max_j)


def fast_rot90(array: np.ndarray, k: int):
    """Rotates an array 90 degrees on the first two axes.
    Equivalent to np.rot90(array, k, (0, 1)), but faster.
    Might make some arrays non-contiguous, not sure if that
    will be a problem. If we need it to be contiguous, performance
    will drop again.
    UPDATE: rot90 also makes in noncontiguous, nvm it's just faster.
    Still, we need to profile both options in a full context.
    """
    if (k % 4) == 0:
        return array[:]
    elif (k % 4) == 1:
        return array.T[::-1, :]
    elif (k % 4) == 2:
        return array[::-1, ::-1]
    else:
        return array.T[:, ::-1]


@dataclass
class Walker:
    pos: Position
    rot: int
    frozen: int = 0


def in_bounds(pos: Position, size: Position) -> bool:
    """Checks whether the position fits within the board of a given size"""
    (i, j) = pos
    (max_i, max_j) = size
    return 0 <= i < max_i and 0 <= j < max_j


def get_neighbors(pos: Position, size: Position, radius: int = 1) -> List[Position]:
    """Get a list of neighboring positions (including self) that are still within
    the boundaries of the board"""
    if radius == 1:  # Default
        increments = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
        return [pos + inc for inc in increments if in_bounds(pos + inc, size)]
    else:
        row0, col0 = pos
        return [
            Position(row1, col1)
            for col1 in range(size[1])
            for row1 in range(size[0])
            if abs(row1 - row0) + abs(col1 - col0) <= radius
        ]


def create_board(size: Tuple[int, int], apples: List[Tuple[int, int]]) -> Board:
    """Creates an empty board with a number of cross-shaped apple patterns"""
    board = np.zeros(size, dtype=np.int8)

    for pos in apples:
        for (i, j) in get_neighbors(Position(*pos), Position(*size), radius=1):
            board[i, j] = 1

    return board


def random_board(size: Tuple[int, int], prob: float = 0.1) -> Board:
    """Creates a board with each square having a probability of containing an apple"""
    prob_map = np.random.rand(*size)
    return (prob_map < prob).astype(np.int8)


def random_crosses(size: Position,
                   num_crosses: int = 10) -> Board:
    """Creates a board with random cross-shaped apple patterns"""
    all_positions = [(row, col) for col in range(size[1]) for row in range(size[0])]
    random_idx = np.random.choice(
                      range(len(all_positions)),
                      num_crosses,
                      replace=False)
    initial_apples = [all_positions[i] for i in random_idx]
    board = create_board(size, initial_apples)
    return board


def agent_initial_position(i: int, total: int) -> Position:
    """
    Finds a starting position for an i-th agent with $total agents overall.
    The idea is that they're aligned in a square in the (0, 0) corner.
    The size of the square is determined by the number of agents.
    For example, with 10 agents, they will be in a 4x4 square, with 6 spots unfilled.
    With 16 agents, they're in a full 4x4 square.
    """
    layout_base = int(np.ceil(np.sqrt(total)))
    idx_map = np.arange(layout_base ** 2).reshape(layout_base, layout_base)
    (rows, cols) = np.where(idx_map == i)
    row, col = rows[0], cols[0]
    return Position(row, col) + (1, 1)


@vectorize
def regrow_prob(n: int) -> float:
    if n == 0:
        return 0.0
    elif n == 1 or n == 2:
        return 0.01
    elif n == 3 or n == 4:
        return 0.05
    else:
        return 0.1


def regenerate_apples(board: Board) -> Board:
    kernel = NEIGHBOR_KERNEL
    neighbor_map: np.ndarray = convolve(board, kernel, mode="constant")
    # The types here are fine, but numba.vectorize preserves the signature
    # noinspection PyTypeChecker
    prob_map = regrow_prob(neighbor_map)

    rand = np.random.rand(*neighbor_map.shape)
    regen_map = rand < prob_map
    # updated_board = np.clip(board + regen_map, 0, 1).astype(int)
    updated_board = (board + regen_map > 0).astype(int)

    return updated_board


def walls_board(size: Tuple[int, int]) -> Board:
    walls = np.zeros(size)
    row_bound, col_bound = size
    for i in range(row_bound):
        walls[i, 0] = 1
        walls[i, col_bound - 1] = 1
    for j in range(col_bound):
        walls[0, j] = 1
        walls[row_bound - 1, j] = 1
    return walls


class HarvestGame:
    def __init__(
        self,
        num_agents: int,
        size: Tuple[int, int],
        sight_width: int = 10,  # default value from DM paper
        sight_dist: int = 20,  # default value from DM paper
        beam_width: int = 5,
        beam_dist: int = 10,
        num_crosses: int = 10,  # number of apple-crosses to start with
    ):

        self.num_agents = num_agents
        self.size = Position(*size)
        self.sight_width = sight_width
        self.sight_dist = sight_dist
        self.beam_width = beam_width
        self.beam_dist = beam_dist
        self.num_crosses = num_crosses

        self.board = np.array([[]])
        self.agents: Dict[str, Walker] = dict()
        self.reputation: Dict[str, int] = dict()

        self.walls = walls_board(self.size)
        # self.walls = np.zeros(self.size)

        self.reset()

    def __repr__(self) -> str:
        return f"HarvestGame(num_agents={self.num_agents}, size={self.size})"

    def reset(self):
        self.board = random_crosses(self.size, self.num_crosses)
        self.agents = {
            f"Agent{i}": Walker(
                pos=agent_initial_position(i, self.num_agents),
                rot=1 + np.random.randint(2),
            )
            # start facing east or south
            for i in range(self.num_agents)
        }

        # remove apples from cells which already contain agents
        for name, agent in self.agents.items():
            self.board[agent.pos] = 0

        self.board[self.walls.astype(bool)] = 0

        self.reputation = {f"Agent{i}": 0 for i in range(self.num_agents)}

    def render(self, ax: Optional[plt.Axes] = None):
        """Writes the image to a pyplot axes"""
        if ax is None:
            fig, ax = plt.subplots()
        board = self.board.copy()
        for name, agent in self.agents.items():
            board[agent.pos] = 2

        board[self.walls.astype(bool)] = 3
        ax.cla()
        ax.imshow(board, cmap=cmap)

    def is_free(self, pos: Position) -> bool:  # TODO: Check if it's on a wall
        """Checks whether the position is within bounds, and unoccupied"""
        if not in_bounds(pos, self.size):
            return False

        for agent_id, agent in self.agents.items():
            if agent.pos == pos:
                return False

        if self.walls[(pos.i, pos.j)]:
            return False

        return True

    def _move_agent(self, agent_id: str, new_pos: Position):
        """Maybe moves an agent to a new position"""
        agent = self.agents[agent_id]
        if self.is_free(new_pos):
            agent.pos = new_pos

    def _set_rotation(self, agent_id: str, rot: int):
        """Debugging/testing method"""
        if rot not in range(4):
            raise ValueError("rot must be % 4")
        agent = self.agents[agent_id]
        agent.rot = rot

    def _rotate_agent(self, agent_id: str, direction: int):
        """Rotates an agent left or right"""
        if direction not in (-1, 1):
            raise ValueError("direction must be in -1, 1")
        agent = self.agents[agent_id]
        # -1: rotate left
        #  1: rotate right
        agent.rot = (agent.rot + direction) % 4

    def process_action(self, agent_id: str, action: int) -> float:
        """Processes a single action for a single agent"""
        agent = self.agents[agent_id]
        pos, rot = agent.pos, agent.rot
        if agent.frozen > 0:
            # if the agent is frozen, should we avoid updating the gradient?
            agent.frozen -= 1
            return 0.0
        if action == GO_FORWARD:
            # Go forward
            new_pos = pos + DIRECTIONS[rot]
            self._move_agent(agent_id, new_pos)
        elif action == GO_BACKWARD:
            # Go backward
            new_pos = pos - DIRECTIONS[rot]
            self._move_agent(agent_id, new_pos)
        elif action == GO_LEFT:
            # Go left
            new_pos = pos + DIRECTIONS[(rot - 1) % 4]
            self._move_agent(agent_id, new_pos)
        elif action == GO_RIGHT:
            # Go right
            new_pos = pos + DIRECTIONS[(rot + 1) % 4]
            self._move_agent(agent_id, new_pos)
        elif action == ROT_LEFT:
            # Rotate left
            self._rotate_agent(agent_id, -1)
        elif action == ROT_RIGHT:
            # Rotate right
            self._rotate_agent(agent_id, 1)
        elif action == SHOOT:
            # Shoot a beam
            affected_agents = self.get_affected_agents(agent_id)
            for _agent in affected_agents:
                _agent.frozen = 25

            self.reputation[agent_id] += 1
            # does reputation increase after shooting?
            # see notebook for weird results
        elif action == NOOP:
            # No-op
            return 0.0
        else:
            raise ValueError(f"Invalid action {action}")

        current_pos = self.agents[agent_id].pos
        if self.board[current_pos]:  # apple in new cell
            self.board[current_pos] = 0
            return 1.0
        else:  # no apple in new cell
            return 0.0

    def get_beam_bounds(self, agent_id: str) -> Tuple[Position, Position]:
        """Returns indices of the (top left, bottom right) (inclusive) boundaries
        of an agent's vision."""
        agent = self.agents[agent_id]
        rot = agent.rot

        # DIRECTIONS[rot] defines the forward direction
        forward = Position(*DIRECTIONS[rot])
        # DIRECTIONS[rot-1] is the agent's left side
        left = Position(*DIRECTIONS[(rot - 1) % 4])
        # DIRECTIONS[rot+1] is the agent's right side
        right = Position(*DIRECTIONS[(rot + 1) % 4])

        bound1 = agent.pos + (
            (forward * self.beam_dist) + (left * self.beam_width)
        )  # Forward-left corner
        bound2 = agent.pos + (right * self.beam_width)  # Backward-right corner

        return bound1, bound2

    def get_affected_agents(self, agent_id: str) -> List[Walker]:
        """Returns a list of agents caught in the ray"""
        bound1, bound2 = self.get_beam_bounds(agent_id)
        return [
            other_agent
            for other_name, other_agent in self.agents.items()
            if other_name != agent_id and other_agent.pos.is_between(bound1, bound2)
        ]

    def get_agent_obs(self, agent_id: str) -> np.ndarray:
        agent = self.agents[agent_id]

        apple_board = self.board
        agent_board = np.zeros_like(self.board)
        for other_agent_id, other_agent in self.agents.items():
            agent_board[other_agent.pos] = 1

        wall_board = self.walls

        # Add any extra layers before this line

        full_board = np.stack(
            [apple_board, agent_board, wall_board], axis=-1
        )  # H x W x 3
        rot = agent.rot

        # Rotate the board so that the agent is always pointing up.
        # I checked, the direction should be correct - still tests, will be good
        board = np.rot90(full_board, rot)

        max_i, max_j = board.shape[:2]

        if rot == 0:
            agent_i, agent_j = agent.pos
        elif rot == 1:
            agent_i, agent_j = max_i - agent.pos[1] - 1, agent.pos[0]
        elif rot == 2:
            agent_i, agent_j = max_i - agent.pos[0] - 1, max_j - agent.pos[1] - 1
        else:
            agent_i, agent_j = agent.pos[1], max_j - agent.pos[0] - 1

        # Horizontal bounds
        bounds_i = (agent_i - self.sight_dist + 1, agent_i + 1)
        (bound_up, bound_down) = bounds_i
        bounds_i_clipped = np.clip(bounds_i, 0, board.shape[0])

        bounds_j = (agent_j - self.sight_width, agent_j + self.sight_width + 1)
        (bound_left, bound_right) = bounds_j
        bounds_j_clipped = np.clip(bounds_j, 0, board.shape[1])

        base_slice = board[
            slice(*bounds_i_clipped), slice(*bounds_j_clipped)
        ]  # <= (20, 21)
        # breakpoint()
        if bound_up < 0:
            padding = np.zeros((-bound_up, base_slice.shape[1], base_slice.shape[2]))
            base_slice = np.concatenate([padding, base_slice], axis=0)
        if bound_left < 0:
            padding = np.zeros((base_slice.shape[0], -bound_left, base_slice.shape[2]))
            base_slice = np.concatenate([padding, base_slice], axis=1)
        if bound_right > max_j:  # board.shape[1]:
            padding = np.zeros(
                (base_slice.shape[0], bound_right - board.shape[1], base_slice.shape[2])
            )
            base_slice = np.concatenate([base_slice, padding], axis=1)
        if bound_down > board.shape[0]:
            raise ValueError("WTF")

        return base_slice  # 20 x 21


class HarvestEnv(MultiAgentEnv):
    def __init__(self, config: Dict[str, str], **kwargs):
        super().__init__()
        self.config = config
        self.time = 0
        self.game = HarvestGame(**kwargs)
        self.original_board = np.copy(self.game.board)

        self.observation_space = Box(0., 1., (self.game.sight_dist * (self.game.sight_width * 2 + 1),))
        self.action_space = Discrete(8)

    def reset(self, *args, **kwargs) -> Dict[str, np.ndarray]:
        self.game.reset()
        self.original_board = np.copy(self.game.board)
        self.time = 0
        return {
            agent_id: Observation(vector=self.game.get_agent_obs(agent_id).ravel().astype(np.float32))
            for agent_id, _ in self.game.agents.items()
        }

    def step(self, actions: Dict[str, Action]):
        self.time += 1

        rewards = {agent_id: 0.0 for agent_id in self.game.agents}
        action_pairs = list(actions.items())
        random.shuffle(action_pairs)
        for (agent_id, action) in actions.items():
            reward = self.game.process_action(agent_id, action.discrete)
            rewards[agent_id] += reward

        done = {agent_id: self.time >= 1000 for agent_id, _ in self.game.agents.items()}
        # done["__all__"] = self.time > 1000  # Required for rllib (I think)

        num_shots = sum(1 for key, action in actions.items() if action == SHOOT)
        info = {"m_shots": np.array([num_shots])}
        self.game.board = regenerate_apples(self.game.board)
        self.game.board = self.game.board * self.original_board

        obs = {
            agent_id: Observation(vector=self.game.get_agent_obs(agent_id).ravel().astype(np.float32))
            for agent_id, _ in self.game.agents.items()
        }

        return obs, rewards, done, info

    def render(self, *args, **kwargs):
        return self.game.render(*args, **kwargs)

    @classmethod
    def get_venv(cls, workers: int = 8, *args, **kwargs) -> SubprocVecEnv:
        venv = SubprocVecEnv([
            cls.get_env_creator(*args, **kwargs)
            for i in range(workers)
        ])
        return venv

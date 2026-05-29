"""Tabular Independent Q-Learning (IQL) baseline for BooleanGameEnv."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class IQLConfig:
    """Hyperparameters for tabular IQL."""

    learning_rate: float = 0.1
    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 2000


class TabularIQLAgent:
    """Single-agent tabular Q-learner used in a multi-agent IQL setup."""

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        config: IQLConfig,
        seed: int | None = None,
    ) -> None:
        if n_states <= 0:
            raise ValueError("n_states must be > 0")
        if n_actions <= 0:
            raise ValueError("n_actions must be > 0")

        self.n_states = n_states
        self.n_actions = n_actions
        self.config = config
        self.q_table = np.zeros((n_states, n_actions), dtype=np.float64)
        self._rng = np.random.default_rng(seed)

    def epsilon(self, step: int) -> float:
        if self.config.epsilon_decay_steps <= 0:
            return self.config.epsilon_end
        progress = min(1.0, step / float(self.config.epsilon_decay_steps))
        return self.config.epsilon_start + progress * (
            self.config.epsilon_end - self.config.epsilon_start
        )

    def select_action(self, state: int, step: int) -> int:
        eps = self.epsilon(step)
        if self._rng.random() < eps:
            return int(self._rng.integers(self.n_actions))

        q_vals = self.q_table[state]
        max_q = np.max(q_vals)
        # Random tie-break among best actions.
        best_actions = np.flatnonzero(q_vals == max_q)
        return int(self._rng.choice(best_actions))

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
    ) -> None:
        best_next = 0.0 if done else float(np.max(self.q_table[next_state]))
        target = reward + self.config.gamma * best_next
        td_error = target - self.q_table[state, action]
        self.q_table[state, action] += self.config.learning_rate * td_error


def binary_vector_to_index(bits: np.ndarray) -> int:
    """Converts a binary vector (e.g. [1,0,1]) into an integer index."""

    flat = np.asarray(bits, dtype=np.int8).reshape(-1)
    idx = 0
    for bit in flat:
        idx = (idx << 1) | int(bit)
    return idx


def index_to_binary_vector(index: int, length: int) -> np.ndarray:
    """Converts an integer action index back to a binary action vector."""

    if length <= 0:
        raise ValueError("length must be > 0")
    if index < 0 or index >= (1 << length):
        raise ValueError(f"index must be in [0, {1 << length}), got {index}")

    out = np.zeros(length, dtype=np.int8)
    for i in range(length - 1, -1, -1):
        out[i] = index & 1
        index >>= 1
    return out

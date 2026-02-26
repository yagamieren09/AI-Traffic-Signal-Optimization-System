"""
Tabular Q-Learning Agent for multi-intersection traffic control.

Uses INDEPENDENT Q-learners (one per intersection) to avoid the
exponential joint action space. Each agent sees only its local
discretized state and controls its own phase.

State per intersection (6-tuple):
    (q_bin_ns, q_bin_ew, w_bin_ns, w_bin_ew, phase, locked)
    - q_bin_ns/ew : avg queue bins for N-S and E-W directions (0-4)
    - w_bin_ns/ew : avg wait bins (0-4)
    - phase       : 0 or 1
    - locked      : 1 if min-green not yet satisfied, else 0

Action per intersection: {0, 1} (desired phase)
"""

import numpy as np
from environment.traffic_env import (
    NUM_INTERSECTIONS, NUM_DIRECTIONS, NUM_PHASES, MIN_GREEN_STEPS
)

QUEUE_BINS = 5
WAIT_BINS  = 5


def _obs_to_per_intersection(obs: np.ndarray):
    return [obs[i*11:(i+1)*11] for i in range(NUM_INTERSECTIONS)]


def _discretize_intersection(chunk: np.ndarray) -> tuple:
    """
    chunk[0:4] = norm queue per direction
    chunk[4:8] = avg wait per direction
    chunk[8]   = phase
    chunk[9]   = steps_ratio (1.0 = unlocked)
    chunk[10]  = neighbor (unused in local discretization for simplicity)
    """
    # N-S directions = 0,1;  E-W = 2,3
    q_ns = float(np.mean(chunk[0:2]))
    q_ew = float(np.mean(chunk[2:4]))
    w_ns = float(np.mean(chunk[4:6]))
    w_ew = float(np.mean(chunk[6:8]))
    phase  = int(round(chunk[8]))
    locked = int(float(chunk[9]) < 1.0)

    qb_ns = min(int(q_ns * QUEUE_BINS), QUEUE_BINS - 1)
    qb_ew = min(int(q_ew * QUEUE_BINS), QUEUE_BINS - 1)
    wb_ns = min(int(w_ns * WAIT_BINS),  WAIT_BINS  - 1)
    wb_ew = min(int(w_ew * WAIT_BINS),  WAIT_BINS  - 1)
    return (qb_ns, qb_ew, wb_ns, wb_ew, phase, locked)


class SingleIntersectionAgent:
    def __init__(
        self,
        lr:         float = 0.3,
        gamma:      float = 0.95,
        eps:        float = 1.0,
        eps_end:    float = 0.05,
        eps_decay:  float = 0.995,
        rng:        np.random.Generator = None,
    ):
        self.lr        = lr
        self.gamma     = gamma
        self.eps       = eps
        self.eps_end   = eps_end
        self.eps_decay = eps_decay
        self.rng       = rng or np.random.default_rng(0)
        self.Q: dict[tuple, np.ndarray] = {}

    def _get_q(self, s: tuple) -> np.ndarray:
        if s not in self.Q:
            self.Q[s] = np.zeros(NUM_PHASES, dtype=np.float64)
        return self.Q[s]

    def select_action(self, state: tuple, greedy: bool = False) -> int:
        if not greedy and self.rng.random() < self.eps:
            return int(self.rng.integers(0, NUM_PHASES))
        return int(np.argmax(self._get_q(state)))

    def update(self, s: tuple, a: int, r: float, ns: tuple):
        q_cur  = self._get_q(s)[a]
        q_next = np.max(self._get_q(ns))
        td     = np.clip(r + self.gamma * q_next - q_cur, -5.0, 5.0)
        self._get_q(s)[a] += self.lr * td

    def decay_epsilon(self):
        self.eps = max(self.eps_end, self.eps * self.eps_decay)

    def num_states(self) -> int:
        return len(self.Q)


class QLearningAgent:
    """
    Wrapper: one independent Q-learning agent per intersection.
    Exposes the same interface as the rest of the codebase.
    """

    def __init__(
        self,
        lr_start:   float = 0.3,
        lr_end:     float = 0.05,
        gamma:      float = 0.95,
        eps_start:  float = 1.0,
        eps_end:    float = 0.05,
        eps_decay:  float = 0.995,
        seed:       int   = 42,
    ):
        self.lr_start  = lr_start
        self.lr_end    = lr_end
        self.eps       = eps_start
        self.eps_end   = eps_end
        self.eps_decay = eps_decay
        master_rng     = np.random.default_rng(seed)
        self.agents    = [
            SingleIntersectionAgent(
                lr=lr_start, gamma=gamma,
                eps=eps_start, eps_end=eps_end, eps_decay=eps_decay,
                rng=np.random.default_rng(int(master_rng.integers(0, 2**31))),
            )
            for _ in range(NUM_INTERSECTIONS)
        ]

    def _states(self, obs: np.ndarray) -> list[tuple]:
        chunks = _obs_to_per_intersection(obs)
        return [_discretize_intersection(c) for c in chunks]

    def select_action(self, obs: np.ndarray, greedy: bool = False) -> list[int]:
        states = self._states(obs)
        return [ag.select_action(s, greedy=greedy) for ag, s in zip(self.agents, states)]

    def update(self, obs: np.ndarray, action: list[int], reward: float, next_obs: np.ndarray):
        """Distribute shared reward equally to each local agent."""
        states      = self._states(obs)
        next_states = self._states(next_obs)
        r_local     = reward  # same normalized reward for all (simple shared reward)
        for ag, s, a, ns in zip(self.agents, states, action, next_states):
            ag.update(s, a, r_local, ns)

    def decay_epsilon(self):
        for ag in self.agents:
            ag.decay_epsilon()
        self.eps = self.agents[0].eps

    def update_lr(self, episode: int, total_episodes: int):
        frac = episode / max(total_episodes - 1, 1)
        lr   = self.lr_start + frac * (self.lr_end - self.lr_start)
        for ag in self.agents:
            ag.lr = lr

    def num_states_visited(self) -> int:
        return sum(ag.num_states() for ag in self.agents)

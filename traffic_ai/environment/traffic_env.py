"""
Traffic Signal Environment
4 connected intersections, each with 4 directions (N/S/E/W).
Poisson arrivals, queue-based storage, normalized state.
"""

import numpy as np
from collections import deque

# ─── Constants ────────────────────────────────────────────────────────────────
NUM_INTERSECTIONS = 4
NUM_DIRECTIONS    = 4          # N, S, E, W  → indices 0-3
MAX_QUEUE         = 20         # cap for normalization
MAX_WAIT          = 100        # steps, cap for normalization
MIN_GREEN_STEPS   = 5          # minimum steps a phase must stay green
NUM_PHASES        = 2          # phase 0: N-S green; phase 1: E-W green
SWITCH_PENALTY    = 0.05       # small reward penalty for switching phase

# Direction connectivity between intersections
# intersection layout (grid):
#   0 — 1
#   |   |
#   2 — 3
# E output of 0 feeds into W input of 1, etc.
NEIGHBORS = {
    0: [1, 2],
    1: [0, 3],
    2: [0, 3],
    3: [1, 2],
}


class Intersection:
    def __init__(self, idx: int, arrival_rate: float, rng: np.random.Generator):
        self.idx          = idx
        self.arrival_rate = arrival_rate
        self.rng          = rng

        # queues[direction] = deque of waiting times
        self.queues: list[deque] = [deque() for _ in range(NUM_DIRECTIONS)]
        self.phase            = 0        # current signal phase
        self.steps_in_phase   = 0        # how many steps current phase has been active
        self.total_wait       = 0.0      # accumulated waiting time this episode
        self.total_throughput = 0        # vehicles that passed

    # ── helpers ───────────────────────────────────────────────────────────────
    def queue_len(self, direction: int) -> int:
        return len(self.queues[direction])

    def norm_queue(self, direction: int) -> float:
        return min(self.queue_len(direction), MAX_QUEUE) / MAX_QUEUE

    def avg_wait(self, direction: int) -> float:
        if not self.queues[direction]:
            return 0.0
        return min(np.mean(list(self.queues[direction])), MAX_WAIT) / MAX_WAIT

    def neighbor_congestion(self, all_intersections) -> float:
        vals = []
        for n_idx in NEIGHBORS[self.idx]:
            ni = all_intersections[n_idx]
            vals.extend([ni.norm_queue(d) for d in range(NUM_DIRECTIONS)])
        return float(np.mean(vals)) if vals else 0.0

    # ── step logic ────────────────────────────────────────────────────────────
    def add_arrivals(self):
        """Poisson arrivals for each direction."""
        for d in range(NUM_DIRECTIONS):
            n = self.rng.poisson(self.arrival_rate / NUM_DIRECTIONS)
            for _ in range(n):
                if self.queue_len(d) < MAX_QUEUE:
                    self.queues[d].append(0)   # new vehicle, wait = 0

    def process_departures(self, max_depart_per_dir: int = 2):
        """Green directions can discharge vehicles."""
        green_dirs = [0, 1] if self.phase == 0 else [2, 3]
        for d in green_dirs:
            for _ in range(max_depart_per_dir):
                if self.queues[d]:
                    self.queues[d].popleft()
                    self.total_throughput += 1

    def update_waiting(self):
        """Increment wait time for all queued vehicles."""
        for d in range(NUM_DIRECTIONS):
            for i in range(len(self.queues[d])):
                self.queues[d][i] = min(self.queues[d][i] + 1, MAX_WAIT)
                self.total_wait += 1

    def can_switch(self) -> bool:
        return self.steps_in_phase >= MIN_GREEN_STEPS

    def apply_action(self, action: int):
        """action: desired phase (0 or 1). Enforces min-green constraint."""
        if action != self.phase and self.can_switch():
            self.phase = action
            self.steps_in_phase = 0
        else:
            self.steps_in_phase += 1

    def reset(self):
        self.queues       = [deque() for _ in range(NUM_DIRECTIONS)]
        self.phase        = 0
        self.steps_in_phase = 0
        self.total_wait   = 0.0
        self.total_throughput = 0


class TrafficEnv:
    """
    Multi-intersection traffic environment.

    Observation per intersection (10 values, all in [0,1]):
        - norm_queue × 4 directions
        - avg_wait   × 4 directions
        - phase (0 or 1, cast to float)
        - steps_in_phase / MIN_GREEN_STEPS (capped at 1.0)
        - neighbor_congestion (scalar)
    Total obs dim per intersection = 11
    Total joint obs = 4 × 11 = 44

    Action space: 4 integers (one phase per intersection), each in {0, 1}
    """

    OBS_DIM = NUM_INTERSECTIONS * 11

    def __init__(self, arrival_rate: float = 0.8, seed: int = 42):
        self.arrival_rate = arrival_rate
        self.seed         = seed
        self.rng          = np.random.default_rng(seed)
        self.intersections: list[Intersection] = [
            Intersection(i, arrival_rate, np.random.default_rng(seed + i))
            for i in range(NUM_INTERSECTIONS)
        ]
        self.step_count = 0

    def _get_obs(self) -> np.ndarray:
        obs = []
        for inter in self.intersections:
            for d in range(NUM_DIRECTIONS):
                obs.append(inter.norm_queue(d))
            for d in range(NUM_DIRECTIONS):
                obs.append(inter.avg_wait(d))
            obs.append(float(inter.phase))
            obs.append(min(inter.steps_in_phase / MIN_GREEN_STEPS, 1.0))
            obs.append(inter.neighbor_congestion(self.intersections))
        return np.array(obs, dtype=np.float32)

    def _compute_reward(self, prev_phases: list[int]) -> float:
        reward = 0.0
        for i, inter in enumerate(self.intersections):
            avg_q = np.mean([inter.norm_queue(d) for d in range(NUM_DIRECTIONS)])
            avg_w = np.mean([inter.avg_wait(d)   for d in range(NUM_DIRECTIONS)])
            switched = float(inter.phase != prev_phases[i])
            reward -= (0.5 * avg_q + 0.3 * avg_w + SWITCH_PENALTY * switched)
        # normalize by num intersections so scale stays ~[-1, 0]
        reward /= NUM_INTERSECTIONS
        return float(np.clip(reward, -1.0, 0.0))

    def reset(self):
        self.rng = np.random.default_rng(self.seed)
        for i, inter in enumerate(self.intersections):
            inter.rng = np.random.default_rng(self.seed + i)
            inter.reset()
        self.step_count = 0
        return self._get_obs()

    def step(self, actions: list[int]):
        """
        actions: list of length NUM_INTERSECTIONS, each 0 or 1.
        Returns: obs, reward, done=False (caller controls episode length)
        """
        prev_phases = [inter.phase for inter in self.intersections]

        for inter, action in zip(self.intersections, actions):
            inter.add_arrivals()
            inter.apply_action(action)
            inter.process_departures()
            inter.update_waiting()

        self.step_count += 1
        obs    = self._get_obs()
        reward = self._compute_reward(prev_phases)
        return obs, reward, False

    # ── diagnostics ───────────────────────────────────────────────────────────
    def get_metrics(self) -> dict:
        total_q   = sum(sum(i.queue_len(d) for d in range(NUM_DIRECTIONS))
                        for i in self.intersections)
        total_thru = sum(i.total_throughput for i in self.intersections)
        total_wait = sum(i.total_wait       for i in self.intersections)
        return {
            "total_queue":      total_q,
            "total_throughput": total_thru,
            "total_wait":       total_wait,
            "avg_queue_per_intersection": total_q / NUM_INTERSECTIONS,
        }

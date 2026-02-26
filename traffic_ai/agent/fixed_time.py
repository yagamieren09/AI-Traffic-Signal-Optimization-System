"""
Fixed-time baseline controller.
Each intersection cycles between phase 0 and phase 1 on a fixed schedule,
while still respecting the MIN_GREEN_STEPS constraint.
"""

from environment.traffic_env import NUM_INTERSECTIONS, MIN_GREEN_STEPS, NUM_PHASES


class FixedTimeController:
    """
    Cycles phases with equal green duration = `cycle_half` steps per phase.
    cycle_half must be >= MIN_GREEN_STEPS to satisfy the constraint.
    """

    def __init__(self, cycle_half: int = 10):
        assert cycle_half >= MIN_GREEN_STEPS, (
            f"cycle_half ({cycle_half}) must be >= MIN_GREEN_STEPS ({MIN_GREEN_STEPS})"
        )
        self.cycle_half  = cycle_half
        self._phase      = [0] * NUM_INTERSECTIONS
        self._step_timer = [0] * NUM_INTERSECTIONS

    def select_action(self, obs=None) -> list[int]:
        """Returns desired phase for each intersection."""
        actions = []
        for i in range(NUM_INTERSECTIONS):
            actions.append(self._phase[i])
            self._step_timer[i] += 1
            if self._step_timer[i] >= self.cycle_half:
                self._phase[i]      = 1 - self._phase[i]
                self._step_timer[i] = 0
        return actions

    def reset(self):
        self._phase      = [0] * NUM_INTERSECTIONS
        self._step_timer = [0] * NUM_INTERSECTIONS

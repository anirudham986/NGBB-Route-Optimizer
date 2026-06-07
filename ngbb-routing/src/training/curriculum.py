"""Curriculum scheduler for ramping instance sizes during training.

Starts with small instances and gradually increases size to improve
generalization, following the schedule defined in the config.
"""


class CurriculumScheduler:
    """Ramps instance size range during training.

    Starts at size_min and linearly increases to size_max
    over a specified number of steps.
    """

    def __init__(self, size_min: int = 20, size_max: int = 50,
                 ramp_steps: int = 100_000, step_size: int = 5):
        self.size_min = size_min
        self.size_max = size_max
        self.ramp_steps = ramp_steps
        self.step_size = step_size

    def get_size_range(self, step: int) -> tuple[int, int]:
        """Get the (min, max) instance size for the current step.

        Args:
            step: Current training step.

        Returns:
            (current_min, current_max) size range.
        """
        progress = min(step / self.ramp_steps, 1.0)
        current_max = int(self.size_min + (self.size_max - self.size_min) * progress)
        # Round to step_size
        current_max = max(self.size_min, (current_max // self.step_size) * self.step_size)
        return self.size_min, current_max

    def get_difficulty(self, step: int) -> float:
        """Get current difficulty level as float in [0, 1]."""
        return min(step / self.ramp_steps, 1.0)

"""Seeded PRNG utilities for reproducible experiments.

Provides a consistent random number generator interface using NumPy
with explicit seed management for full reproducibility.
"""

import numpy as np


class SeededRNG:
    """Wrapper around NumPy's Generator for reproducible random sampling.

    Usage:
        rng = SeededRNG(42)
        value = rng.uniform(0, 100)
        integers = rng.integers(1, 10, size=5)
    """

    def __init__(self, seed: int):
        """Initialize with a specific seed.

        Args:
            seed: Integer seed for the PRNG. Same seed guarantees
                  identical sequences across runs.
        """
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def uniform(self, low: float = 0.0, high: float = 1.0,
                size: int | tuple[int, ...] | None = None) -> np.ndarray | float:
        """Sample uniform random values in [low, high)."""
        return self._rng.uniform(low, high, size=size)

    def integers(self, low: int, high: int,
                 size: int | tuple[int, ...] | None = None) -> np.ndarray | int:
        """Sample random integers in [low, high]."""
        return self._rng.integers(low, high, size=size, endpoint=True)

    def normal(self, loc: float = 0.0, scale: float = 1.0,
               size: int | tuple[int, ...] | None = None) -> np.ndarray | float:
        """Sample from normal distribution."""
        return self._rng.normal(loc, scale, size=size)

    def choice(self, a, size: int | None = None,
               replace: bool = True, p=None):
        """Random choice from array or range."""
        return self._rng.choice(a, size=size, replace=replace, p=p)

    def shuffle(self, x: np.ndarray) -> None:
        """Shuffle array in-place."""
        self._rng.shuffle(x)

    def fork(self, child_seed: int | None = None) -> "SeededRNG":
        """Create a child RNG with a derived seed.

        Useful for parallelism where each worker needs a distinct
        but reproducible stream.

        Args:
            child_seed: If None, derives from parent's next integer.
        """
        if child_seed is None:
            child_seed = int(self._rng.integers(0, 2**31))
        return SeededRNG(child_seed)

    @property
    def numpy_rng(self) -> np.random.Generator:
        """Access the underlying NumPy Generator directly."""
        return self._rng


def mulberry32(seed: int):
    """Port of the mulberry32 PRNG (matches JavaScript implementation).

    Returns a callable that produces floats in [0, 1) on each call.
    Used for cross-language reproducibility with the visualizer.

    Args:
        seed: 32-bit integer seed.

    Returns:
        Callable that returns the next pseudo-random float.
    """
    state = [seed & 0xFFFFFFFF]

    def _next() -> float:
        state[0] = (state[0] + 0x6D2B79F5) & 0xFFFFFFFF
        t = state[0]
        t = ((t ^ (t >> 15)) * (1 | t)) & 0xFFFFFFFF
        t = (t + ((t ^ (t >> 7)) * (61 | t))) & 0xFFFFFFFF
        t = (t ^ (t >> 14)) & 0xFFFFFFFF
        return t / 4294967296

    return _next

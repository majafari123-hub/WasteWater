"""Tiny backend for testing the harness; not a wastewater model."""

import numpy as np
from .backend import BackendStep


class MockBackend:
    state_size = 4
    action_size = 2

    def __init__(self):
        self.state = np.zeros(self.state_size)

    def reset(self, seed: int | None = None) -> BackendStep:
        self.state = np.array([1.0, 2.0, 3.0, 4.0])
        return BackendStep(self.state.copy(), {"cost": 0.0})

    def step(self, action: np.ndarray) -> BackendStep:
        self.state[:2] += np.asarray(action)
        return BackendStep(self.state.copy(), {"cost": float(np.sum(action**2))})

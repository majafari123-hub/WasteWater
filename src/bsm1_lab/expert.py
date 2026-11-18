"""Expert-controller boundary."""

from typing import Protocol
import numpy as np


class ExpertController(Protocol):
    def act(self, observation: np.ndarray, mask: np.ndarray) -> np.ndarray: ...


class FixedActionExpert:
    """Temporary interface smoke-test expert; replace with BSM1 baseline control."""

    def __init__(self, action: np.ndarray):
        self.action = np.asarray(action, dtype=float)

    def act(self, observation: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return self.action.copy()

"""Simulator boundary.

The controller must never receive ``state`` directly.  A backend may keep a
larger privileged state internally and return it only for evaluation/logging.
"""

from dataclasses import dataclass
from typing import Protocol
import numpy as np


@dataclass(frozen=True)
class BackendStep:
    state: np.ndarray
    metrics: dict[str, float]
    terminated: bool = False


class SimulatorBackend(Protocol):
    state_size: int
    action_size: int

    def reset(self, seed: int | None = None) -> BackendStep: ...

    def step(self, action: np.ndarray) -> BackendStep: ...

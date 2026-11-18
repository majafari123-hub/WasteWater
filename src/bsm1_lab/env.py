"""Controller-facing BSM1 environment."""

from dataclasses import dataclass
import numpy as np

from .backend import SimulatorBackend
from .observation import ObservationModel


@dataclass(frozen=True)
class EnvStep:
    observation: np.ndarray
    mask: np.ndarray
    reward: float
    terminated: bool
    info: dict


class BSM1Env:
    """Wrap a simulator while withholding privileged process state."""

    def __init__(self, backend: SimulatorBackend, observation_model: ObservationModel):
        self.backend = backend
        self.observation_model = observation_model
        self._state: np.ndarray | None = None

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
        self.observation_model.reset(seed)
        result = self.backend.reset(seed)
        self._state = result.state.copy()
        observation, mask = self.observation_model.observe(self._state)
        return observation, mask, {"ground_truth": self._state.copy(), **result.metrics}

    def step(self, action: np.ndarray) -> EnvStep:
        result = self.backend.step(np.asarray(action, dtype=float))
        self._state = result.state.copy()
        observation, mask = self.observation_model.observe(self._state)
        # The objective will be replaced by the BSM1 effluent/energy cost.
        reward = -float(result.metrics.get("cost", 0.0))
        return EnvStep(
            observation=observation,
            mask=mask,
            reward=reward,
            terminated=result.terminated,
            info={"ground_truth": self._state.copy(), **result.metrics},
        )

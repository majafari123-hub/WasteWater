"""Observation model used to enforce partial observability."""

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ObservationSpec:
    indices: tuple[int, ...]
    noise_std: float = 0.0
    dropout_probability: float = 0.0


class ObservationModel:
    def __init__(self, spec: ObservationSpec):
        self.spec = spec
        self.rng = np.random.default_rng()

    def reset(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    def observe(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return observation and the mask actually delivered to the controller."""
        obs = np.asarray(state, dtype=float)[list(self.spec.indices)].copy()
        mask = self.rng.random(obs.shape) >= self.spec.dropout_probability
        if self.spec.noise_std:
            obs += self.rng.normal(0.0, self.spec.noise_std, size=obs.shape)
        obs[~mask] = 0.0
        return obs, mask.astype(np.float32)

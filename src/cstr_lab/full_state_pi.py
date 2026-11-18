"""Full-state PI-style feedback controller for the three-state CSTR."""

from __future__ import annotations

import numpy as np


class FullStatePIController:
    """PI temperature control augmented with full-state feedback.

    The integral term acts on reactor-temperature error. Concentration and
    jacket temperature provide instantaneous state feedback, so this is a
    practical full-state PI controller rather than a temperature-only loop.
    """

    def __init__(
        self,
        target_state: tuple[float, float, float] = (0.8, 330.0, 300.0),
        kp_temperature: float = 2.0,
        ki_temperature: float = 0.15,
        k_concentration: float = 8.0,
        k_jacket: float = 0.35,
        action_bias: float = 300.0,
        action_low: float = 280.0,
        action_high: float = 330.0,
    ):
        self.target = np.asarray(target_state, dtype=float)
        self.kp_temperature = kp_temperature
        self.ki_temperature = ki_temperature
        self.k_concentration = k_concentration
        self.k_jacket = k_jacket
        self.action_bias = action_bias
        self.action_low = action_low
        self.action_high = action_high
        self.integral_temperature_error = 0.0

    def reset(self) -> None:
        self.integral_temperature_error = 0.0

    def act(self, state: np.ndarray, dt: float) -> float:
        ca, reactor_temperature, jacket_temperature = np.asarray(state, dtype=float)
        target_ca, target_temperature, target_jacket = self.target

        temperature_error = target_temperature - reactor_temperature
        concentration_error = target_ca - ca
        jacket_error = target_jacket - jacket_temperature
        self.integral_temperature_error += temperature_error * dt

        action = (
            self.action_bias
            + self.kp_temperature * temperature_error
            + self.ki_temperature * self.integral_temperature_error
            - self.k_concentration * concentration_error
            + self.k_jacket * jacket_error
        )
        return float(np.clip(action, self.action_low, self.action_high))

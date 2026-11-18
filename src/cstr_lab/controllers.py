"""Conventional baseline controllers for the CSTR."""

from __future__ import annotations


class PIDController:
    """Temperature PID using only the measured reactor temperature."""

    def __init__(
        self,
        setpoint: float = 330.0,
        kp: float = 2.0,
        ki: float = 0.15,
        kd: float = 0.0,
        action_bias: float = 300.0,
        action_low: float = 280.0,
        action_high: float = 330.0,
    ):
        self.setpoint = setpoint
        self.kp, self.ki, self.kd = kp, ki, kd
        self.action_bias = action_bias
        self.action_low, self.action_high = action_low, action_high
        self.integral = 0.0
        self.previous_error: float | None = None

    def reset(self) -> None:
        self.integral = 0.0
        self.previous_error = None

    def act(self, reactor_temperature: float, dt: float) -> float:
        error = self.setpoint - float(reactor_temperature)
        self.integral += error * dt
        derivative = 0.0 if self.previous_error is None else (error - self.previous_error) / dt
        self.previous_error = error
        action = self.action_bias + self.kp * error + self.ki * self.integral + self.kd * derivative
        return float(max(self.action_low, min(self.action_high, action)))

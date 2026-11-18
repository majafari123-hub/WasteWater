"""Three-state nonlinear exothermic CSTR simulator.

States:
    C_A: reactant concentration
    T_R: reactor temperature
    T_J: cooling-jacket temperature

The control is coolant supply temperature. Feed concentration and feed
temperature are stochastic disturbances, held constant over each time step.
The simulator returns full state for experiment logging; controllers should
receive observations through a separate observation function.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class CSTRConfig:
    q: float = 100.0
    volume: float = 100.0
    rho: float = 1000.0
    heat_capacity: float = 0.239
    delta_h: float = -5.0e4
    activation_over_r: float = 8750.0
    k0: float = 7.2e10
    ua: float = 5.0e4
    jacket_time_constant: float = 2.0
    feed_concentration: float = 1.0
    feed_temperature: float = 350.0
    action_low: float = 280.0
    action_high: float = 330.0
    temperature_limit: float = 400.0
    process_noise_std: tuple[float, float] = (0.015, 0.25)


@dataclass(frozen=True)
class CSTRStep:
    state: np.ndarray
    action: float
    feed_concentration: float
    feed_temperature: float
    time: float
    terminated: bool
    metrics: dict[str, float]


class CSTRSimulator:
    """Discrete-time wrapper around the nonlinear CSTR ODE."""

    state_names = ("C_A", "T_R", "T_J")
    action_name = "coolant_supply_temperature"

    def __init__(self, config: CSTRConfig | None = None, dt: float = 0.1):
        self.config = config or CSTRConfig()
        self.dt = float(dt)
        self.rng = np.random.default_rng()
        self.state = np.array([0.8, 330.0, 300.0], dtype=float)
        self.time = 0.0
        self.last_disturbance = (
            self.config.feed_concentration,
            self.config.feed_temperature,
        )

    def reset(self, seed: int | None = None, state: np.ndarray | None = None) -> CSTRStep:
        self.rng = np.random.default_rng(seed)
        self.time = 0.0
        self.state = np.array(
            [0.8, 330.0, 300.0] if state is None else state,
            dtype=float,
        )
        self.last_disturbance = (
            self.config.feed_concentration,
            self.config.feed_temperature,
        )
        return self._result(self.config.action_high)

    def observe(self, noise_std: float = 0.2, reveal_concentration: bool = False) -> np.ndarray:
        """Return an observation without exposing hidden state by default."""
        if reveal_concentration:
            values = self.state.copy()
            values += self.rng.normal(0.0, noise_std, size=3)
            return values
        return np.array([self.state[1] + self.rng.normal(0.0, noise_std)])

    def step(self, coolant_supply_temperature: float) -> CSTRStep:
        cfg = self.config
        action = float(np.clip(coolant_supply_temperature, cfg.action_low, cfg.action_high))

        caf = max(0.05, cfg.feed_concentration + self.rng.normal(0.0, cfg.process_noise_std[0]))
        tf = cfg.feed_temperature + self.rng.normal(0.0, cfg.process_noise_std[1])
        self.last_disturbance = (caf, tf)

        def rhs(_time: float, x: np.ndarray) -> np.ndarray:
            ca, tr, tj = x
            rate = cfg.k0 * np.exp(-cfg.activation_over_r / max(tr, 1.0)) * ca
            dca = cfg.q / cfg.volume * (caf - ca) - rate
            dtr = (
                cfg.q / cfg.volume * (tf - tr)
                + (-cfg.delta_h) * rate / (cfg.rho * cfg.heat_capacity)
                + cfg.ua * (tj - tr) / (cfg.rho * cfg.heat_capacity * cfg.volume)
            )
            dtj = (action - tj) / cfg.jacket_time_constant
            return np.array([dca, dtr, dtj])

        solution = solve_ivp(
            rhs,
            (self.time, self.time + self.dt),
            self.state,
            method="RK45",
            rtol=1e-6,
            atol=1e-8,
        )
        if not solution.success:
            raise RuntimeError(f"CSTR integration failed: {solution.message}")

        self.state = solution.y[:, -1]
        self.time += self.dt
        return self._result(action, caf=caf, tf=tf)

    def _result(
        self,
        action: float,
        caf: float | None = None,
        tf: float | None = None,
    ) -> CSTRStep:
        caf = self.last_disturbance[0] if caf is None else caf
        tf = self.last_disturbance[1] if tf is None else tf
        terminated = bool(
            not np.isfinite(self.state).all()
            or self.state[1] >= self.config.temperature_limit
            or self.state[1] <= 250.0
            or self.state[0] < 0.0
        )
        metrics = {
            "temperature_error": float(self.state[1] - 330.0),
            "conversion": float(1.0 - self.state[0] / max(caf, 1e-9)),
            "control_effort": float((action - 300.0) ** 2),
        }
        return CSTRStep(
            state=self.state.copy(),
            action=float(action),
            feed_concentration=float(caf),
            feed_temperature=float(tf),
            time=float(self.time),
            terminated=terminated,
            metrics=metrics,
        )

"""Run the self-contained three-state nonlinear CSTR."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from cstr_lab import CSTRSimulator, PIDController


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="artifacts/cstr_pid_rollout.csv")
    args = parser.parse_args()

    simulator = CSTRSimulator(dt=0.1)
    controller = PIDController()
    result = simulator.reset(seed=args.seed)
    controller.reset()

    rows = []
    for _ in range(args.steps):
        temperature = float(simulator.observe(noise_std=0.2)[0])
        action = controller.act(temperature, simulator.dt)
        result = simulator.step(action)
        rows.append({
            "time": result.time,
            "C_A": result.state[0],
            "T_R": result.state[1],
            "T_J": result.state[2],
            "observation_T_R": temperature,
            "action_T_c": result.action,
            "feed_C_A": result.feed_concentration,
            "feed_T": result.feed_temperature,
            "conversion": result.metrics["conversion"],
            "terminated": result.terminated,
        })
        if result.terminated:
            break

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    temperatures = np.array([row["T_R"] for row in rows])
    conversions = np.array([row["conversion"] for row in rows])
    print(f"saved {len(rows)} steps to {output}")
    print(f"temperature range: {temperatures.min():.2f}–{temperatures.max():.2f} K")
    print(f"mean conversion: {conversions.mean():.3f}")
    print(f"terminated: {rows[-1]['terminated']}")


if __name__ == "__main__":
    main()

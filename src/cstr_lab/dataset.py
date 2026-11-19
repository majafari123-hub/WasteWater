"""Trajectory recording for ground-truth and learner datasets."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import numpy as np

from .model import CSTRConfig


def save_cstr_dataset(
    output_dir: str | Path,
    *,
    ground_truth: dict[str, np.ndarray],
    learner: dict[str, np.ndarray],
    config: CSTRConfig,
    dt: float,
    observation_description: str,
    seeds: list[int],
) -> None:
    """Save aligned privileged and controller-visible datasets."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "ground_truth.npz", **ground_truth)
    np.savez_compressed(output / "learner.npz", **learner)

    metadata: dict[str, Any] = {
        "format": "CSTR transition dataset v1",
        "simulator": "WasteWater self-contained nonlinear three-state CSTR",
        "state_names": ["C_A", "T_R", "T_J"],
        "state_units": ["concentration", "K", "K"],
        "action_names": ["coolant_supply_temperature"],
        "action_units": ["K"],
        "disturbance_names": ["feed_C_A", "feed_T"],
        "observation_names": ["T_R_noisy"],
        "observation_description": observation_description,
        "dt": dt,
        "config": asdict(config),
        "episode_seeds": seeds,
        "ground_truth_file": "ground_truth.npz",
        "learner_file": "learner.npz",
        "learner_excludes": [
            "state_t",
            "state_next",
            "disturbances",
            "hidden process variables",
        ],
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def load_dataset(path: str | Path) -> dict[str, np.ndarray]:
    """Load an NPZ dataset into ordinary NumPy arrays."""
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}

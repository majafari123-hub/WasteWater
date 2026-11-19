"""Generate aligned ground-truth and learner CSTR datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from cstr_lab import CSTRSimulator
from cstr_lab.dataset import save_cstr_dataset
from cstr_lab.full_state_pi import FullStatePIController


def generate_episode(
    episode_id: int,
    seed: int,
    steps: int,
    observation_noise_std: float,
) -> tuple[dict[str, list], dict[str, list]]:
    simulator = CSTRSimulator(dt=0.1)
    controller = FullStatePIController()

    rng = np.random.default_rng(seed)
    initial_state = np.array([
        rng.uniform(0.70, 0.90),
        rng.uniform(326.0, 334.0),
        rng.uniform(295.0, 305.0),
    ])
    simulator.reset(seed=seed, state=initial_state)
    controller.reset()

    gt = {key: [] for key in (
        "episode_id", "step", "time", "state_t", "state_next",
        "observation_t", "observation_next", "action",
        "disturbances", "temperature_error", "conversion",
        "control_effort", "reward", "terminated",
    )}
    learner = {key: [] for key in (
        "episode_id", "step", "time", "observation_t",
        "observation_next", "observation_mask_t", "observation_mask_next",
        "action", "reward", "terminated",
    )}

    for step in range(steps):
        state_t = simulator.state.copy()
        observation_t = simulator.observe(noise_std=observation_noise_std)
        action = controller.act(state_t, simulator.dt)
        transition = simulator.step(action)
        observation_next = simulator.observe(noise_std=observation_noise_std)

        reward = -(
            transition.metrics["temperature_error"] ** 2
            + 0.001 * transition.metrics["control_effort"]
        )

        gt["episode_id"].append(episode_id)
        gt["step"].append(step)
        gt["time"].append(transition.time)
        gt["state_t"].append(state_t)
        gt["state_next"].append(transition.state.copy())
        gt["observation_t"].append(observation_t)
        gt["observation_next"].append(observation_next)
        gt["action"].append(transition.action)
        gt["disturbances"].append([
            transition.feed_concentration,
            transition.feed_temperature,
        ])
        gt["temperature_error"].append(transition.metrics["temperature_error"])
        gt["conversion"].append(transition.metrics["conversion"])
        gt["control_effort"].append(transition.metrics["control_effort"])
        gt["reward"].append(reward)
        gt["terminated"].append(transition.terminated)

        learner["episode_id"].append(episode_id)
        learner["step"].append(step)
        learner["time"].append(transition.time)
        learner["observation_t"].append(observation_t)
        learner["observation_next"].append(observation_next)
        learner["observation_mask_t"].append([1.0])
        learner["observation_mask_next"].append([1.0])
        learner["action"].append(transition.action)
        learner["reward"].append(reward)
        learner["terminated"].append(transition.terminated)

        if transition.terminated:
            break

    return gt, learner


def stack_records(records: dict[str, list]) -> dict[str, np.ndarray]:
    return {key: np.asarray(value) for key, value in records.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--noise-std", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--output", default="artifacts/cstr_dataset")
    args = parser.parse_args()

    gt_records = None
    learner_records = None
    seeds = []

    for episode_id in range(args.episodes):
        episode_seed = args.seed + episode_id
        seeds.append(episode_seed)
        gt, learner = generate_episode(
            episode_id,
            episode_seed,
            args.steps,
            args.noise_std,
        )
        if gt_records is None:
            gt_records = {key: [] for key in gt}
            learner_records = {key: [] for key in learner}
        for key in gt_records:
            gt_records[key].extend(gt[key])
        for key in learner_records:
            learner_records[key].extend(learner[key])

    simulator = CSTRSimulator(dt=0.1)
    save_cstr_dataset(
        args.output,
        ground_truth=stack_records(gt_records),
        learner=stack_records(learner_records),
        config=simulator.config,
        dt=simulator.dt,
        observation_description=(
            "Noisy reactor temperature only; concentration and jacket "
            "temperature are hidden from the learner."
        ),
        seeds=seeds,
    )

    count = len(gt_records["step"])
    print(f"saved {count} transitions from {args.episodes} episodes to {Path(args.output)}")
    print(f"ground truth: {Path(args.output) / 'ground_truth.npz'}")
    print(f"learner data: {Path(args.output) / 'learner.npz'}")
    print(f"metadata: {Path(args.output) / 'metadata.json'}")


if __name__ == "__main__":
    main()

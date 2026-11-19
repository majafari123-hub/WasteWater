# CSTR datasets

Generate aligned datasets with:

    git pull
    python -m pip install -e .
    PYTHONPATH=src python scripts/generate_cstr_dataset.py --episodes 100 --steps 500

This writes:

    artifacts/cstr_dataset/ground_truth.npz
    artifacts/cstr_dataset/learner.npz
    artifacts/cstr_dataset/metadata.json

## Ground-truth dataset

The privileged file contains:

- state_t and state_next: all three physical states
- observation_t and observation_next
- action
- disturbances
- temperature_error
- conversion
- control_effort
- reward
- episode and termination metadata

Use this file for simulator diagnostics, plotting, and controller evaluation.

## Learner dataset

The learner file contains only:

- current and next observations
- actions
- rewards
- observation masks
- episode and time metadata

It intentionally excludes the hidden state and disturbance arrays. This is
the file to use when training a controller-visible predictive or knowledge
model.

## Example loading

    import numpy as np

    ground_truth = np.load(
        "artifacts/cstr_dataset/ground_truth.npz",
        allow_pickle=False,
    )
    learner = np.load(
        "artifacts/cstr_dataset/learner.npz",
        allow_pickle=False,
    )

    print(ground_truth["state_t"].shape)
    print(learner["observation_t"].shape)

For a diffusion model, a useful training sample is a history window:

    [o_(t-H+1), a_(t-H+1), ..., o_t, a_t]

with target future observations or states. For Active Inference, the learner
dataset supplies action-conditioned observation transitions, while the
ground-truth file remains reserved for evaluation and diagnostics.

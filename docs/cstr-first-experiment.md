# First experiment: a three-state nonlinear CSTR

This is a deliberately small process-control problem.

## Hidden process

The physical state is:

    x = [C_A, T_R, T_J]

- C_A: reactant concentration
- T_R: reactor temperature
- T_J: cooling-jacket temperature

The action is coolant supply temperature. Feed concentration and feed
temperature are stochastic disturbances held over each integration step.

## Controller view

The baseline script exposes only a noisy reactor-temperature measurement to a PID
controller. The complete state, disturbances, action, and metrics are saved to CSV
for analysis.

## Run

    python -m pip install -e .
    PYTHONPATH=src python scripts/run_cstr.py

The output is artifacts/cstr_pid_rollout.csv.

This is a process simulator, not a validated plant model. It is intended for
learning and control experiments. The next steps are to inspect trajectories,
add plots, and compare the temperature-only PID with full-state and open-loop
runs.

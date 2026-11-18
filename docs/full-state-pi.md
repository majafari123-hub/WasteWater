# Full-state PI experiment

This experiment gives the controller the complete physical state:

    [C_A, T_R, T_J]

The controller uses integral action on reactor-temperature error, plus
instantaneous feedback from reactant concentration and jacket temperature.

Run:

    git pull
    python -m pip install -e .
    PYTHONPATH=src python scripts/run_cstr_full_state_pi.py

Output:

    artifacts/cstr_full_state_pi_rollout.csv

This is a privileged-controller baseline. It is useful for measuring the
benefit of state information before introducing a state estimator or a
partial-observation controller.

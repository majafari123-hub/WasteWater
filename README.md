# BSM1 Diffusion + Active Inference Lab

This project is an experiment harness for the Benchmark Simulation Model No. 1 (BSM1) wastewater-treatment process.

The design keeps four things separate:

- simulator ground truth: the complete hidden process state;
- controller observations: a selected, noisy, possibly masked subset of state;
- expert actions: a conventional baseline controller;
- learned-controller actions: diffusion-model and Active-Inference experiments.

The BSM1 simulator backend is intentionally isolated behind `SimulatorBackend`. This lets us test the experiment interface without leaking privileged state into the controller. The verified EXPOsan/QSDsan BSM1 backend will be connected once its source is available in the workspace.

## Layout

```text
src/bsm1_lab/
  backend.py        simulator interface and transition types
  observation.py    partial-observation and sensor-noise model
  expert.py         baseline-controller interface
  env.py            controller-facing environment
  mock_backend.py   tiny deterministic backend for interface tests only
```

## Quick start

```bash
python -m pip install -e .
python scripts/smoke_test.py
```

The smoke test uses the clearly labelled mock backend; it is not a BSM1 process model. The next integration step is to add the actual BSM1 backend and its validation tests against the IWA reference outputs.

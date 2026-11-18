# Actual BSM1 simulator

The process simulator is the maintained EXPOsan implementation of the
International Water Association Benchmark Simulation Model No. 1. It uses ASM1
biochemical kinetics, five activated-sludge reactors, and a 10-layer secondary
clarifier.

## Install

From the repository root:

```bash
python -m venv .venv
. .venv/bin/activate          # Windows: .venv\\Scripts\\activate
python -m pip install -r requirements-bsm1.txt
```

The EXPOsan source is installed from its public repository because the BSM1
module is distributed there rather than as a standalone PyPI package.

## Run

```bash
python scripts/run_bsm1.py --days 100
```

The script writes the final privileged process state and effluent metrics to
`artifacts/bsm1_final_state.json`.

## Important distinction

This runner is for inspecting the complete physical/process simulation. It is
not yet the partial-observation environment. Later, the controller wrapper will
select observations from this ground truth and keep the full state available
only for evaluation.

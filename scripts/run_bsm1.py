"""Run the actual EXPOsan BSM1 process simulator.

This is intentionally separate from the controller-facing environment. It
simulates the complete process with privileged state available for inspection.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=float, default=100.0)
    parser.add_argument("--method", default="BDF")
    parser.add_argument("--output", default="artifacts/bsm1_final_state.json")
    args = parser.parse_args()

    from exposan import bsm1

    bsm1.load(reload=True, suspended_growth_model="ASM1", reactor_model="CSTR")
    system = bsm1.sys
    system.simulate(t_span=(0.0, args.days), method=args.method)

    units = system.flowsheet.unit
    state = {}
    for name in ("A1", "A2", "O1", "O2", "O3"):
        unit = getattr(units, name)
        state[name] = np.asarray(unit._state[:13], dtype=float).tolist()
    clarifier = getattr(units, "C1")
    state["C1_TSS_layers"] = np.asarray(clarifier._state[-10:], dtype=float).tolist()

    effluent = system.outs[0]
    metrics = {}
    for name in ("COD", "BOD", "TN", "TSS"):
        value = getattr(effluent, name, None)
        if value is not None:
            metrics[name] = float(value)

    result = {
        "simulator": "EXPOsan BSM1",
        "horizon_days": args.days,
        "solver": args.method,
        "reactor_state_order": list(bsm1.components.IDs[:13]),
        "final_state": state,
        "effluent_metrics": metrics,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(output), "effluent_metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()

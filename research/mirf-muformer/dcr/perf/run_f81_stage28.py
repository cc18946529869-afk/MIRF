"""One truncated-Newton run from the Stage 2.7 previous-10000 terminal state.

The historical best-projection vector is checked and recorded, then left unused.
L-BFGS curvature pairs are not loaded. Ordinary objective calls stop at 3000.
The independent terminal audit is counted separately. The clock stops at 45 minutes.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .audit_f81_stage28_residuals import edge_table
from .check_f81_joint import data
from .f81_stage28_trust import incident_edges, solve_subspace_trust
from .f81_zero_boundary import ZeroBoundary


STAGE27_BEST_PROJECTION = 0.011006638603404433
STAGE27_TERMINAL_PROJECTION = 0.013947183793858144
STAGE27_TERMINAL_OBJECTIVE = 18409.51825984109
TERMINAL_LENGTHS_SHA = "7e237869f1a77b3049e0765e0121cbeceea5c3960b434b9c4fdde8eaa58d7222"
BEST_LENGTHS_SHA = "1ddf1359281da922f1ef8175d8baf27b5dda387d8ddf22ec0b9dafe8d7c48cac"


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lengths_sha(path):
    lengths = np.ascontiguousarray(np.load(path)["lengths"], dtype=np.float64)
    return lengths, hashlib.sha256(lengths.tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-npz", type=Path, required=True)
    parser.add_argument("--backup-npz", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--case", default="previous-10000")
    parser.add_argument("--expect-start-sha", default=TERMINAL_LENGTHS_SHA)
    for flag in ("prep", "gate", "library", "checkpoint", "stage26", "boundary"):
        parser.add_argument("--" + flag, type=Path, required=True)
    args = parser.parse_args()
    start, start_sha = lengths_sha(args.start_npz)
    _backup, backup_sha = lengths_sha(args.backup_npz)
    if start_sha != args.expect_start_sha or backup_sha != BEST_LENGTHS_SHA:
        raise SystemExit("start or backup checkpoint does not match the Stage 2.7 recovery")
    engine, auditor, _point, facts, digest = data(args)
    guard = ZeroBoundary(engine, args.boundary)
    adjacency = incident_edges(edge_table(engine))
    args.trace.parent.mkdir(parents=True, exist_ok=True)
    with args.trace.open("x", encoding="utf-8") as handle:
        def emit(row):
            handle.write(json.dumps(row, allow_nan=False) + "\n")
            handle.flush()

        _terminal, result = solve_subspace_trust(
            engine, auditor, guard, start, emit, adjacency,
            archive=args.archive, maxiter=400, maxeval=3000, threads=4,
            wall_seconds=45 * 60, max_cg=6)
    best = result["checkpoints"].get("best_projection", {})
    final = result["checkpoints"].get("final", {})
    ceiling = STAGE27_TERMINAL_OBJECTIVE + 1e-6
    gate_level = STAGE27_BEST_PROJECTION / 10.
    candidates = {
        "best_projection": best,
        "final": final,
    }
    passed = {}
    for name, meta in candidates.items():
        if not meta:
            passed[name] = False
            continue
        passed[name] = bool(
            meta["projection"] <= gate_level
            and meta["objective"] <= ceiling
            and result["feasible"]
            and result["cross_path_max"] <= 1e-8)
    report = {
        **facts,
        "case": args.case,
        "method": result["method"],
        "lbfgs_history": "discarded; no Stage 2.7 secant pair was loaded",
        "start_role": "stage27_final" if start_sha == TERMINAL_LENGTHS_SHA else "caller_checkpoint",
        "start_lengths_sha256": start_sha,
        "backup_role": "stage27_best_projection_not_used",
        "backup_lengths_sha256": backup_sha,
        "stage26_start_sha256": digest,
        "stage27_best_projection": STAGE27_BEST_PROJECTION,
        "stage27_terminal_projection": STAGE27_TERMINAL_PROJECTION,
        "stage27_terminal_objective": STAGE27_TERMINAL_OBJECTIVE,
        "stage_gate_projection": gate_level,
        "stage_gate_passed": any(passed.values()),
        "stage_gate_by_checkpoint": passed,
        "formal_convergence": result["both_gates_passed"],
        "solver": file_sha(Path(__file__).with_name("f81_stage28_trust.py")),
        "result": result,
    }
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

"""Replay only the Stage 2.7 previous-10000 run and keep full checkpoints.

The optimizer arithmetic is unchanged. The archive records the initial state,
the lowest ordinary projection, the lowest objective, and the stopping state.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .check_f81_joint import data
from .f81_joint_optimize import solve
from .f81_zero_boundary import ZeroBoundary


def projections(path):
    values = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["event"] == "iterate":
            values.append(row["projection"])
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("previous-10000",), default="previous-10000")
    parser.add_argument("--stage27", type=Path, required=True)
    parser.add_argument("--archived-trace", type=Path, required=True)
    for flag in ("prep", "gate", "library", "checkpoint", "stage26", "boundary", "dual",
                 "reference", "trace", "archive"):
        parser.add_argument("--" + flag, type=Path, required=True)
    args = parser.parse_args()
    archived = json.loads(args.stage27.read_text(encoding="utf-8"))["cases"][args.case]
    engine, auditor, point, facts, digest = data(args)
    guard = ZeroBoundary(engine, args.boundary)
    args.trace.parent.mkdir(parents=True, exist_ok=True)
    with args.trace.open("x", encoding="utf-8") as handle:
        def emit(row):
            handle.write(json.dumps(row, allow_nan=False) + "\n")
            handle.flush()

        terminal, result = solve(engine, auditor, guard, point, emit, archive=args.archive)
    terminal_sha = hashlib.sha256(np.ascontiguousarray(terminal).tobytes()).hexdigest()
    replayed = projections(args.trace)
    original = projections(args.archived_trace)
    report = {
        **facts,
        "case": args.case,
        "start_sha256": digest,
        "terminal_sha256": terminal_sha,
        "archived_terminal_sha256": archived["terminal_sha256"],
        "terminal_matches_stage27": terminal_sha == archived["terminal_sha256"],
        "projection_trace_matches": replayed == original,
        "replayed_iterations": len(replayed),
        "archived_iterations": len(original),
        "best_ordinary_projection": result["best_ordinary_projection"],
        "archived_best_ordinary_projection": archived["best_ordinary_projection"],
        "final_projection": result["projection"],
        "archived_final_projection": archived["projection"],
        "final_objective": result["objective"],
        "archived_final_objective": archived["objective"],
        "checkpoints": result["checkpoints"],
        "counts": result["counts"],
        "status": result["status"],
        "feasible": result["feasible"],
    }
    print(json.dumps(report, indent=2, allow_nan=False))
    if not report["terminal_matches_stage27"] or not report["projection_trace_matches"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

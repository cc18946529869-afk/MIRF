"""Recompute two saved Stage 2.8 states with both gradient paths.

The calls are a check of stored checkpoints. They are not an optimization run.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .check_f81_joint import data
from .f81_optimize import projected_norm
from .f81_zero_boundary import ZeroBoundary


def lengths_of(path):
    saved = np.load(path)
    lengths = np.ascontiguousarray(saved["lengths"], dtype=np.float64)
    return saved, lengths, hashlib.sha256(lengths.tobytes()).hexdigest()


def recompute(engine, auditor, guard, path):
    saved, lengths, digest = lengths_of(path)
    item = engine.evaluate(lengths, threads=4, gradient=True, diagonal=False)
    sites = len(item.site_log_likelihood)
    gradient = -item.length_gradient / sites
    audit = auditor.evaluate(lengths, threads=4, log_domain=True)
    audit_gradient = -audit.length_gradient / sites
    gap = np.abs(audit_gradient - gradient)
    objective = -item.log_likelihood / sites
    projection = projected_norm(lengths, gradient)
    return {
        "path": str(path),
        "lengths_sha256": digest,
        "stored_objective": float(saved["objective"]),
        "stored_projection": float(saved["projection"]),
        "objective": objective,
        "independent_objective": -audit.log_likelihood / sites,
        "projection": projection,
        "independent_projection": projected_norm(lengths, audit_gradient),
        "objective_abs_difference": abs(objective - float(saved["objective"])),
        "projection_abs_difference": abs(projection - float(saved["projection"])),
        "gradient_finite": bool(np.isfinite(gradient).all()),
        "cross_path_max": float(gap.max()),
        "cross_path_within_1e-8": bool(gap.max() <= 1e-8),
        "feasible": bool(guard.feasible(lengths)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--case", default="previous-10000")
    for flag in ("prep", "gate", "library", "checkpoint", "stage26", "boundary"):
        parser.add_argument("--" + flag, type=Path, required=True)
    args = parser.parse_args()
    engine, auditor, _point, facts, _digest = data(args)
    guard = ZeroBoundary(engine, args.boundary)
    official = recompute(engine, auditor, guard, args.official)
    diagnostic = recompute(engine, auditor, guard, args.diagnostic)
    diagnostic_better = bool(
        diagnostic["feasible"] and diagnostic["gradient_finite"] and diagnostic["cross_path_within_1e-8"]
        and diagnostic["projection"] < official["projection"]
        and diagnostic["objective"] < official["objective"])
    report = {
        **{key: facts[key] for key in (
            "indices_sha256", "reference_profile_sha256", "topology_sha256", "library_sha256")},
        "official": official,
        "diagnostic": diagnostic,
        "diagnostic_is_better": diagnostic_better,
        "continuation_checkpoint": "diagnostic" if diagnostic_better else "official",
    }
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "diagnostic_is_better": diagnostic_better,
        "official_projection": official["projection"],
        "diagnostic_projection": diagnostic["projection"],
        "official_feasible": official["feasible"],
        "diagnostic_feasible": diagnostic["feasible"],
        "official_cross": official["cross_path_max"],
        "diagnostic_cross": diagnostic["cross_path_max"],
    }, indent=2))
    if not (official["feasible"] and diagnostic["feasible"] and official["cross_path_within_1e-8"]
            and diagnostic["cross_path_within_1e-8"] and official["gradient_finite"]
            and diagnostic["gradient_finite"]):
        raise SystemExit(2)


if __name__ == "__main__":
    main()

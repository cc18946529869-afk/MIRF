"""Independent review of the single Stage 2.8 previous-10000 run.

Recomputes the saved best-projection state and the final state with both
gradient paths. Those calls are not part of the 3000-evaluation budget.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .check_f81_joint import data
from .f81_optimize import ARMIJO_C1, projected_norm
from .f81_zero_boundary import ZeroBoundary


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recompute(engine, auditor, guard, path):
    saved = np.load(path)
    lengths = np.ascontiguousarray(saved["lengths"], dtype=np.float64)
    item = engine.evaluate(lengths, threads=4, gradient=True, diagonal=True)
    sites = len(item.site_log_likelihood)
    gradient = -item.length_gradient / sites
    diagonal = item.negative_log_hessian_diagonal / sites
    audit = auditor.evaluate(lengths, threads=4, log_domain=True)
    audit_gradient = -audit.length_gradient / sites
    gap = np.abs(audit_gradient - gradient)
    return {
        "lengths_sha256": hashlib.sha256(lengths.tobytes()).hexdigest(),
        "objective": -item.log_likelihood / sites,
        "projection": projected_norm(lengths, gradient),
        "independent_projection": projected_norm(lengths, audit_gradient),
        "saved_gradient_max_abs": float(np.max(np.abs(gradient - saved["physical_gradient"]))),
        "saved_diagonal_max_abs": float(np.max(np.abs(diagonal - saved["hessian_diagonal"]))),
        "cross_path_max": float(gap.max()),
        "feasible": bool(guard.feasible(lengths)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-json", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--replay-json", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--case", default="previous-10000")
    for flag in ("prep", "gate", "library", "checkpoint", "stage26", "boundary"):
        parser.add_argument("--" + flag, type=Path, required=True)
    args = parser.parse_args()
    run = json.loads(args.run_json.read_text(encoding="utf-8"))
    replay = json.loads(args.replay_json.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in args.trace.read_text(encoding="utf-8").splitlines() if line]
    fingerprint_keys = ("indices_sha256", "reference_profile_sha256", "topology_sha256", "library_sha256")
    fingerprints_match = all(run[key] == replay[key] for key in fingerprint_keys)
    accepted = [row for row in rows if row.get("event") == "trial" and row.get("status") == "accepted"]
    armijo_holds = all(
        row["slope"] < 0 and row["objective_change"] <= ARMIJO_C1 * row["slope"] + 1e-12
        for row in accepted)
    projection = None
    projection_monotone = True
    for row in rows:
        if row.get("event") == "initial":
            projection = row["projection"]
        elif row.get("status") == "accepted":
            if projection is not None and row["projection"] > projection + 1e-12:
                projection_monotone = False
            projection = row["projection"]
    checkpoints = {}
    hashes_match = True
    for name in ("initial", "best_projection", "best_objective", "final"):
        meta = json.loads((args.archive / f"{name}.json").read_text(encoding="utf-8"))
        digest = sha256(args.archive / f"{name}.npz")
        lengths = np.ascontiguousarray(np.load(args.archive / f"{name}.npz")["lengths"], dtype=np.float64)
        length_digest = hashlib.sha256(lengths.tobytes()).hexdigest()
        ok = digest == meta["sha256"] and length_digest == meta["lengths_sha256"]
        hashes_match = hashes_match and ok
        checkpoints[name] = {**meta, "recomputed_sha256": digest, "hash_ok": ok}
    counts = run["result"]["counts"]
    ledger_ok = counts["objective"] <= 3000 and counts["independent_objective"] >= 1
    engine, auditor, _point, _facts, _digest = data(args)
    guard = ZeroBoundary(engine, args.boundary)
    best = recompute(engine, auditor, guard, args.archive / "best_projection.npz")
    final = recompute(engine, auditor, guard, args.archive / "final.npz")
    verification = run["result"]["hvp_verification"]
    report = {
        "fingerprints_match": fingerprints_match,
        "objective_budget_ok": ledger_ok,
        "objective_calls": counts["objective"],
        "gradient_calls": counts["gradient"],
        "diagonal_calls": counts["diagonal"],
        "hessian_products": counts["hessian_products"],
        "independent_calls": counts["independent_objective"],
        "accepted_steps": len(accepted),
        "armijo_holds": armijo_holds,
        "projection_did_not_increase_on_accepted_steps": projection_monotone,
        "checkpoint_hashes_match": hashes_match,
        "checkpoints": checkpoints,
        "best_projection_recompute": best,
        "final_recompute": final,
        "hvp_max_relative_error": verification["max_relative_error"],
        "hvp_accepted": verification["accepted"],
        "terminal_independent_projection": final["independent_projection"],
        "cross_path_max": final["cross_path_max"],
        "feasible": final["feasible"],
        "stop_reason": run["result"]["status"],
        "stage_gate_passed": run["stage_gate_passed"],
        "formal_convergence": run["formal_convergence"],
        "lbfgs_history": run["lbfgs_history"],
    }
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "fingerprints_match", "objective_budget_ok", "armijo_holds", "checkpoint_hashes_match",
        "hvp_accepted", "terminal_independent_projection", "cross_path_max", "feasible",
        "stop_reason", "stage_gate_passed", "formal_convergence")}, indent=2))
    required = (
        fingerprints_match, ledger_ok, armijo_holds, hashes_match,
        verification["accepted"], best["saved_gradient_max_abs"] <= 1e-8,
        final["cross_path_max"] <= 1e-8, final["feasible"])
    if not all(required):
        raise SystemExit(2)


if __name__ == "__main__":
    main()

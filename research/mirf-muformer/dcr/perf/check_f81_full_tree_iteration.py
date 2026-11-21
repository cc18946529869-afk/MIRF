"""Continue the frozen mixed-coordinate update on three fixed F81 checkpoints."""
from __future__ import annotations

import argparse
from decimal import Decimal, localcontext
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from .check_f81_edge_conditioning import (
    TangentReference,
    local_attempt,
    precision_audit,
    public,
    ref_terms,
    scan_vectors,
)
from .check_f81_gradient_attribution import recover_warm_terminal
from .check_f81_observed_bridge import load_case
from .f81_zero_boundary import ZeroBoundary

PROJECTION_GATE = 1e-5
PLATEAU_UPDATES = 8
PLATEAU_FRACTION = 0.01


def projected_residuals(point, gradient):
    return point - np.maximum(0., point-gradient)


def recurrent_period(edges):
    for n in range(1, min(8, len(edges)//3)+1):
        if edges[-n:] == edges[-2*n:-n] == edges[-3*n:-2*n]:
            return edges[-n:]
    return None


def plateau(history):
    return (len(history) > PLATEAU_UPDATES
            and history[-1] >= (1.-PLATEAU_FRACTION)*history[-1-PLATEAU_UPDATES])


def both_gates(scan):
    return (max(scan["projected_gradient_inf"], scan["rerooted_projected_gradient_inf"]) <= PROJECTION_GATE
            and scan["cross_path_gradient_passed"])


def residual_summary(point, gradient):
    residual = np.abs(projected_residuals(point, gradient))
    order = np.argsort(-residual, kind="stable")[:12]
    return {"counts_above": {str(t): int(np.count_nonzero(residual > t))
                             for t in (1e-5, 1e-2, 1., 1e2, 1e4, 1e6)},
            "zero_edges": int(np.count_nonzero(point == 0)),
            "active_zero_edges": int(np.count_nonzero((point == 0) & (gradient >= 0))),
            "top": [{"edge": int(i), "length": float(point[i]), "gradient": float(gradient[i]),
                     "abs_projected_residual": float(residual[i])} for i in order]}


def transfer_summary(before_point, trial, chosen, before, after):
    residual_before = np.abs(projected_residuals(before_point, before["vectors"][1]))
    residual_after = np.abs(projected_residuals(trial, after["vectors"][1]))
    change = residual_after-residual_before
    largest_increases = np.argsort(-change, kind="stable")[:5]
    return {"chosen_abs_before": float(residual_before[chosen]),
            "chosen_abs_after": float(residual_after[chosen]),
            "other_edges_increased": int(np.count_nonzero(change > 0) - (change[chosen] > 0)),
            "largest_increases": [{"edge": int(i), "before": float(residual_before[i]),
                                   "after": float(residual_after[i]), "increase": float(change[i])}
                                  for i in largest_increases if change[i] > 0 and i != chosen]}


def save_checkpoint(path, point):
    filename = path.with_suffix(".writing")
    with filename.open("wb") as output:
        np.save(output, point, allow_pickle=False)
        output.flush()
        os.fsync(output.fileno())
    filename.replace(path)


def fixed_input(args):
    engine, auditor, facts = load_case(args.prep, args.gate, 10000, args.library)
    point = np.load(args.checkpoint, allow_pickle=False)
    manifest = json.loads(args.manifest.read_text())
    for key in ("alignment_weights_sha256", "indices_sha256", "reference_profile_sha256",
                "topology_sha256", "library_sha256"):
        assert facts[key] == manifest[key]
    replay = 0
    if args.warm_trace:
        assert hashlib.sha256(point.tobytes()).hexdigest() == manifest["initial_sha256"]
        point, replay = recover_warm_terminal(engine, point, args.warm_trace, manifest)
    digest = hashlib.sha256(point.tobytes()).hexdigest()
    previous = json.loads(args.stage25.read_text())["cases"][args.case]
    assert digest == previous["initial_sha256"] == manifest["terminal_sha256" if replay else "checkpoint_sha256"]
    return engine, auditor, point, facts, previous, digest, replay


def audit_disagreement(engine, point, current, args, dual):
    edge = current["max_gradient_disagreement"]["edge"]
    split = {bits: info[0] for bits, info in ref_terms(engine, point, args.reference, edge).items()}
    result = precision_audit(point, edge, split, dual, current)
    reference = split[256]
    for level in result["higher_precision_paths"].values():
        a, b = Decimal(level["split"]), Decimal(level["tangent"])
        level["split_abs_to_256bit_reference"] = str(abs(a-reference))
        level["tangent_abs_to_256bit_reference"] = str(abs(b-reference))
        level["both_pass_256bit_reference"] = max(abs(a-reference), abs(b-reference)) <= Decimal("1e-8")
    return {"edge": edge, "length": float(point[edge]), **result}


def run(args, emit):
    started = perf_counter()
    engine, auditor, point, facts, previous, digest, replay = fixed_input(args)
    guard = ZeroBoundary(engine, args.boundary)
    dual = TangentReference(auditor, args.dual)
    current = scan_vectors(engine, auditor, point)
    initial = public(current)
    assert all(initial[k] == v for k, v in previous["before"].items())
    save_checkpoint(args.out_checkpoint, point)
    retained = []
    history = [current["rerooted_projected_gradient_inf"]]
    totals = {"replay": replay, "full_tree": 2, "scalar_conditioned": 0,
              "high_precision_split": 0, "high_precision_tangent": 0,
              "high_precision_scalar_terminal": 0}
    stage25_audit = None
    repeated = None
    stop_diagnosis = None
    trials = 0
    emit({"event": "initial", "case": args.case, "scan": initial, "replay_evaluations": replay,
          "timing_seconds": perf_counter()-started})
    loop_started = perf_counter()
    while True:
        if both_gates(current):
            status = "both_frozen_gates"
            break
        if perf_counter()-loop_started >= args.max_wall_seconds:
            status = "diagnostic_wall_budget"
            break

        gradient = current["vectors"][1]
        residual = projected_residuals(point, gradient)
        edge = int(np.abs(residual).argmax())
        t0 = perf_counter()
        proposal, trial, candidate = local_attempt(engine, auditor, guard, point, current, edge,
                                                  args.reference, dual, compare=False)
        local = proposal["local_strategies"][0]
        totals["scalar_conditioned"] += local["evaluations"]
        totals["full_tree"] += proposal["candidate_full_gradient_calls"]
        totals["high_precision_split"] += proposal["split_calls"]
        totals["high_precision_tangent"] += proposal["tangent_calls"]
        totals["high_precision_scalar_terminal"] += proposal["candidate_high_precision_response_calls"]
        trials += 1
        f_pass = (candidate["total_log_likelihood"] >= current["total_log_likelihood"]
                  and candidate["rerooted_log_likelihood"] >= current["rerooted_log_likelihood"])
        keep = proposal["whole_tree_update_retained"] and f_pass
        emit({"event": "proposal", "iteration": trials, "retained": keep,
              "full_likelihood_non_decreasing": f_pass,
              "elapsed_seconds": perf_counter()-t0, "cumulative_elapsed_seconds": perf_counter()-loop_started,
              "zero_edge_check_passed": True, **proposal})
        if not keep:
            stop_diagnosis = {"candidate": residual_summary(trial, candidate["vectors"][1]),
                              "retained": residual_summary(point, gradient),
                              "cross_edge_residual_transfer": transfer_summary(point, trial, edge, current, candidate)}
            if trial[edge] == point[edge]:
                status = "no_local_move"
            elif f_pass and (candidate["projected_gradient_inf"] >= current["projected_gradient_inf"]
                             or candidate["rerooted_projected_gradient_inf"] >= current["rerooted_projected_gradient_inf"]):
                status = "full_tree_projection_did_not_decrease"
            elif not f_pass:
                status = "full_tree_objective_increase"
            else:
                status = "proposal_failed_validation"
            break
        retained.append(edge)
        point, current = trial, candidate
        save_checkpoint(args.out_checkpoint, point)
        history.append(current["rerooted_projected_gradient_inf"])
        if len(retained) == 3:
            assert all(current[k] == v for k, v in previous["after"].items()), args.case
            if args.case == "short-10000":
                stage25_audit = audit_disagreement(engine, point, current, args, dual)
                totals["high_precision_split"] += 2
                totals["high_precision_tangent"] += 2
                emit({"event": "stage25_discrepancy", "after_updates": 3, "audit": stage25_audit})
        if both_gates(current):
            status = "both_frozen_gates"
            break
        repeated = recurrent_period(retained)
        if repeated:
            status = "repeating_coordinate_cycle"
            break
        if plateau(history):
            status = "projection_plateau"
            break

    if status in ("projection_plateau", "repeating_coordinate_cycle"):
        stop_diagnosis = {"retained": residual_summary(point, current["vectors"][1]),
                          "recent_accepted_edges": retained[-24:],
                          "recent_projection_history": history[-25:]}
    assert guard.feasible(point)
    if stage25_audit is not None and len(retained) == 3:
        terminal_audit = stage25_audit
    else:
        terminal_audit = audit_disagreement(engine, point, current, args, dual)
        totals["high_precision_split"] += 2
        totals["high_precision_tangent"] += 2
    return {**facts, "case": args.case, "initial_sha256": digest,
            "terminal_sha256": hashlib.sha256(point.tobytes()).hexdigest(),
            "status": status, "proposed_updates": trials, "retained_edges": retained,
            "independent_projected_gradient_trace": history, "initial": initial, "terminal": public(current),
            "terminal_discrepancy_audit": terminal_audit, "stage25_short_discrepancy_audit": stage25_audit,
            "terminal_residual_distribution": residual_summary(point, current["vectors"][1]),
            "stage25_three_updates_reproduced": len(retained) >= 3,
            "both_frozen_gates_passed": both_gates(current), "terminal_zero_edge_check_passed": True,
            "stop_diagnosis": stop_diagnosis, "evaluation_counts": totals,
            "diagnostic_wall_seconds": args.max_wall_seconds, "elapsed_seconds": perf_counter()-started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("short-10000", "warm-10000", "previous-10000"), required=True)
    parser.add_argument("--max-wall-seconds", type=float, default=2400.)
    for flag in ("prep", "gate", "library", "boundary", "reference", "dual", "checkpoint",
                 "manifest", "stage25", "trace", "out-checkpoint", "warm-trace"):
        parser.add_argument("--" + flag, type=Path, required=flag != "warm-trace")
    args = parser.parse_args()
    with args.trace.open("x", encoding="utf-8") as trace, localcontext() as ctx:
        ctx.prec = 115

        def emit(x):
            trace.write(json.dumps(x, allow_nan=False) + "\n")
            trace.flush()

        out = run(args, emit)
    print(json.dumps(out, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

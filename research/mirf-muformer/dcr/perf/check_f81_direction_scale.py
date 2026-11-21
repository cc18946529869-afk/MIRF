"""Reconstruct archived accepted F81 states and probe fixed projected rays."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import resource
from time import perf_counter

import numpy as np

from potts_fit import _two_loop

from .check_f81_observed_bridge import load_case, start_lengths
from .f81_optimize import ARMIJO_C1, projected_norm, projected_scale_step
from .f81_zero_boundary import ZeroBoundary


def infnorm(x):
    return float(np.max(np.abs(x)))


def ratio(a, b):
    return float(a / b) if b else None


def tangent(point, direction):
    return np.where((point == 0) & (direction < 0), 0., direction)


def curvature_is_usable(step, change, sy):
    return sy > 1e-10 * np.linalg.norm(step) * np.linalg.norm(change)


def cosine(a, b):
    return ratio(float(a @ b), float(np.linalg.norm(a) * np.linalg.norm(b)))


def reconstruct_direction(point, gradient, history):
    active = (point == 0) & (gradient >= 0)
    free = ~active
    pairs = []
    pair_records = []
    for iteration, step, change, sy_full in history:
        step, change = step[free], change[free]
        sy = float(step @ change) if active.any() else sy_full
        yy = float(change @ change)
        kept = curvature_is_usable(step, change, sy) if active.any() else True
        pair_records.append({"accepted_iteration": iteration, "sTy_full": sy_full,
                             "sTy_restricted": sy, "yTy_restricted": yy, "used": bool(kept)})
        if kept:
            pairs.append((step, change, sy))
    raw = np.zeros_like(point)
    steps, changes, curvatures = (list(values) for values in zip(*pairs)) if pairs else ([], [], [])
    raw[free] = _two_loop(gradient[free], steps, changes, curvatures)
    direction = tangent(point, raw)
    free_gradient = np.where(active, 0., gradient)
    last_pair = next((p for p in reversed(pair_records) if p["used"]), None)
    details = {
        "gamma": ratio(last_pair["sTy_restricted"], last_pair["yTy_restricted"]) if last_pair else None,
        "latest_used_pair": last_pair, "curvature_pairs": pair_records,
        "active_constraints": int(np.count_nonzero(active)), "zero_edges": int(np.count_nonzero(point == 0)),
        "raw_outward_directions_clipped_at_zero": int(np.count_nonzero(raw != direction)),
        "free_gradient_inf": infnorm(free_gradient), "free_gradient_l2": float(np.linalg.norm(free_gradient)),
        "projected_gradient_inf": projected_norm(point, gradient), "raw_direction_inf": infnorm(raw),
        "direction_inf": infnorm(direction), "direction_l2": float(np.linalg.norm(direction)),
        "directional_derivative": float(gradient @ direction),
        "gradient_direction_cosine": cosine(gradient, direction),
        "free_gradient_direction_cosine": cosine(free_gradient, direction),
        "tangent_over_raw_direction_l2": ratio(np.linalg.norm(direction), np.linalg.norm(raw)),
    }
    return direction, free_gradient, details


def displacement_metrics(point, direction, alpha, trial):
    movement = trial - point
    scaled = alpha * direction
    raw_point = point + scaled
    return {
        "displacement_inf": infnorm(movement),
        "displacement_l2": float(np.linalg.norm(movement)),
        "displacement_over_unprojected_inf": ratio(infnorm(movement), infnorm(scaled)),
        "displacement_over_unprojected_l2": ratio(np.linalg.norm(movement), np.linalg.norm(scaled)),
        "projection_clipped_coordinates": int(np.count_nonzero(raw_point < 0)),
        "coordinates_with_movement": int(np.count_nonzero(movement)),
    }


def rebuild(engine, ledger, checkpoint, prior_fit, emit):
    assert prior_fit["search"]["history_resets"] == 0, "This replay requires an unreset archive"
    starts = [event for event in ledger if event["event"] == "line_search"]
    accepts = [event for event in ledger if event.get("outcome") == "accepted"]
    calls = [event for event in ledger if event["event"] == "accepted"]
    evals = {event["call"]: event for event in ledger if event["event"] == "evaluation"}
    point = start_lengths("short", checkpoint.size)
    n_sites = len(engine.pruning.states)
    history = []
    current = engine.evaluate(point, threads=4)
    assert current.log_likelihood == evals[1]["log_likelihood"]
    n = len(accepts)
    assert len(starts) == n + 1 == len(calls) + 1
    for k, search in enumerate(starts):
        gradient = -current.length_gradient / n_sites
        direction, free_gradient, details = reconstruct_direction(point, gradient, history)
        assert search["method"] == "lbfgs" and search["iteration"] == k and search["history"] == len(history)
        for field in ("direction_inf", "direction_l2", "directional_derivative"):
            assert details[field] == search[field], (k, field, details[field], search[field])
        details.update({"event": "direction", "iteration": k, "archived_start_alpha": search["start_step"],
                        "objective": -current.log_likelihood / n_sites})
        if k == n:
            emit(details)
            break
        accepted = accepts[k]
        assert accepted["search"] == search["search"] and calls[k]["iteration"] == k + 1
        alpha = accepted["step"]
        trial = np.maximum(0, point + alpha * direction)
        delta = trial - point
        assert float(gradient @ delta) == accepted["projected_directional_derivative"]
        assert infnorm(delta) == accepted["displacement_inf"]
        latest = engine.evaluate(trial, threads=4)
        expected = evals[calls[k]["calls"]]
        assert latest.log_likelihood == expected["log_likelihood"]
        new_gradient = -latest.length_gradient / n_sites
        assert projected_norm(trial, new_gradient) == expected["projected_gradient_per_site"]
        sy = float(delta @ (new_gradient - gradient))
        used = curvature_is_usable(delta, new_gradient - gradient, sy)
        if used:
            history.append((k + 1, delta, new_gradient - gradient, sy))
            history = history[-10:]
        details.update({"accepted_alpha": alpha, "accepted_trial": displacement_metrics(point, direction, alpha, trial),
                        "accepted_sTy_full": sy, "accepted_curvature_kept": bool(used)})
        emit(details)
        point, current = trial, latest
    fingerprint = hashlib.sha256(point.tobytes()).hexdigest()
    assert fingerprint == prior_fit["checkpoint_sha256"]
    assert point.tobytes() == checkpoint.tobytes(), "Checkpoint must match bit for bit"
    return current, direction, free_gradient, fingerprint


def scan(engine, auditor, checkpoint, current, direction, free_gradient, guard, emit):
    new_value = -current.log_likelihood / len(current.site_log_likelihood)
    gradient = -current.length_gradient / len(current.site_log_likelihood)
    logged = auditor.evaluate(checkpoint, threads=4, log_domain=True, gradient=False)
    rays = (
        ("lbfgs", direction),
        ("reset_projected_negative_gradient", tangent(checkpoint, -free_gradient)),
        ("coordinate_normalized_negative_gradient", tangent(checkpoint, -free_gradient / np.maximum(1, np.abs(free_gradient)))),
    )
    summary = {}
    for label, ray in rays:
        conflict = guard.first_conflict(checkpoint, ray)
        boundary_step = guard.bounded_step(conflict)
        anchor = projected_scale_step(checkpoint, ray, gradient, new_value, boundary_step)
        # Include alpha=1 and larger powers even for directions whose ordinary
        # line searches begin below one. Anchor powers also test those directions
        # on their own feasible scale. No probe updates the checkpoint.
        grid = {"powers_of_two": [math.ldexp(1., k) for k in range(37)],
                "scaled_anchor": [math.ldexp(anchor, k) for k in range(-23, 13)]}
        steps = {}
        for key, points in grid.items():
            for step in points:
                steps.setdefault(step, []).append(key)
        probes = []
        for step, sources in sorted(steps.items()):
            trial = np.maximum(0, checkpoint + step * ray)
            delta = trial - checkpoint
            q = float(gradient @ delta)
            row = {"event": "probe", "direction": label, "alpha": step, "grids": sources,
                   "projected_inner_product": q, "armijo_rhs": new_value + ARMIJO_C1 * q,
                   **displacement_metrics(checkpoint, ray, step, trial)}
            row["feasible"] = bool(np.isfinite(trial).all()) and guard.feasible(trial)
            if row["feasible"] and np.any(delta):
                ordinary = engine.evaluate(trial, threads=4, gradient=False)
                checked = auditor.evaluate(trial, threads=4, log_domain=True, gradient=False)
                row["finite"] = math.isfinite(ordinary.log_likelihood) and math.isfinite(checked.log_likelihood)
                if row["finite"]:
                    value = -ordinary.log_likelihood / len(ordinary.site_log_likelihood)
                    change = -math.fsum(checked.site_log_likelihood - logged.site_log_likelihood) / len(logged.site_log_likelihood)
                    row.update({"objective": value, "armijo_gap": value - row["armijo_rhs"],
                                "independent_objective_change": change,
                                "independent_armijo_gap": change - ARMIJO_C1 * q,
                                "accepted_on_both_paths": bool(q < 0 and value <= row["armijo_rhs"]
                                                              and change < 0 and change <= ARMIJO_C1 * q)})
            emit(row)
            probes.append(row)
        passed = [p for p in probes if p.get("accepted_on_both_paths")]
        best = min(passed, key=lambda p: p["independent_objective_change"], default=None)
        summary[label] = {
            "direction_inf": infnorm(ray),
            "first_zero_conflict": conflict if math.isfinite(conflict) else None,
            "anchor": anchor, "trials": len(probes),
            "feasible": sum(p["feasible"] for p in probes),
            "finite_pruning_checks": sum(p.get("finite", False) for p in probes),
            "passes": len(passed),
            "unit": next(p for p in probes if p["alpha"] == 1.),
            "best_probe": best,
            "largest_accepted_alpha": max((p["alpha"] for p in passed), default=None),
            "passing_unit_extensions": sum(p["alpha"] > 1 for p in passed),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, choices=(1000, 10000), required=True)
    for key in ("prep", "gate", "library", "boundary", "archive", "trace"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    started = perf_counter()
    engine, auditor, details = load_case(args.prep, args.gate, args.rows, args.library)
    basename = f"scaled-{args.rows}"
    archived = json.loads((args.archive / (basename + ".json")).read_text())
    for key in ("alignment_weights_sha256", "indices_sha256", "reference_profile_sha256",
                "library_sha256", "topology_sha256"):
        assert archived[key] == details[key]
    checkpoint = np.load(args.archive / (basename + ".npy"), allow_pickle=False)
    ledger = [json.loads(line) for line in (args.archive / (basename + ".jsonl")).read_text().splitlines()]
    guard = ZeroBoundary(engine, args.boundary)
    with args.trace.open("x", encoding="utf-8") as log:
        def emit(record):
            log.write(json.dumps(record, allow_nan=False) + "\n")
            log.flush()

        terminal, direction, gradient, sha = rebuild(engine, ledger, checkpoint, archived, emit)
        results = scan(engine, auditor, checkpoint, terminal, direction, gradient, guard, emit)
    details.update({"checkpoint_sha256": sha, "replay_verified": True,
                    "historical_accepted_steps_reconstructed": archived["search"]["iterations"],
                    "archived_independent_gate": archived["independent_terminal"],
                    "ray_summaries": results, "elapsed_seconds": perf_counter() - started,
                    "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
    print(json.dumps(details, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

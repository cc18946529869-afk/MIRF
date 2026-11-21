"""Compare a projected Armijo initializer on frozen fixed-topology fixtures."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import resource

import numpy as np

from .check_f81_observed_bridge import load_case, start_lengths, audit_trace
from .f81_optimize import ARMIJO_C1, optimize_projected, projected_norm, projected_scale_step
from .f81_zero_boundary import ZeroBoundary


def check_initializer():
    point = np.array([0., 0.1, 2.])
    direction = np.array([2e5, -1e5, -1e-10])
    gradient = np.array([-2e7, 1e-5, 3.])
    value = 25.
    step = projected_scale_step(point, direction, gradient, value, 1.)
    dx = np.maximum(0, point + step * direction) - point
    predicted = -ARMIJO_C1 * (gradient @ dx)
    assert 0.009 * value < predicted <= 0.01 * value
    assert projected_scale_step(point, direction * 1e-14, gradient, value, 1.) == 1.
    assert projected_scale_step(point, -direction, gradient, value, 1.) == 1.
    assert projected_scale_step(np.array([0.]), np.array([1e308]), np.array([-1e308]), value, 1.) < 1.
    print(json.dumps({"passed": True, "projection_at_unit_step_differs_from_raw": True,
                      "accepted_initial_prediction_ratio": predicted / value}))


def replay(engine, independent, point, derivative, terminal, directions, guard):
    n_sites = len(terminal.site_log_likelihood)
    value = -terminal.log_likelihood / n_sites
    reference = independent.evaluate(point, threads=4, gradient=False, log_domain=True)
    output = []
    for name, unscaled, direction in directions:
        scaled = projected_scale_step(point, direction, derivative, value, unscaled)
        trials = []
        for index in range(24):
            alpha = math.ldexp(scaled, -index)
            trial = np.maximum(0, point + alpha * direction)
            dx = trial - point
            q = float(derivative @ dx)
            rhs = value + ARMIJO_C1 * q
            row = {"attempt": index + 1, "step": alpha, "projected_inner_product": q,
                   "armijo_rhs": rhs, "displacement_inf": float(np.abs(dx).max())}
            if q >= 0 or rhs < 0 or not guard.feasible(trial):
                row["outcome"] = "no_descent_bound_or_feasibility"
            else:
                objective = -engine.evaluate(trial, threads=4, gradient=False).log_likelihood / n_sites
                row.update({"objective": objective, "gap": objective - rhs,
                            "outcome": "accepted" if objective <= rhs else "rejected"})
                if row["outcome"] == "accepted":
                    check = independent.evaluate(trial, threads=4, gradient=False, log_domain=True)
                    gain = -math.fsum(check.site_log_likelihood - reference.site_log_likelihood) / n_sites
                    row.update({"independent_objective_change": gain,
                                "independent_armijo_gap": gain - ARMIJO_C1 * q})
            trials.append(row)
            if row["outcome"] == "accepted":
                break
        output.append({"direction": name, "previous_start_step": unscaled,
                       "scaled_start_step": scaled, "old_search_ledger": "frozen baseline JSONL",
                       "current_objective": value, "trials": trials})
    return output


def run(args):
    engine, auditor, facts = load_case(args.prep, args.gate, args.rows, args.library)
    guard = ZeroBoundary(engine, args.boundary)
    events = []
    replays = []

    def finished(point, derivative, terminal, last, status):
        if args.baseline_replay:
            assert status == "no_finite_armijo_step" and len(last) == 2
            replays.extend(replay(engine, auditor, point, derivative, terminal, last, guard))

    with args.trace.open("x", encoding="utf-8") as stream:
        def emit(event):
            events.append(event)
            stream.write(json.dumps(event, allow_nan=False) + "\n")
            stream.flush()

        point, search = optimize_projected(
            engine, start_lengths("short", facts["edges"]), emit=emit, threads=4, maxiter=150,
            maxeval=260, gtol=1e-5, restrict_history=True, zero_boundary=guard,
            search_diagnostics=True, terminal_diagnostic=finished if args.baseline_replay else None,
            scale_initial_step=not args.baseline_replay,
        )
    if args.baseline_replay:
        previous = [json.loads(row) for row in args.baseline_replay.read_text().splitlines()]
        numeric = lambda rows: [{k: v for k, v in x.items() if k != "seconds"} for x in rows]
        assert numeric(events) == numeric(previous), "Old solver must reconstruct the archived terminal"
        assert all(r["trials"][-1].get("independent_armijo_gap", math.inf) <= 0 for r in replays)
    np.save(args.checkpoint, point, allow_pickle=False)
    ordinary = engine.evaluate(point, threads=4)
    independent = auditor.evaluate(point, threads=4, log_domain=True)
    grad, ref = (-result.length_gradient / facts["sites"] for result in (ordinary, independent))
    difference = np.abs(grad - ref)
    proj = projected_norm(point, ref)
    assert proj == float(np.linalg.norm(point - np.clip(point - ref, 0, None), ord=np.inf))
    return {**facts, "search": search, "trace": audit_trace(events, search),
            "same_terminal_as_archived_replay": args.baseline_replay is not None,
            "same_terminal_ray_replay": replays,
            "checkpoint_sha256": hashlib.sha256(point.tobytes()).hexdigest(),
            "independent_terminal": {"log_likelihood": independent.log_likelihood,
                                     "gradient_max_abs_difference": float(difference.max()),
                                     "projected_gradient": proj, "abs_gate": 1e-8,
                                     "gradient_agreement_passed": bool(difference.max() <= 1e-8),
                                     "convergence_passed": bool(proj <= 1e-5)},
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    for opt in ("prep", "gate", "library", "boundary", "trace", "checkpoint", "baseline-replay"):
        parser.add_argument("--" + opt, type=Path)
    parser.add_argument("--rows", type=int, choices=(1000, 10000))
    args = parser.parse_args()
    if args.check:
        check_initializer()
    else:
        print(json.dumps(run(args), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

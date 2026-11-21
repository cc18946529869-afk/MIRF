"""Run the fixed-budget F81 optimizer alternatives on the same archived fixture."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource

import numpy as np

from .check_f81_observed_bridge import audit_trace, load_case, start_lengths
from .f81_optimize import ARMIJO_C1, optimize_projected, projected_norm
from .f81_zero_boundary import ZeroBoundary


STRATEGIES = {
    "baseline": {},
    "expansion": {"expand_steps": True},
    "recovery": {"recover_history": True},
    "combined": {"expand_steps": True, "recover_history": True},
    "normalized": {"normalized_only": True},
}


def array_hash(x):
    return hashlib.sha256(x.tobytes()).hexdigest()


def run(args):
    engine, auditor, facts = load_case(args.prep, args.gate, args.rows, args.library)
    guard = ZeroBoundary(engine, args.boundary)
    archived = json.loads((args.archive / f"scaled-{args.rows}.json").read_text())
    for key in ("alignment_weights_sha256", "indices_sha256", "reference_profile_sha256",
                "library_sha256", "topology_sha256"):
        assert archived[key] == facts[key]
    if args.start == "archived":
        initial = np.load(args.archive / f"scaled-{args.rows}.npy", allow_pickle=False)
        assert array_hash(initial) == archived["checkpoint_sha256"]
    else:
        initial = start_lengths("short", facts["edges"])

    events = []
    with args.trace.open("x", encoding="utf-8") as stream:
        def emit(event):
            events.append(event)
            stream.write(json.dumps(event, allow_nan=False) + "\n")
            stream.flush()

        candidate, fit = optimize_projected(
            engine, initial, emit=emit, maxeval=260, maxiter=150, gtol=1e-5, threads=4,
            restrict_history=True, zero_boundary=guard, scale_initial_step=True,
            search_diagnostics=True, iteration_diagnostics=True, **STRATEGIES[args.strategy],
        )
    ledger = audit_trace(events, fit)
    assert ledger["accepted_monotone"] and guard.feasible(candidate)
    numerical = lambda rows: [{k: v for k, v in row.items() if k != "seconds"}
                              for row in rows if row["event"] != "direction_state"]
    baseline_reproduction = None
    if args.strategy == "baseline" and args.start == "short":
        prior = (args.archive / f"scaled-{args.rows}.jsonl").read_text().splitlines()
        baseline_reproduction = numerical(events) == numerical(map(json.loads, prior))
        assert baseline_reproduction and array_hash(candidate) == archived["checkpoint_sha256"]

    # The fitted objective is already recomputed inside the solver. Evaluate
    # the ordinary gradient and an independent rerooted log gradient once here.
    ordinary = engine.evaluate(candidate, threads=4)
    independent = auditor.evaluate(candidate, threads=4, log_domain=True)
    grad = -ordinary.length_gradient / facts["sites"]
    ref = -independent.length_gradient / facts["sites"]
    difference = float(np.max(np.abs(grad - ref)))
    residual = projected_norm(candidate, ref)
    assert residual == float(np.max(np.abs(candidate - np.clip(candidate - ref, 0., None))))
    assert ordinary.log_likelihood == fit["terminal_log_likelihood"]
    assert projected_norm(candidate, grad) == fit["terminal_projected_gradient_per_site"]
    slopes = [e["free_gradient_cosine"] for e in events if e["event"] == "direction_state"
              and e["free_gradient_cosine"] is not None]
    return {**facts, "arm": args.strategy, "start": args.start, "initial_sha256": array_hash(initial),
            "terminal_sha256": array_hash(candidate), "baseline_matches_prior_exactly": baseline_reproduction,
            "search": fit, "ledger_audit": ledger, "armijo_c1": ARMIJO_C1, "max_line_trials": 24,
            "search_cosine_median": float(np.median(slopes)) if slopes else None,
            "search_cosine_last": slopes[-1] if slopes else None,
            "terminal_mean_negative_log_likelihood": -ordinary.log_likelihood / facts["sites"],
            "independent_terminal": {"feasible_by_zero_guard": True,
                                     "log_likelihood": independent.log_likelihood,
                                     "projected_gradient_inf": residual,
                                     "gradient_max_abs_difference": difference,
                                     "projected_gradient_passed": residual <= 1e-5,
                                     "gradient_agreement_passed": difference <= 1e-8},
            "passes_both_gates": bool(fit["passed"] and residual <= 1e-5 and difference <= 1e-8),
            "additional_audit_evaluations": 2,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, choices=(1000, 10000), required=True)
    parser.add_argument("--strategy", choices=STRATEGIES, required=True)
    parser.add_argument("--start", choices=("short", "archived"), required=True)
    for key in ("prep", "gate", "library", "boundary", "archive", "trace"):
        parser.add_argument("--" + key, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args), allow_nan=False, indent=2))


if __name__ == "__main__":
    main()

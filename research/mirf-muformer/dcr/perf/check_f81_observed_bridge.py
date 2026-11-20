"""Run one frozen-optimizer bridge case on an observational subset and fixed fixture."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
from time import perf_counter

import numpy as np

import potts_fit
from .check_f81_pruning import artificial_topology
from .f81_optimize import optimize_projected
from .f81_unrooted import UnrootedPruning, UnrootedTopology
from .f81_zero_boundary import ZeroBoundary


STARTS = ("short", "long", "heterogeneous")
FILES = ("mature_codes.npy", "neighbor_counts.tsv")
N_SITES = 263
GTOL = 1e-5


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


class ConflictAuditPruning(UnrootedPruning):
    """Test the exact zero-edge feasibility condition independently of pruning."""

    def __init__(self, topology, observations, profiles, library, checks):
        super().__init__(topology, observations, profiles, library)
        self.checks = checks

    def evaluate(self, lengths, **kwargs):
        result = super().evaluate(lengths, **kwargs)
        if np.isfinite(result.log_likelihood):
            return result
        layout = self.topology.layout
        zero = np.asarray(lengths)[self.topology.child_to_edge] == 0
        component = np.arange(len(layout.parent)+1)
        for child in reversed(np.flatnonzero(zero)):
            component[child] = component[layout.parent[child]]
        tip = component[:layout.leaves]
        order = np.argsort(tip, kind="stable")
        equal = tip[order[1:]] == tip[order[:-1]]
        rows = self.pruning.states.T
        mismatches = np.any(rows[order[1:][equal]] != rows[order[:-1][equal]], axis=1)
        # Adjacent unequal rows in component order witness a structural conflict.
        contrary = np.unique(tip[order[1:][equal]][mismatches])
        self.checks.append({"zero_edges": int(zero.sum()),
                            "conflicting_zero_components": int(len(contrary))})
        return result


def load_case(prep, gate, sample_size, library, boundary_checks=None):
    recorded = json.loads(gate.read_text())["sha256"]
    actual = {name: digest(prep/name) for name in FILES}
    if any(recorded[name] != actual[name] for name in FILES):
        raise ValueError("Observational alignment/weights do not match the frozen study files")
    matrix, weights, total = potts_fit.load_official_alignment(prep)
    if matrix.shape != (209644, N_SITES):
        raise ValueError("Unexpected full reference shape")
    hist = np.vstack([np.bincount(matrix[:, i], weights=weights, minlength=21) for i in range(N_SITES)])
    profiles = hist / total
    if int(np.count_nonzero(profiles == 0)) != 72:
        raise ValueError("Unexpected reference zero-frequency pattern")
    indices = np.linspace(0, len(matrix)-1, sample_size, dtype=np.int64)
    sample = matrix[indices]
    if np.any(np.take_along_axis(profiles, sample.T, axis=1) == 0):
        raise ValueError("Observed subset contains a state outside its frozen profile")
    counts = np.array([np.unique(sample[:, k]).size for k in range(N_SITES)])
    rooted = artificial_topology(sample_size, binary=True)
    tree, _, _ = UnrootedTopology.from_rooted(rooted, np.ones(len(rooted.parent)))

    # Reconstruct the same unrooted edges in the original parameter order,
    # then choose a different root for the independent terminal replay.
    pairs = np.empty((len(tree.child_to_edge), 2), dtype=np.int32)
    pairs[tree.child_to_edge] = np.column_stack((np.arange(len(tree.layout.parent)), tree.layout.parent))
    auditor_tree = UnrootedTopology.from_edges(sample_size, pairs, root=sample_size)
    assert len(pairs) == 2*sample_size-3
    assert not np.array_equal(tree.child_to_edge, auditor_tree.child_to_edge)
    facts = {
        "scope": "same archived observational MSA; deterministic row subset on a data-independent fixed topology",
        "sample_size": sample_size, "sites": N_SITES, "edges": len(pairs),
        "full_weight_total": total, "alignment_weights_sha256": actual,
        "reference_support_nonzero": int(np.count_nonzero(profiles)),
        "reference_support_zero": int(np.count_nonzero(profiles == 0)),
        "observed_union_support": int(np.unique(sample).size),
        "observed_site_support_min_median_max": [int(counts.min()), float(np.median(counts)), int(counts.max())],
        "indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
        "reference_profile_sha256": hashlib.sha256(profiles.tobytes()).hexdigest(),
        "topology_sha256": hashlib.sha256(pairs.tobytes()).hexdigest(),
        "library_sha256": digest(library),
    }
    engine = (UnrootedPruning(tree, sample, profiles, library) if boundary_checks is None else
              ConflictAuditPruning(tree, sample, profiles, library, boundary_checks))
    return engine, UnrootedPruning(auditor_tree, sample, profiles, library), facts


def start_lengths(name, size):
    if name == "short":
        return np.full(size, .06)
    if name == "long":
        return np.full(size, .45)
    return .08 + .26 * ((53*np.arange(size) % 97) / 96)


def audit_trace(events, result):
    evaluations = [record for record in events if record["event"] == "evaluation"]
    accepted = [record for record in events if record["event"] == "accepted"]
    assert [row["call"] for row in evaluations] == list(range(1, len(evaluations)+1))
    assert len(evaluations) == result["solver_likelihood_gradient_evaluations"]
    assert len(accepted) == result["iterations"]
    assert sum(not row["finite"] for row in evaluations) == result["infeasible_trials"]
    assert len(evaluations) <= 260
    accepted_calls = [1, *(row["calls"] for row in accepted)]
    assert accepted_calls == sorted(set(accepted_calls))
    accepted_values = []
    for call in accepted_calls:
        evaluation = evaluations[call-1]
        assert evaluation["finite"]
        accepted_values.append(evaluation["log_likelihood"])
    largest_drop = float(max(0., -np.diff(accepted_values).min())) if accepted else 0.
    return {
        "accepted_monotone": bool(largest_drop <= 8*np.spacing(abs(accepted_values[0]))),
        "max_accepted_likelihood_drop": largest_drop,
        "accepted_finite": True,
        "zero_edges_from_trace": evaluations[accepted_calls[-1]-1]["zero_edges"],
        "last_accepted_likelihood": accepted_values[-1],
    }


def terminal_coordinate_probe(engine, guard, lengths, independent):
    """Check an improving feasible coordinate move without changing the endpoint."""
    derivative = -independent.length_gradient / len(independent.site_log_likelihood)
    residual = lengths - np.clip(lengths - derivative, 0, None)
    coordinate = int(np.argmax(np.abs(residual)))
    if residual[coordinate] == 0:
        return {"exact_stationarity_in_probe": True}
    direction = np.zeros_like(lengths)
    direction[coordinate] = -np.sign(residual[coordinate])
    limit = guard.first_conflict(lengths, direction)
    start = float(lengths[coordinate])
    proposed = max(1e-12, min(1e-3, max(start, 1e-6)/4))
    if direction[coordinate] < 0:
        proposed = min(proposed, start/4)
    if np.isfinite(limit):
        proposed = min(proposed, 0.5*limit)
    checks = []
    for attempt in range(8):
        step = proposed * 2.**(-attempt)
        trial = np.maximum(0, lengths + step*direction)
        if np.array_equal(trial, lengths) or not guard.feasible(trial):
            continue
        value = engine.evaluate(trial, gradient=False, threads=1, log_domain=True).log_likelihood
        if np.isfinite(value):
            gain = value - independent.log_likelihood
            checks.append({"step":step, "log_likelihood_gain":gain})
            if gain > 16*np.spacing(abs(independent.log_likelihood)):
                break
    return {"edge_length_before_probe":start, "local_derivative_per_site":float(derivative[coordinate]),
            "direction_sign": int(direction[coordinate]), "checks":checks}


def run(args):
    boundary_checks = [] if args.audit_boundary else None
    engine, repeat, facts = load_case(args.prep, args.gate, args.rows, args.library, boundary_checks)
    lengths = start_lengths(args.start, facts["edges"])
    guard = ZeroBoundary(engine, args.boundary_library) if args.boundary_library else None
    if guard is not None:
        facts["boundary_library_sha256"] = digest(args.boundary_library)
    events = []
    with args.trace.open("x", encoding="utf-8") as log:
        def emit(record):
            events.append(record)
            log.write(json.dumps(record, allow_nan=False) + "\n")
            log.flush()
        try:
            endpoint, fit = optimize_projected(engine, lengths, restrict_history=True, threads=args.threads,
                                               gtol=GTOL, maxiter=150, maxeval=260, emit=emit, zero_boundary=guard)
        except FloatingPointError as error:
            return {**facts, "start": args.start, "passed_bridge_gate": False, "status": "arithmetic_exception",
                    "exception": str(error), "evaluations_recorded_before_exception": len(
                         [r for r in events if r["event"] == "evaluation"])}
    assert fit["restricted_curvature_to_free_coordinates"] is True
    ledger = audit_trace(events, fit)
    assert (ledger["zero_edges_from_trace"] == fit["terminal_zero_edges"] == np.count_nonzero(endpoint == 0))
    assert abs(ledger["last_accepted_likelihood"] - fit["terminal_log_likelihood"]) < 1e-8
    mark = perf_counter()
    logcheck = repeat.evaluate(endpoint, threads=args.threads, log_domain=True)
    audit_seconds = perf_counter() - mark
    if logcheck.length_gradient is None:
        raise ValueError("The independent terminal is impossible")
    derivative = -logcheck.length_gradient / facts["sites"]
    residual = float(np.max(np.abs(endpoint - np.clip(endpoint - derivative, 0, None))))
    direct = engine.evaluate(endpoint, threads=args.threads)
    max_score_difference = float(np.max(np.abs(direct.length_gradient - logcheck.length_gradient)) / facts["sites"])
    per_site_difference = float(np.max(np.abs(direct.site_log_likelihood - logcheck.site_log_likelihood)))
    cert = {
        "independent_root_and_log_domain": True,
        "terminal_log_domain_sites": logcheck.log_domain_sites,
        "terminal_log_likelihood": logcheck.log_likelihood,
        "projected_gradient_per_site": residual,
        "terminal_max_site_log_likelihood_difference": per_site_difference,
        "terminal_max_per_site_score_difference": max_score_difference,
        "likelihood_score_agreement": per_site_difference <= 2e-9 and max_score_difference <= 1e-8,
        "wall_seconds": audit_seconds,
        "additional_normal_domain_comparison_evaluations": 1,
    }
    return {**facts, "start": args.start, "initial_range": [float(lengths.min()), float(lengths.max())],
            "threads": args.threads, "optimizer": fit, "ledger_check": ledger, "independent_terminal": cert,
            **({"terminal_probe": terminal_coordinate_probe(repeat, ZeroBoundary(repeat, args.boundary_library),
                                                           endpoint, logcheck)}
               if args.probe_terminal else {}),
            "zero_branch_trial_replay": (None if boundary_checks is None else
                {"checks": len(boundary_checks),
                 "confirmed_conflicting_trials": sum(r["conflicting_zero_components"] > 0 for r in boundary_checks),
                 "detail": boundary_checks}),
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "passed_bridge_gate": bool(fit["passed"] and ledger["accepted_monotone"] and
                                       cert["likelihood_score_agreement"] and residual <= GTOL)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prep", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--rows", type=int, required=True, choices=(1000, 10000))
    parser.add_argument("--start", required=True, choices=STARTS)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--audit-boundary", action="store_true")
    parser.add_argument("--boundary-library", type=Path)
    parser.add_argument("--probe-terminal", action="store_true")
    args = parser.parse_args()
    if args.probe_terminal and not args.boundary_library:
        parser.error("--probe-terminal requires --boundary-library")
    print(json.dumps(run(args), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

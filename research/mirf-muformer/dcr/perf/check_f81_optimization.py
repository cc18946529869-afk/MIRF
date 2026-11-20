"""Reproducible root and optimizer tests using only artificial categorical fixtures."""
from __future__ import annotations

import argparse
from functools import partial
import json
import math
from pathlib import Path
import resource

import numpy as np
import scipy

from .check_f81_pruning import artificial_topology, dense_enumeration
from .f81_optimize import optimize, optimize_projected
from .f81_pruning import Pruning, Topology, build
from .f81_unrooted import UnrootedPruning, UnrootedTopology


def root_checks(library):
    rows = np.array([[0, 20, 5], [5, 10, 5], [20, 5, 0], [10, 20, 20]])
    profile = np.zeros((3, 21))
    profile[:, [0, 5, 10, 20]] = [[.10, .25, .40, .25], [.2, .25, .25, .30], [.25, .35, .05, .35]]
    legacy = Topology.from_parents(4, [4, 4, 5, 5, 6, 6])
    before = Pruning(legacy, rows, profile, library)
    leaf_lengths = [.06, .27, .32, .19]
    log_error = grad_error = split_score_error = 0.0
    for left in (0, .17, .4, .55):
        x = np.array([*leaf_lengths, left, .55 - left])
        canonical, lengths, old_to_new = UnrootedTopology.from_rooted(legacy, x)
        assert len(lengths) == 5 and canonical.layout.max_degree == 3
        after = UnrootedPruning(canonical, rows, profile, library)
        old = before.evaluate(x)
        new = after.evaluate(lengths)
        reference = dense_enumeration(rows, profile, legacy.parent, x)
        log_error = max(log_error, abs(old.log_likelihood - new.log_likelihood),
                        float(np.max(np.abs(new.site_log_likelihood - reference))))
        grad_error = max(grad_error, float(np.max(np.abs(old.length_gradient
                                                         - new.length_gradient[old_to_new]))))
        split_score_error = max(split_score_error, abs(old.length_gradient[-1] - old.length_gradient[-2]))
    assert log_error < 1e-12 and grad_error < 1e-11 and split_score_error < 1e-11
    edges = np.array([[0, 4], [1, 4], [2, 5], [3, 5], [4, 5]])
    e1, e2 = (UnrootedPruning(UnrootedTopology.from_edges(4, edges, root), rows, profile, library)
              for root in (4, 5))
    vals = np.r_[leaf_lengths, 0.31]
    a, b = e1.evaluate(vals), e2.evaluate(vals)
    re_root_difference = abs(a.log_likelihood - b.log_likelihood)
    re_root_grad = float(np.max(np.abs(a.length_gradient - b.length_gradient)))
    assert re_root_difference < 1e-12 and re_root_grad < 1e-11

    star = UnrootedTopology.from_edges(4, np.column_stack([np.arange(4), np.full(4, 4)]))
    star_engine = UnrootedPruning(star, rows, profile, library)
    vals[-1] = 0
    binary = e1.evaluate(vals)
    multifurcating = star_engine.evaluate(vals[:-1])
    zero_error = abs(binary.log_likelihood - multifurcating.log_likelihood)
    zero_gradient_error = float(np.max(np.abs(binary.length_gradient[:-1] - multifurcating.length_gradient)))
    step = 2**-21
    def value(s):
        lengths = vals.copy()
        lengths[-1] = s
        return float(dense_enumeration(rows, profile, e1.topology.layout.parent,
                                      lengths[e1.topology.child_to_edge]).sum())
    one_sided = (4 * (value(step) - value(0)) - (value(2 * step) - value(0))) / (2 * step)
    zero_one_sided_error = abs(binary.length_gradient[-1] - one_sided)
    assert zero_error < 1e-12 and zero_gradient_error < 1e-11 and zero_one_sided_error < 1e-7
    try:
        UnrootedTopology.from_edges(4, np.column_stack([np.arange(6), legacy.parent]))
    except ValueError:
        rejected = True
    else:
        raise AssertionError("Degree-two optimizer root was accepted")
    return {
        "split_positions_tested": 4, "legacy_rooted_edge_count": 6, "unrooted_edge_count": 5,
        "max_likelihood_or_dense_reference_error": log_error,
        "max_gradient_error_after_mapping": grad_error,
        "max_difference_between_split_edge_scores": split_score_error,
        "reroot_likelihood_error": re_root_difference, "reroot_gradient_error": re_root_grad,
        "zero_edge_contraction_likelihood_error": zero_error,
        "zero_edge_contraction_shared_gradient_error": zero_gradient_error,
        "zero_edge_right_derivative_error": zero_one_sided_error,
        "legacy_degree_two_root_rejected_for_optimization": rejected,
    }


def toy_data(n, seed):
    """Four invented categories, with fixed profile patterns and no biological input."""
    rng = np.random.default_rng(seed)
    old = artificial_topology(n, binary=True)
    s = .045 + (.31 - .045) * rng.random(len(old.parent))
    symbols = np.array([0, 5, 10, 20], dtype=np.uint8)
    nsites = 263
    proportions = np.array([.18, .21, .28, .33])
    active_profiles = np.array([np.roll(proportions, j % 4) for j in range(nsites)])
    cdf = active_profiles.cumsum(axis=1)[:, :-1]
    matrix = np.empty((len(old.parent)+1, nsites), dtype=np.uint8)
    matrix[-1] = symbols[(rng.random(nsites)[:, None] >= cdf).sum(axis=1)]
    refresh = -np.expm1(-s)
    for child in range(len(s)-1, -1, -1):
        new = symbols[(rng.random(nsites)[:, None] >= cdf).sum(axis=1)]
        matrix[child] = np.where(rng.random(nsites) < refresh[child], new, matrix[old.parent[child]])
    profile = np.zeros((nsites, 21))
    profile[:, symbols] = active_profiles
    topology, lengths, _ = UnrootedTopology.from_rooted(old, s)
    assert len(lengths) == 2*n - 3
    return topology, matrix[:n], profile


def run_one(topology, rows, profile, library, record, stage, threads, method):
    engine = UnrootedPruning(topology, rows, profile, library)
    with record.open("x", encoding="utf-8") as stream:
        def log(x):
            stream.write(json.dumps(x, allow_nan=False) + "\n")
            stream.flush()
        nedge = len(topology.child_to_edge)
        if method in ("projected", "projected-free"):
            solver = partial(optimize_projected, restrict_history=method == "projected-free")
        else:
            solver = partial(optimize, scale=.01 if method == "scaled-scipy" else 1.)
        x, results = solver(engine, np.full(nedge, 0.18), emit=log, threads=threads)
    results.update(stage=stage, leaves=len(rows), sites=rows.shape[1], edges=nedge, threads=threads,
                   scipy_version=scipy.__version__, numpy_version=np.__version__,
                   max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    assert results["terminal_log_likelihood"] >= results["initial_log_likelihood"]
    return x, results


def small_checks(library, logs, method):
    topology = UnrootedTopology.from_edges(4, np.column_stack([np.arange(4), np.full(4, 4)]))
    rows = np.tile(np.resize([0,5,10,20],263), (4, 1))
    profile = np.zeros((263, 21))
    profile[:, [0, 5, 10, 20]] = .25
    x, boundary = run_one(topology, rows, profile, library, logs/"exact-zero.jsonl", "all-zero-star", 1, method)
    assert boundary["passed"] and np.all(x == 0)
    assert abs(boundary["terminal_log_likelihood"] - 263*math.log(.25)) < 1e-10
    topology, rows, profile = toy_data(8, 61)
    _, interior = run_one(topology, rows, profile, library, logs/"small-fit.jsonl", "small-heterogeneous-tree", 1, method)
    assert interior["passed"]
    return [boundary, interior]


def full_coordinate_check():
    n = 209644
    old = artificial_topology(n, binary=True)
    new, lengths, mapping = UnrootedTopology.from_rooted(old, np.ones(len(old.parent)))
    assert len(lengths) == 2*n - 3 and new.layout.max_degree == 3
    assert (np.bincount(mapping) == 2).sum() == np.count_nonzero(lengths == 2) == 1
    assert (np.bincount(mapping) == 1).sum() == len(lengths) - 1
    return {"leaves": n, "root_degree": 3, "legacy_edges": len(old.parent),
            "unrooted_edges": len(lengths), "merged_edges": 2, "likelihood_or_optimization_run": False}


def search_budget_check(library):
    topology, rows, profile = toy_data(8, 61)
    engine = UnrootedPruning(topology, rows, profile, library)
    starts = np.full(len(topology.child_to_edge), .18)
    events = []
    endpoint, summary = optimize_projected(engine, starts, threads=1, maxeval=1,
                                           restrict_history=True, emit=events.append)
    assert summary["solver_likelihood_gradient_evaluations"] == 1
    assert summary["additional_terminal_evaluations"] == 1
    assert summary["status"] == "hard_limit_on_likelihood_gradient_evaluations" and not summary["passed"]
    assert np.array_equal(starts, endpoint) and len(events) == 1
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--leaves", type=int, choices=(1000, 10000))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--method", choices=("projected", "projected-free", "scipy", "scaled-scipy"), default="projected-free")
    parser.add_argument("--logs", type=Path)
    parser.add_argument("--full-layout-only", action="store_true")
    args = parser.parse_args()
    if args.full_layout_only:
        print(json.dumps(full_coordinate_check(), indent=2, allow_nan=False))
        return
    if not args.library or not args.logs:
        parser.error("Specify --library and --logs")
    if args.check:
        command = build(args.library)
        result = {"build_command": command, "root_tests": root_checks(args.library),
                  "optimizations": small_checks(args.library, args.logs, args.method),
                  "evaluation_budget_check": search_budget_check(args.library)}
    elif args.leaves and args.seed is not None:
        topology, rows, profile = toy_data(args.leaves, args.seed)
        _, result = run_one(topology, rows, profile, args.library,
                            args.logs/f"scale-{args.leaves}-{args.seed}.jsonl", "artificial-scale", args.threads, args.method)
        result["seed"] = args.seed
    else:
        parser.error("Use --check or provide --leaves and --seed")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

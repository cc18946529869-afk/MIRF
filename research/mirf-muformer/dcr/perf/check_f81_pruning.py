"""Validate arbitrary-tree kernel and time fixed topologies on observed rows."""
from __future__ import annotations

import argparse
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import resource
from time import perf_counter

import numpy as np

import potts_fit
from .check_categorical_tree import profiles
from .f81_pruning import Pruning, Topology, build


def dense_enumeration(rows, pi, parent, lengths):
    """Independent sum over internal state assignments for small trees."""
    leaves = len(rows)
    n = len(parent) + 1
    sums = []
    lengths = np.asarray(lengths, dtype=np.longdouble)
    retention = np.exp(-lengths)
    redraw = -np.expm1(-lengths)
    for site, freq in enumerate(pi):
        matrices = retention[:, None, None] * np.eye(21) + redraw[:, None, None] * freq[None, None, :]
        configs = np.array(list(product(range(21), repeat=n - leaves)), dtype=int)
        states = np.empty((len(configs), n), dtype=int)
        states[:, :leaves] = rows[:, site]
        states[:, leaves:] = configs
        probability = freq[states[:, -1]].astype(np.longdouble)
        for child, node in enumerate(parent):
            probability *= matrices[child, states[:, node], states[:, child]]
        sums.append(np.log(probability.sum(dtype=np.longdouble)))
    return np.asarray(sums)


def checks(library):
    group1 = np.array(profiles()[0], dtype=float)
    group2 = np.array(profiles()[1], dtype=float)
    group3 = np.array(profiles()[2], dtype=float)
    pi = np.vstack([x / x.sum() for x in (group1, group2, group3, group1)])
    rows = np.array([[0, 20, 20, 3], [2, 0, 20, 4], [20, 5, 20, 3],
                     [1, 16, 20, 15], [3, 7, 20, 18]], dtype=np.uint8)
    cases = (
        (np.array([4, 4, 5, 6, 5, 6]), np.array([.12, 0, .21, .38, .07, .24])),
        (np.array([5, 5, 5, 6, 6, 6]), np.array([0, .13, .21, .38, .07, .24])),
        (np.array([5, 5, 6, 6, 7, 7, 7]), np.array([.17, .28, .07, .31, 0, .22, .18])),
        (np.array([5, 5, 5, 5, 5]), np.array([.16, 0, .04, .371, .43])),
    )
    dense_error = gradient_error = parallel_error = 0.0
    log_check_error = 0.0
    comparisons = 0
    for parent, lengths in cases:
        case_rows = rows[:4] if parent[0] == 4 else rows
        layout = Topology.from_parents(len(case_rows), parent)
        evaluator = Pruning(layout, case_rows, pi, library)
        answer = evaluator.evaluate(lengths)
        same = evaluator.evaluate(lengths, threads=2)
        assert np.array_equal(answer.site_log_likelihood, same.site_log_likelihood)
        parallel_error = max(parallel_error, float(np.max(np.abs(answer.length_gradient - same.length_gradient))))
        replay = evaluator.evaluate(lengths, log_domain=True)
        assert replay.log_domain_sites == len(pi)
        log_check_error = max(log_check_error, abs(answer.log_likelihood - replay.log_likelihood),
                              float(np.max(np.abs(answer.length_gradient - replay.length_gradient))))
        expected = dense_enumeration(case_rows, pi, parent, lengths)
        dense_error = max(dense_error, float(np.max(np.abs(answer.site_log_likelihood - expected))))
        for edge in range(len(lengths)):
            changed = lengths.astype(np.longdouble)
            if lengths[edge] == 0:
                # Fourth-order forward difference at the feasible boundary.
                step = 2 ** -23
                displacements = []
                for i in range(1, 5):
                    changed[edge] = i * step
                    value = dense_enumeration(case_rows, pi, parent, changed)
                    displacements.append((value - expected).sum())
                deriv = np.dot((48, -36, 16, -3), displacements) / (12 * step)
            else:
                step = 2 ** -20
                changed[edge] += step
                plus = dense_enumeration(case_rows, pi, parent, changed)
                changed[edge] -= 2 * step
                minus = dense_enumeration(case_rows, pi, parent, changed)
                deriv = ((plus - minus) / (2 * step)).sum()
            gradient_error = max(gradient_error, float(abs(deriv - answer.length_gradient[edge])))
            comparisons += 1
    assert dense_error < 1e-12 and gradient_error < 5e-7 and parallel_error < 1e-11, (dense_error, gradient_error, parallel_error)

    # The zero edge links the two internal nodes. Contracting it leaves the
    # exact joint evidence on the same five observation slots.
    root_zero_parent = np.array([5, 5, 5, 6, 6, 6])
    root_zero_edges = np.array([.11, .22, .34, .21, .29, 0])
    merged = Pruning(Topology.from_parents(5, np.full(5, 5)), rows, pi, library)
    split = Pruning(Topology.from_parents(5, root_zero_parent), rows, pi, library)
    collapse_error = abs(split.evaluate(root_zero_edges).log_likelihood
                         - merged.evaluate(root_zero_edges[:-1]).log_likelihood)
    assert collapse_error < 1e-12
    impossible = merged.evaluate(np.zeros(5))
    assert impossible.log_likelihood == -math.inf and impossible.length_gradient is None
    large_edges = merged.evaluate(np.full(5, 800.0))
    independent = np.log(np.take_along_axis(pi, rows.T, axis=1)).sum()
    saturation_error = abs(large_edges.log_likelihood - independent)
    assert large_edges.log_domain_sites == len(pi) and saturation_error < 1e-12

    nstar = 2000
    star_pi = np.full((1, 21), 1 / 21)
    star_rows = np.zeros((nstar, 1), dtype=np.int8)
    star_rows[nstar // 2:] = 1
    star_lengths = np.full(nstar, .01)
    star = Pruning(Topology.from_parents(nstar, np.full(nstar, nstar)), star_rows, star_pi, library)
    star_result = star.evaluate(star_lengths)
    stay = math.exp(-.01)
    redraw = -math.expm1(-.01)
    a = math.log(stay + redraw / 21)
    b = math.log(redraw / 21)
    star_reference = np.logaddexp.reduce([-math.log(21) + nstar / 2 * (a + b)] * 2
                                        + [-math.log(21) + nstar*b] * 19)
    high_degree_error = abs(star_result.log_likelihood - star_reference)
    assert star_result.log_domain_sites == 1 and high_degree_error < 1e-8
    assert log_check_error < 1e-9
    try:
        Topology.from_parents(5, np.full(5, 4))
    except ValueError:
        invalid_rejected = True
    else:
        raise AssertionError("Malformed topology was accepted")
    return {"topology_cases": len(cases), "all_branch_derivatives_compared": comparisons,
            "max_site_loglik_error_dense_enumeration": dense_error,
            "max_analytic_gradient_error_vs_dense_finite_diff": gradient_error,
            "max_gradient_roundoff_1_vs_2_threads": parallel_error,
            "zero_edge_contraction_loglik_error": collapse_error,
            "impossible_site_gives_minus_infinity": True,
            "scaled_vs_forced_log_error": log_check_error,
            "high_degree_star_leaves": nstar,
            "high_degree_star_log_domain_sites": star_result.log_domain_sites,
            "high_degree_star_loglik_error": high_degree_error,
            "long_finite_edge_saturation_loglik_error": saturation_error,
            "invalid_topology_rejected": invalid_rejected}


def artificial_topology(leaves, binary=False):
    """Create a fixed, data-independent irregular tree for cost testing."""
    level = list(range(leaves))
    parents = [-1] * leaves
    next_node = leaves
    while len(level) > 1:
        upper = []
        i = 0
        while i < len(level):
            count = min(2 if binary else 2 + next_node % 4, len(level) - i)
            if count == 1:
                upper.append(level[i])
            else:
                for j in range(i, i + count):
                    parents[level[j]] = next_node
                upper.append(next_node)
                parents.append(-1)
                next_node += 1
            i += count
        level = upper
    return Topology.from_parents(leaves, parents[:-1])


def observed_benchmark(prep, n, library, threads, binary):
    mark = perf_counter()
    all_rows, weights, total = potts_fit.load_official_alignment(prep)
    histogram = np.stack([np.bincount(all_rows[:, c], weights=weights, minlength=21)
                          for c in range(all_rows.shape[1])])
    profile = histogram / total
    indices = np.linspace(0, len(all_rows) - 1, n, dtype=int)
    layout = artificial_topology(n, binary)
    engine = Pruning(layout, all_rows[indices], profile, library)
    edge_id = np.arange(len(layout.parent))
    lengths = 0.04 + 0.45 * ((edge_id * 53) % 97) / 96
    start = perf_counter()
    scalar = engine.evaluate(lengths, gradient=False, threads=threads)
    likelihood_seconds = perf_counter() - start
    start = perf_counter()
    result = engine.evaluate(lengths, threads=threads)
    gradient_seconds = perf_counter() - start
    assert scalar.log_likelihood == result.log_likelihood

    # Compare a prespecified directional derivative, perturbing the supplied
    # benchmark lengths only. Input topology and lengths are not estimated.
    vector = np.where(edge_id % 3 == 0, -1., 1.)
    analytic = float(np.dot(result.length_gradient, vector))
    eps = 2 ** -16
    start = perf_counter()
    plus = engine.evaluate(lengths + eps * vector, gradient=False, threads=threads)
    minus = engine.evaluate(lengths - eps * vector, gradient=False, threads=threads)
    diff = math.fsum((plus.site_log_likelihood - minus.site_log_likelihood) / (2 * eps))
    check_seconds = perf_counter() - start
    error = abs(analytic - diff)
    if error > 1e-3 + 2e-6 * abs(analytic):
        raise AssertionError(f"Directional finite difference disagrees: {error}")
    return {"leaves": n, "edges": len(lengths), "sites": len(profile), "max_degree": layout.max_degree,
            "threads": threads, "binary": binary, "log_domain_sites": result.log_domain_sites,
            "zero_profile_entries_retained": int(np.count_nonzero(profile == 0)),
            "scope": "observed MSA on a supplied data-independent topological fixture and heterogeneous lengths",
            "likelihood_wall_seconds": likelihood_seconds, "likelihood_and_gradient_wall_seconds": gradient_seconds,
            "directional_check_wall_seconds": check_seconds,
            "directional_absolute_error": error,
            "directional_error_scaled": error / max(1., abs(analytic)),
            "total_wall_seconds": perf_counter() - mark,
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--prep", type=Path)
    parser.add_argument("--leaves", type=int, choices=[10000, 50000, 100000, 209644])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--binary", action="store_true")
    args = parser.parse_args()
    if args.check:
        command = build(args.library)
        data = checks(args.library)
        data["build_command"] = command
        with args.library.open("rb") as handle:
            data["library_sha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
    elif args.prep and args.leaves:
        data = observed_benchmark(args.prep, args.leaves, args.library, args.threads, args.binary)
    else:
        parser.error("Use --check or specify --prep and --leaves")
    print(json.dumps(data, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()

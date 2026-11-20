"""Independent exhaustive, randomized and floating-point boundary checks."""
from __future__ import annotations

import argparse
from fractions import Fraction
from itertools import product
import json
import math
from pathlib import Path
from time import perf_counter

import numpy as np

from .f81_pruning import Topology, build
from .f81_unrooted import UnrootedTopology, UnrootedPruning
from .f81_zero_boundary import ZeroBoundary, build_boundary


def rooted_fixture(n, rng):
    queue = list(range(n))
    parents = [-1] * n
    while len(queue) > 1:
        rng.shuffle(queue)
        left, right = queue.pop(), queue.pop()
        parent = len(parents)
        parents[left] = parents[right] = parent
        parents.append(-1)
        queue.append(parent)
    return Topology.from_parents(n, parents[:-1])


def abstract_data(rng, n):
    rows = rng.integers(0, 4, size=(n, 3), dtype=np.uint8)
    rows[1] = rows[0]
    pi = np.zeros((3, 21))
    pi[:, :4] = [0.15, 0.27, 0.33, 0.25]
    return rows, pi


def edge_coordinates(tree):
    pairs = np.empty((len(tree.layout.parent), 2), dtype=np.int32)
    pairs[tree.child_to_edge] = np.column_stack([np.arange(len(pairs)), tree.layout.parent])
    return pairs


def brute_components(n, pairs, rows, zero_mask):
    """Build a zero-edge graph and flood-fill leaves, comparing whole rows."""
    adj = [[] for _ in range(len(pairs)+1)]
    for a, b in pairs[zero_mask]:
        adj[a].append(b)
        adj[b].append(a)
    seen = set()
    for first in range(n):
        if first in seen:
            continue
        pending = [first]
        seen.add(first)
        while pending:
            node = pending.pop()
            if node < n and not np.array_equal(rows[node], rows[first]):
                return False
            for other in adj[node]:
                if other not in seen:
                    seen.add(other)
                    pending.append(other)
    return True


def brute_boundary(n, pairs, rows, x, d):
    permanent = (x == 0) & (d <= 0)
    times = {edge: Fraction(float(x[edge])) / -Fraction(float(d[edge]))
             for edge in np.flatnonzero((x > 0) & (d < 0))}
    expected = None
    changed_to_infeasible = False
    sweeps = 0
    for point in [Fraction(0), *sorted(set(times.values()))]:
        zero = permanent.copy()
        for edge, at in times.items():
            if at <= point:
                zero[edge] = True
        feasible = brute_components(n, pairs, rows, zero)
        if not feasible and expected is None:
            expected = point
        if changed_to_infeasible:
            assert not feasible
        changed_to_infeasible = changed_to_infeasible or not feasible
        sweeps += 1
    try:
        return math.inf if expected is None else float(expected), sweeps
    except OverflowError:
        return math.inf, sweeps


def check_ray(guard, rows, pairs, x, d):
    expected, sweeps = brute_boundary(len(rows), pairs, rows, x, d)
    measured = guard.first_conflict(x, d)
    assert measured == expected, (measured, expected)
    return measured, sweeps


def small_special(pruning, boundary):
    pairs = np.column_stack([np.arange(4), np.full(4, 4)])
    tree = UnrootedTopology.from_edges(4, pairs)
    rows = np.array([[0], [1], [0], [2]], dtype=np.uint8)
    pi = np.zeros((1, 21))
    pi[0, :3] = [0.2, 0.3, 0.5]
    engine = UnrootedPruning(tree, rows, pi, pruning)
    guard = ZeroBoundary(engine, boundary)
    # The two equal leaves may collapse at once. A conflicting third
    # event at the same time makes the joint boundary infeasible.
    point, direction = np.ones(4), np.array([-2., -2., -2., 0.])
    tie, _ = check_ray(guard, rows, pairs, point, direction)
    assert tie == .5 and not guard.feasible(np.maximum(0, point + tie * direction))
    allowed = np.array([0., 1., 0., 1.])
    assert guard.feasible(allowed) and math.isfinite(engine.evaluate(allowed, gradient=False).log_likelihood)
    # Edge 0 reopens for alpha > 0, so edge 1 can close at t=1 without
    # conflicting, provided edge 2 also leaves its initial zero component.
    d = np.array([1., -1., 1., 0.])
    opened, _ = check_ray(guard, rows, pairs, allowed, d)
    assert math.isinf(opened) and guard.feasible(np.maximum(0, allowed+d))
    closed, _ = check_ray(guard, rows, pairs, allowed, np.array([1., -1., 0., 0.]))
    assert closed == 1

    # At a subnormal length the conservative ray cap still rounds to an actual
    # zero. The separate exact feasibility check must reject this trial;
    # halving that same step then re-enters the feasible binary64 domain.
    tiny = math.nextafter(0., 1.)
    x = np.array([tiny, 0., 1., 1.])
    d = np.array([-tiny, 0., 0., 0.])
    limit, _ = check_ray(guard, rows, pairs, x, d)
    step = guard.bounded_step(limit)
    rounded = np.maximum(0, x + step*d)
    recovered = np.maximum(0, x + (step*.5)*d)
    assert guard.feasible(x) and not guard.feasible(rounded) and guard.feasible(recovered)
    assert engine.evaluate(rounded, gradient=False).log_likelihood == -math.inf
    assert math.isfinite(engine.evaluate(recovered, gradient=False).log_likelihood)
    return {"ties": True, "compatible_zero_edges": True, "conflict_after_existing_zeros": True,
            "existing_zero_edges_open": True, "binary64_checker_rejects_rounded_conflict": True,
            "halving_recovers_feasibility": True}


def extreme_rays(pruning, boundary, rng, runs):
    rows, pi = abstract_data(rng, 8)
    layout = rooted_fixture(8, rng)
    tree, _, _ = UnrootedTopology.from_rooted(layout, np.ones(len(layout.parent)))
    pairs = edge_coordinates(tree)
    engine = UnrootedPruning(tree, rows, pi, pruning)
    guard = ZeroBoundary(engine, boundary)
    sweeps = 0
    for i in range(runs):
        if i % 4 == 0:
            # Several nearly simultaneous hits, including ulp-scale differences.
            x = np.nextafter(np.ones(len(pairs)), rng.choice((0., 1., 2.), len(pairs)))
            d = -np.nextafter(x, rng.choice((0., 1., 2.), len(pairs)))
        else:
            x = np.ldexp(rng.uniform(.5, 1., len(pairs)), rng.integers(-1073, 1024, len(pairs)))
            d = -np.ldexp(rng.uniform(.5, 1., len(pairs)), rng.integers(-1073, 1024, len(pairs)))
        limit, scans = check_ray(guard, rows, pairs, x, d)
        sweeps += scans
        step = guard.bounded_step(limit)
        float_trial = np.maximum(0, x + step*d)
        if not guard.feasible(float_trial):
            # This is the actual integration safeguard: choose a smaller
            # step only when binary64 projection violates the exact graph.
            while not guard.feasible(float_trial):
                step *= 0.5
                float_trial = np.maximum(0, x + step*d)
        assert brute_components(len(rows), pairs, rows, float_trial == 0)
    return sweeps


def test_all(pruning, boundary, seed, random_cases, exhaustive):
    rng = np.random.default_rng(seed)
    mark = perf_counter()
    checked = sweeps = compared_likelihood = 0
    special = small_special(pruning, boundary)
    if exhaustive:
        for n in (4, 5, 6):
            rows, pi = abstract_data(rng, n)
            rooted = rooted_fixture(n, rng)
            tree, _, _ = UnrootedTopology.from_rooted(rooted, np.ones(len(rooted.parent)))
            pairs = edge_coordinates(tree)
            engine = UnrootedPruning(tree, rows, pi, pruning)
            guard = ZeroBoundary(engine, boundary)
            for mask in product((False, True), repeat=len(pairs)):
                zeros = np.array(mask)
                x = np.where(zeros, 0., .17)
                target = brute_components(n, pairs, rows, zeros)
                assert guard.feasible(x) == target
                assert math.isfinite(engine.evaluate(x, gradient=False).log_likelihood) == target
                checked += 1
                compared_likelihood += 1

    passes = failures = 0
    for _ in range(random_cases):
        n = int(rng.integers(7, 17))
        rows, pi = abstract_data(rng, n)
        rooted = rooted_fixture(n, rng)
        tree, _, _ = UnrootedTopology.from_rooted(rooted, np.ones(len(rooted.parent)))
        pairs = edge_coordinates(tree)
        engine = UnrootedPruning(tree, rows, pi, pruning)
        guard = ZeroBoundary(engine, boundary)
        x = rng.integers(0, 7, size=len(pairs)) / 8
        target = brute_components(n, pairs, rows, x == 0)
        assert guard.feasible(x) == target
        assert math.isfinite(engine.evaluate(x, gradient=False).log_likelihood) == target
        compared_likelihood += 1
        if not target:
            failures += 1
            x = x + 0.125  # Test ray starting from a feasible point.
        else:
            passes += 1
        direction = rng.integers(-4, 4, size=len(pairs)) / 4
        limit, scan = check_ray(guard, rows, pairs, x, direction)
        sweeps += scan
        step = guard.bounded_step(limit)
        trial = np.maximum(0, x + step*direction)
        assert guard.feasible(trial)
        assert math.isfinite(engine.evaluate(trial, gradient=False).log_likelihood)
        compared_likelihood += 1
    extremes = max(1, random_cases // 4)
    extreme_sweeps = extreme_rays(pruning, boundary, rng, extremes)
    return {"passed": True, "seed": seed, "exhaustive_masks": checked, "random_cases": random_cases,
            "floating_point_extreme_rays": extremes, "extreme_fraction_graph_sweeps": extreme_sweeps,
            "random_current_feasible": passes, "random_current_infeasible": failures,
            "exact_fraction_event_graph_sweeps": sweeps, "likelihood_agreement_checks": compared_likelihood,
            "edge_cases": special, "wall_seconds": perf_counter() - mark}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pruning-library", type=Path, required=True)
    parser.add_argument("--boundary-library", type=Path, required=True)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--seed", type=int, default=31121)
    parser.add_argument("--random-cases", type=int, default=2400)
    parser.add_argument("--skip-exhaustive", action="store_true")
    args = parser.parse_args()
    commands = [build(args.pruning_library), build_boundary(args.boundary_library)] if args.build else []
    result = test_all(args.pruning_library, args.boundary_library, args.seed, args.random_cases,
                      not args.skip_exhaustive)
    if commands:
        result["build_commands"] = commands
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

"""Numerical tree model checks on artificial categorical data only."""
from __future__ import annotations

import argparse
from decimal import Decimal, localcontext
import json
import math
from pathlib import Path
import resource
from time import perf_counter

import numpy as np


def transition(pi, length):
    """Rank-one finite-state transition in units of the refresh clock."""
    mass = -math.expm1(-length)
    return math.exp(-length) * np.eye(len(pi)) + mass * pi[None, :]


def profiles():
    # Integer mass makes the high-precision reference independently reproducible.
    return (
        tuple(range(1, 22)),
        (8, 0, 2, 0, 1, 13, 0, 1, 0, 1, 4, 1, 0, 3, 7, 0, 1, 1, 0, 4, 24),
        (0,) * 20 + (1,),
        (1, 49) + (0,) * 19,
    )


def matrix_reference(length, masses):
    with localcontext() as ctx:
        ctx.prec = 82
        stay = (-Decimal(length)).exp()
        den = Decimal(sum(masses))
        return np.array([[float((stay if a == b else Decimal(0)) + (1-stay) * Decimal(m) / den)
                          for b, m in enumerate(masses)] for a in range(len(masses))])


def check():
    matrix_max = stable_max = semi_max = rate_max = 0.0
    distinct_rates = []
    for counts in profiles():
        pi = np.asarray(counts) / sum(counts)
        nu = 1 - float(pi @ pi)
        distinct_rates.append(nu)
        for text in ("0", "0.000000000001", "0.03125", "1.5", "40", "800"):
            length = float(text)
            q = transition(pi, length)
            ref = matrix_reference(text, counts)
            matrix_max = max(matrix_max, float(abs(q - ref).max()))
            stable_max = max(stable_max, float(abs(q.sum(axis=1) - 1).max()),
                             float(abs(pi @ q - pi).max()),
                             float(abs(pi[:, None] * q - (pi[:, None] * q).T).max()))
            nonexistent = np.flatnonzero(pi == 0)
            for b in nonexistent:
                assert np.all(q[np.arange(len(pi)) != b, b] == 0)
                assert q[b, b] == math.exp(-length)
            gap_state = 20  # A regular categorical state in this fixture.
            assert abs(q[0, gap_state] - (-math.expm1(-length) * pi[gap_state])) < 2e-15
            assert np.all(q >= 0)
        base_length = 0.625
        semigroup = transition(pi, 0.375) @ transition(pi, 0.25)
        semi_max = max(semi_max, float(abs(semigroup - transition(pi, base_length)).max()))
        q = transition(pi, base_length)
        np.fill_diagonal(q, 0)
        endpoint_mismatch = float((pi @ q).sum())
        rate_max = max(rate_max, abs(endpoint_mismatch - nu * (-math.expm1(-base_length))))
        if nu > 0:
            for_rate_unit = transition(pi, (nu * base_length) / nu)
            assert np.max(abs(for_rate_unit - transition(pi, base_length))) < 2e-15
    assert np.array_equal(transition(np.asarray(profiles()[0]) / sum(profiles()[0]), 0.0), np.eye(21))
    assert max(matrix_max, stable_max, semi_max, rate_max) < 2e-15
    fixture = small_fixture()
    paths = ((0,1,2,3), (0,2,1,3), (0,3,1,2))
    correct = [quartet_loglike(fixture, path, .37, [.07,.11,.13,.17], .23) for path in paths]
    challenge = quartet_challenge()
    challenge_values = [quartet_loglike(challenge, path, .22, [.025]*4, .4) for path in paths]
    assert int(np.argmax(challenge_values)) == 0
    reference = quartet_decimal(fixture)
    assert abs(reference - correct[0]) < 2e-12
    alignment = np.asarray([v for _, v in fixture]).T
    pi = np.asarray([np.asarray(m, dtype=float) / sum(m) for m, _ in fixture])
    fixed_tree = fixed_tree_loglike(alignment, pi, .15)
    quartet_fixed = quartet_loglike(fixture, paths[0], .15, [.15]*4, .15)
    assert abs(fixed_tree - quartet_fixed) < 2e-12
    return {
        "profiles": len(distinct_rates),
        "states": 21,
        "matrix_entries_checked": len(distinct_rates) * 6 * 21 * 21,
        "maximum_matrix_error": matrix_max,
        "maximum_stochastic_stationary_reversibility_error": stable_max,
        "maximum_semigroup_error": semi_max,
        "maximum_endpoint_mismatch_error": rate_max,
        "stationary_expected_jumps_per_refresh_unit": distinct_rates,
        "quartet_numeric_log_likelihood": correct[0],
        "quartet_high_precision_log_likelihood": reference,
        "fixed_tree_vs_dense_pruning_error": abs(fixed_tree - quartet_fixed),
        "quartet_topology_challenge_log_likelihoods": challenge_values,
        "quartet_best_split": "01|23",
    }


def small_fixture():
    fixed = []
    for i in range(42):
        masses = profiles()[i % 2]
        support = [j for j, m in enumerate(masses) if m]
        values = [support[(i + k * k) % len(support)] for k in range(4)]
        fixed.append((masses, values))
    return fixed


def quartet_challenge():
    counts = (5,) * 21
    return [(counts, [j, j, (j + 4) % 21, (j + 4) % 21]) for j in range(21)]


def quartet_loglike(fixture, order, root_edge, pendant, second_root_edge, units=1.0):
    terms = []
    for masses, values in fixture:
        pi = np.asarray(masses) / sum(masses)
        leaves = [transition(pi, pendant[i] * units)[:, values[i]] for i in range(4)]
        left = leaves[order[0]] * leaves[order[1]]
        right = leaves[order[2]] * leaves[order[3]]
        likelihood = pi @ ((transition(pi, root_edge * units) @ left)
                           * (transition(pi, second_root_edge * units) @ right))
        terms.append(math.log(likelihood))
    return math.fsum(terms)


def quartet_decimal(fixture):
    with localcontext() as ctx:
        ctx.prec = 64

        def q(pi, length, a, b):
            retain = (-Decimal(length)).exp()
            return (retain if a == b else Decimal(0)) + (1 - retain) * pi[b]

        root_term = Decimal(0)
        for masses, values in fixture:
            pi = [Decimal(m) / sum(masses) for m in masses]
            left = [q(pi, ".07", a, values[0]) * q(pi, ".11", a, values[1]) for a in range(21)]
            right = [q(pi, ".13", b, values[2]) * q(pi, ".17", b, values[3]) for b in range(21)]
            site = sum(pi[r] * sum(q(pi, ".37", r, a) * left[a] for a in range(21))
                       * sum(q(pi, ".23", r, b) * right[b] for b in range(21)) for r in range(21))
            root_term += site.ln()
        return float(root_term)


def fixed_tree_loglike(rows, pi, length):
    """Evaluate one artificial balanced topology, with scaling after each split."""
    stay = math.exp(-length)
    change = -math.expm1(-length)
    partial = np.eye(pi.shape[1])[rows]
    scale_log = 0.0
    while len(partial) > 1:
        pairs = len(partial) // 2
        # Rank-one multiply Q v along one edge, then multiply child likelihoods.
        child = partial[:2 * pairs]
        average = np.einsum("nsa,sa->ns", child, pi, optimize=False)
        message = stay * child + change * average[:, :, None]
        parent = message[::2] * message[1::2]
        scale = parent.sum(axis=2)
        if np.any(scale <= 0):
            raise ValueError("Impossible artificial site under its assigned profile")
        scale_log += float(np.log(scale).sum())
        parent /= scale[:, :, None]
        partial = np.concatenate((parent, partial[-1:])) if len(partial) % 2 else parent
    return scale_log + float(np.log((partial[0] * pi).sum(axis=1)).sum())


def scale_check(leaves):
    rows = (3 * np.arange(leaves, dtype=np.int64)[:, None] + np.arange(263)[None, :]
            + (np.arange(leaves, dtype=np.int64)[:, None] // 7)) % 21
    profiles_array = np.tile(np.asarray(profiles()[0], dtype=float), (263, 1))
    for i in range(len(profiles_array)):
        profiles_array[i] = np.roll(profiles_array[i], i % 21)
    pi = profiles_array / profiles_array.sum(axis=1)[:, None]
    start = perf_counter()
    values = []
    for length in (.03, .08, .16, .32, .64):
        before = perf_counter()
        value = fixed_tree_loglike(rows, pi, length)
        values.append({"effective_length": length, "log_likelihood": value, "seconds": perf_counter() - before})
    assert all(math.isfinite(x["log_likelihood"]) for x in values)
    return {"leaves": leaves, "sites": 263, "scope": "artificial fixed topology, five effective lengths",
            "values": values, "seconds": perf_counter() - start,
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def fixtures(output):
    output.mkdir()
    fixture = small_fixture()
    alphabet = "0123456789ABCDEFGHIJK"
    all_morph = ["".join(alphabet[row[i]] for _,row in fixture) for i in range(4)]
    assert set("".join(all_morph)) == set(alphabet)
    (output/"categories.phy").write_text(f"4 {len(fixture)}\n" + "".join(
        f"r{j} {row}\n" for j,row in enumerate(all_morph)))
    (output/"toy.tree").write_text("((r0:0.07,r1:0.11):0.37,(r2:0.13,r3:0.17):0.23);\n")
    (output/"profiles.txt").write_text("".join(
        f"{i+1} " + " ".join(f"{m/sum(masses):.17g}" for m in masses) + "\n"
        for i,(masses,_) in enumerate(fixture)))
    average_rate = sum(1 - sum((x/sum(masses))**2 for x in masses) for masses,_ in fixture) / len(fixture)
    return {"raw_effective_length_log_likelihood": quartet_decimal(fixture),
            "unit_rate_branch_as_input_log_likelihood": quartet_loglike(
                fixture, (0,1,2,3), .37, [.07,.11,.13,.17], .23, 1 / average_rate),
            "mean_stationary_rate": average_rate}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true")
    action.add_argument("--scale", type=int, choices=(1000, 5000, 10000))
    action.add_argument("--fixture-dir", type=Path)
    args = parser.parse_args()
    result = check() if args.check else scale_check(args.scale) if args.scale else fixtures(args.fixture_dir)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

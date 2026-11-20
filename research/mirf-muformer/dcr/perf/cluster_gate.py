"""Check the preregistered cluster information gates before any A fits."""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import subprocess
from time import perf_counter

import numpy as np

import potts_fit


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def build(output: Path):
    source = Path(__file__).with_suffix(".c")
    library = output / "cluster_gate.so"
    command = ["gcc", "-O3", "-std=c11", "-Wall", "-Wextra", "-Werror", "-mavx2",
               "-mpopcnt", "-fopenmp", "-shared", "-fPIC", str(source), "-o", str(library)]
    subprocess.run(command, check=True)
    return library, command


def scan(matrix, degree, weights, library, threads):
    n = len(matrix)
    if matrix.shape != (n, 263) or not n or n >= (1 << 31) or np.any(degree < 0):
        raise ValueError("Invalid input shape or reference degrees")
    if degree.shape != (n,) or weights.shape != (n,) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Invalid degree or weight vector")
    diversity = [
        1 - float(np.square(np.bincount(matrix[:, i], minlength=21) / n).sum())
        for i in range(263)
    ]
    ordered = np.ascontiguousarray(matrix[:, np.argsort(-np.asarray(diversity), kind="stable")], dtype=np.uint8)
    offset = np.concatenate(([0], np.cumsum(degree, dtype=np.int64)))
    labels = [np.empty(n, dtype=np.uint32) for _ in range(2)]
    masses = [np.zeros(n, dtype=np.float64) for _ in range(2)]
    totals = [ctypes.c_int64() for _ in range(2)]
    kernel = ctypes.CDLL(str(library)).cluster_scan
    kernel.argtypes = [ctypes.c_void_p, ctypes.c_uint32] + [ctypes.c_void_p] * 6 + [
        ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.c_int
    ]
    kernel.restype = ctypes.c_int
    ptr = lambda a: a.ctypes.data
    status = kernel(ptr(ordered), n, ptr(offset), ptr(weights), *map(ptr, labels),
                    *map(ptr, masses), *map(ctypes.byref, totals), threads)
    if status:
        raise RuntimeError(f"Native scan rejected its allocation or reference degrees: {status}")
    return labels, masses, [t.value for t in totals]


def summarize(labels, edge_mass, weights):
    groups, inverse = np.unique(labels, return_inverse=True)
    sizes = np.bincount(inverse)
    weight = np.bincount(inverse, weights=weights)
    sum_sq = np.bincount(inverse, weights=weights * weights)
    possible = (weight * weight - sum_sq) / 2
    larger = sizes > 1
    chain = 1 - edge_mass[groups][larger] / possible[larger]
    if np.any(chain < -1e-9) or np.any(chain > 1 + 1e-9):
        raise ArithmeticError("Within-component edge mass exceeds possible mass")
    chain = np.clip(chain, 0, 1)
    shares = weight / float(weights.sum())
    biggest = int(np.argmax(weight))
    heavy_possible = possible[biggest]
    heavy_chain = (float(np.clip(1 - edge_mass[groups[biggest]] / heavy_possible, 0, 1))
                   if sizes[biggest] > 1 else None)
    largest_rows = int(sizes.max()) / len(labels)
    largest_weight = float(shares[biggest])
    effective = 1 / float(shares @ shares)
    reasons = []
    if largest_rows >= 0.50:
        reasons.append("largest_component_row_share_at_least_0.50")
    if largest_weight >= 0.50:
        reasons.append("largest_component_weight_share_at_least_0.50")
    if effective < 4:
        reasons.append("weight_effective_components_below_4")
    if heavy_chain is not None and heavy_chain > 0.50:
        reasons.append("heaviest_component_chain_fraction_above_0.50")
    return {
        "components": len(groups),
        "singleton_components": int(np.sum(sizes == 1)),
        "maximum_rows": int(sizes.max()),
        "maximum_row_share": largest_rows,
        "largest_weight_share": largest_weight,
        "effective_weight_components": effective,
        "heaviest_component_rows": int(sizes[biggest]),
        "heaviest_component_chain_fraction": heavy_chain,
        "non_singleton_chain_quantiles": [float(x) for x in np.quantile(chain, [0, .25, .50, .75, 1])] if chain.size else None,
        "eligible": not reasons,
        "failed_gates": reasons,
    }


def self_check(library):
    import itertools
    rng = np.random.default_rng(20261120)
    # Three markers differ in disjoint segments: exact thresholds, chained
    # endpoints, duplicate rows, and a disconnected row.
    matrix = np.zeros((6, 263), dtype=np.uint8)
    matrix[1, :26] = 1
    matrix[2, :52] = 1
    matrix[3, :53] = 1
    matrix[4] = 2
    matrix[5] = matrix[2]
    permutation = rng.permutation(len(matrix))
    matrix = matrix[permutation]
    dist = (matrix[:, None] != matrix[None, :]).sum(axis=2)
    result, masses, total = scan(matrix, (dist <= 52).sum(axis=1) - 1, np.ones(len(matrix)), library, 2)
    for limit, labels, mass, count in zip((52, 26), result, masses, total):
        adjacency = dist <= limit
        reachable = adjacency.copy()
        for k in range(len(matrix)):
            reachable |= reachable[:, k:k+1] & reachable[k:k+1, :]
        np.testing.assert_array_equal(labels, reachable.argmax(axis=1))
        assert count == int(np.triu(adjacency, k=1).sum())
        expected_mass = np.zeros(len(matrix))
        for i, j in itertools.combinations(range(len(matrix)), 2):
            if adjacency[i, j]:
                expected_mass[labels[i]] += 1
        np.testing.assert_array_equal(mass, expected_mass)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prep", required=True, type=Path)
    parser.add_argument("--scratch", required=True, type=Path)
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=32)
    args = parser.parse_args()
    args.scratch.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    library, build_command = build(args.scratch)
    self_check(library)
    matrix, weights, total = potts_fit.load_official_alignment(args.prep)
    expected = np.loadtxt(args.prep / "neighbor_counts.tsv", usecols=1, skiprows=1, dtype=np.int64)
    print(f"Brute-force cross-check passed. Scanning all pairs of {len(matrix):,} archived rows.", flush=True)
    labels, edge_masses, n_pairs = scan(matrix, expected - 1, weights, library, args.threads)
    reference = json.loads((args.prep / "neighbor_summary.json").read_text())
    if n_pairs[0] != reference["neighbor_pairs"]:
        raise ValueError("Main threshold did not reproduce the frozen pair count")
    data = {
        "rows": len(matrix),
        "sites": matrix.shape[1],
        "official_weight_total": total,
        "sha256": {name: sha256(args.prep / name) for name in ("mature_codes.npy", "neighbor_counts.tsv")},
        "build_command": build_command,
        "library_sha256": sha256(library),
        "threads": args.threads,
        "graph_seconds": perf_counter() - start,
        "main_degree_vector_matches_reference": True,
        "granularities": {},
    }
    for threshold, groups, masses, edges in zip((211, 237), labels, edge_masses, n_pairs):
        np.save(args.scratch / f"clusters_{threshold}.npy", groups, allow_pickle=False)
        attrs = summarize(groups, masses, weights)
        data["granularities"][str(threshold)] = {
            "edges": edges,
            "cluster_assignments_sha256": sha256(args.scratch / f"clusters_{threshold}.npy"),
            **attrs,
            "planned": [
                {
                    "id": f"A{threshold}-{i:03d}",
                    "role": "pilot_and_screening" if i == 1 else "screening",
                    "state": "eligible_not_submitted" if attrs["eligible"] else "blocked_before_fitting",
                }
                for i in range(1, 5)
            ],
        }
    with args.ledger.open("x", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(data, indent=2), flush=True)


if __name__ == "__main__":
    main()

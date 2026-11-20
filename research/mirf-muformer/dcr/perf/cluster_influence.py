"""Audit weights and chaining in every component of both observed partitions."""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
from pathlib import Path
from time import perf_counter

import numpy as np

import potts_fit
from .cluster_gate import build, scan, sha256


def summarize(labels, edges, weights):
    groups, inverse = np.unique(labels, return_inverse=True)
    sizes = np.bincount(inverse)
    mass = np.bincount(inverse, weights=weights)
    sum_sq = np.bincount(inverse, weights=weights * weights)
    possible = (mass * mass - sum_sq) / 2
    low = possible - edges[groups]
    if np.any(low < -1e-9 * np.maximum(1, possible)):
        raise ArithmeticError("Edge masses exceed all within-component pair mass")
    low = np.maximum(0, low)
    low[sizes == 1] = 0
    ratio = np.divide(low, possible, out=np.zeros_like(low), where=sizes > 1)
    shares = mass / weights.sum()
    concentration = shares * shares
    concentration /= concentration.sum()
    ranked = np.lexsort((groups, -shares))
    products = zip(sizes[ranked], shares[ranked], low[ranked],
                   possible[ranked], ratio[ranked], concentration[ranked])
    return groups, ranked, products, {
        "clusters": len(groups),
        "positive_pair_clusters": int(np.sum(sizes > 1)),
        "chained_clusters_ratio_above_0.50": int(np.sum(ratio > 0.50)),
        "chained_cluster_weight_share": float(shares[ratio > 0.50].sum()),
        "chained_clusters_concentration_share": float(concentration[ratio > 0.50].sum()),
        "low_similarity_mass_share_of_within_cluster_pairs": float(low.sum() / possible.sum()),
        "heaviest_cluster_weight_share": float(shares[ranked[0]]),
        "heaviest_cluster_concentration_share": float(concentration[ranked[0]]),
        "heaviest_cluster_chain_fraction": float(ratio[ranked[0]]),
        "heaviest_cluster_low_similarity_pair_mass": float(low[ranked[0]]),
        "heaviest_cluster_total_pair_mass": float(possible[ranked[0]]),
        "heaviest_cluster_weight_to_average": float(shares[ranked[0]] * len(groups)),
    }


def diameter_contrast(cluster_rows, weights, threshold):
    """First fit in fixed observed-row order, subject to a pairwise diameter."""
    size = len(cluster_rows)
    admissible = np.empty((size, size), dtype=bool)
    for start in range(0, size, 32):
        block = slice(start, start + 32)
        admissible[block] = ((cluster_rows[block, None, :] == cluster_rows[None, :, :])
                             .sum(axis=2) >= threshold)
    groups = []
    for row in range(size):
        for members in groups:
            if admissible[row, members].all():
                members.append(row)
                break
        else:
            groups.append([row])
    for members in groups:
        assert admissible[np.ix_(members, members)].all()
    masses = [float(weights[members].sum()) for members in groups]
    return {
        "scope": "only the heaviest original component; no fits",
        "rule": "first compatible group in original row order, all within-group pairs pass",
        "diameter_bound_mismatches": 263 - threshold,
        "subclusters": len(groups),
        "largest_subcluster_rows": max(map(len, groups)),
        "largest_subcluster_share_of_parent_mass": max(masses) / sum(masses),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prep", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--assignments", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    start = perf_counter()
    gate = json.loads(args.gate.read_text())
    if any(sha256(args.prep / n) != h for n, h in gate["sha256"].items()):
        raise ValueError("Alignment or weights differ from validated input")
    matrix, weights, total = potts_fit.load_official_alignment(args.prep)
    counts = np.loadtxt(args.prep / "neighbor_counts.tsv", skiprows=1, usecols=1, dtype=np.int64)
    args.output.mkdir()
    args.scratch.mkdir()
    library, _ = build(args.scratch)
    labels, within_mass, pairs = scan(matrix, counts - 1, weights, library, args.threads)

    path = args.output / "all-cluster-influence.csv.gz"
    results = {}
    with path.open("xb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", filename="", mtime=0) as compressed:
        with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
            csv_writer = csv.writer(text, lineterminator="\n")
            csv_writer.writerow(("threshold", "weight_rank", "rows", "weight_share", "low_similarity_pair_mass",
                                 "within_cluster_pair_mass", "chain_fraction", "concentration_contribution"))
            for threshold, labels_at_threshold, mass, edges in zip((211, 237), labels, within_mass, pairs):
                entry = gate["granularities"][str(threshold)]
                recorded = args.assignments / f"clusters_{threshold}.npy"
                if (sha256(recorded) != entry["cluster_assignments_sha256"]
                    or not np.array_equal(labels_at_threshold, np.load(recorded, allow_pickle=False))
                    or edges != entry["edges"]):
                    raise ValueError("Graph differs from frozen membership")
                groups, ranked, records, result = summarize(labels_at_threshold, mass, weights)
                for rank, record in enumerate(records, 1):
                    size, share, low, possible, ratio, contribution = record
                    csv_writer.writerow((threshold, rank, int(size), share, low, possible,
                                         ratio if size > 1 else "", contribution))
                keep = labels_at_threshold == groups[ranked[0]]
                result["heaviest_diameter_contrast"] = diameter_contrast(matrix[keep], weights[keep], threshold)
                results[str(threshold)] = result
    report = {"scope": "complete component diagnostics, weights from frozen observed data",
              "rows": len(matrix), "total_weight": total, "threads": args.threads,
              "granularities": results, "csv_sha256": sha256(path), "seconds": perf_counter() - start}
    with (args.output / "global-influence.json").open("x") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()

"""Per-edge projected residuals of recovered Stage 2.8 checkpoints.

The residual is the existing bound projection
r = s - max(0, s - g). No other gradient norm is substituted for it.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .check_f81_joint import data
from .f81_joint_optimize import chart
from .f81_optimize import projected_norm
from .f81_zero_boundary import ZeroBoundary


def edge_table(engine):
    tree = engine.topology
    pairs = np.empty((len(tree.child_to_edge), 2), dtype=np.int32)
    pairs[tree.child_to_edge] = np.column_stack((np.arange(len(tree.layout.parent)), tree.layout.parent))
    return pairs


def summarize(lengths, gradient, diagonal, pairs):
    residual = lengths - np.maximum(0., lengths - gradient)
    positive, active, mixed, mixed_diagonal = chart(lengths, gradient, diagonal)
    absolute = np.abs(residual)
    square = residual * residual
    total_square = float(square.sum())
    order = np.argsort(-absolute)
    n = len(lengths)
    def share(fraction):
        count = max(1, int(np.ceil(fraction * n)))
        chosen = order[:count]
        return {"edges": count, "squared_residual_share": float(square[chosen].sum() / total_square)}
    thresholds = {}
    for gate in (1e-5, 1e-4, 1e-3, 1e-2):
        mask = absolute > gate
        thresholds[f"{gate:.0e}"] = {
            "edges": int(mask.sum()),
            "positive_edges": int((mask & positive).sum()),
            "zero_edges": int((mask & ~positive).sum()),
        }
    def group(mask):
        return {
            "edges": int(mask.sum()),
            "squared_residual": float(square[mask].sum()),
            "squared_residual_share": float(square[mask].sum() / total_square) if total_square else 0.,
            "max_abs_residual": float(absolute[mask].max()) if np.any(mask) else 0.,
        }
    top1 = order[:max(1, int(np.ceil(0.01 * n)))]
    parents, children = pairs[:, 1], pairs[:, 0]
    node_hits = {}
    for edge in top1:
        for node in (int(parents[edge]), int(children[edge])):
            node_hits[node] = node_hits.get(node, 0) + 1
    ranked_nodes = sorted(node_hits.items(), key=lambda item: item[1], reverse=True)
    peak = int(order[0])
    neighbors = []
    peak_nodes = {int(parents[peak]), int(children[peak])}
    for edge, (child, parent) in enumerate(pairs.tolist()):
        if edge != peak and ({child, parent} & peak_nodes):
            neighbors.append(int(edge))
    return {
        "projection_infinity_norm": projected_norm(lengths, gradient),
        "absolute_residual": {
            "max": float(absolute.max()),
            "median": float(np.median(absolute)),
            "p90": float(np.quantile(absolute, 0.90)),
            "p95": float(np.quantile(absolute, 0.95)),
            "p99": float(np.quantile(absolute, 0.99)),
        },
        "edges_above": thresholds,
        "by_sign": {"positive": group(positive), "zero": group(~positive)},
        "squared_residual_concentration": {"top_1_percent": share(0.01), "top_5_percent": share(0.05), "top_10_percent": share(0.10)},
        "top_edges": [int(edge) for edge in order[:100]],
        "top_1_percent_edges": [int(edge) for edge in top1],
        "peak_edge": {
            "edge_id": peak,
            "parent_id": int(parents[peak]),
            "child_id": int(children[peak]),
            "length": float(lengths[peak]),
            "physical_gradient": float(gradient[peak]),
            "projection_residual": float(residual[peak]),
            "hessian_diagonal": float(diagonal[peak]),
            "mixed_gradient": float(mixed[peak]),
            "incident_edges": neighbors,
        },
        "top_1_percent_node_hits": [{"node": node, "edges": count} for node, count in ranked_nodes[:10]],
        "top_1_percent_edges_touching_top_5_nodes": int(sum(count for _, count in ranked_nodes[:5])),
        "columns": {
            "residual": residual, "mixed": mixed, "mixed_diagonal": mixed_diagonal,
            "positive": positive, "active": active,
        },
    }


def write_table(path, pairs, lengths, gradient, diagonal, summary):
    columns = summary["columns"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["edge_id", "parent_id", "child_id", "length", "physical_gradient",
                         "projection_residual", "hessian_diagonal", "zero_edge", "active_boundary",
                         "mixed_gradient"])
        for edge, (child, parent) in enumerate(pairs.tolist()):
            writer.writerow([
                edge, int(parent), int(child), float(lengths[edge]), float(gradient[edge]),
                float(columns["residual"][edge]), float(diagonal[edge]),
                int(lengths[edge] == 0), int(columns["active"][edge]), float(columns["mixed"][edge]),
            ])


def audit_one(engine, guard, path):
    saved = np.load(path)
    lengths = np.ascontiguousarray(saved["lengths"], dtype=np.float64)
    fresh = engine.evaluate(lengths, threads=4, gradient=True, diagonal=True)
    gradient = -fresh.length_gradient / len(fresh.site_log_likelihood)
    diagonal = fresh.negative_log_hessian_diagonal / len(fresh.site_log_likelihood)
    objective = -fresh.log_likelihood / len(fresh.site_log_likelihood)
    summary = summarize(lengths, gradient, diagonal, edge_table(engine))
    report = {key: value for key, value in summary.items() if key != "columns"}
    report.update({
        "checkpoint": str(path),
        "lengths_sha256": hashlib.sha256(lengths.tobytes()).hexdigest(),
        "saved_gradient_max_abs_difference": float(np.max(np.abs(gradient - saved["physical_gradient"]))),
        "saved_diagonal_max_abs_difference": float(np.max(np.abs(diagonal - saved["hessian_diagonal"]))),
        "recomputed_objective": objective,
        "saved_objective": float(saved["objective"]),
        "feasible": bool(guard.feasible(lengths)),
        "zero_edges": int(np.count_nonzero(lengths == 0)),
    })
    table = path.with_suffix(".tsv")
    write_table(table, edge_table(engine), lengths, gradient, diagonal, summary)
    report["table"] = str(table)
    report["table_sha256"] = hashlib.sha256(table.read_bytes()).hexdigest()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--case", default="previous-10000")
    for flag in ("prep", "gate", "library", "checkpoint", "stage26", "boundary"):
        parser.add_argument("--" + flag, type=Path, required=True)
    args = parser.parse_args()
    engine, _auditor, _point, facts, _digest = data(args)
    guard = ZeroBoundary(engine, args.boundary)
    reports = [audit_one(engine, guard, path) for path in args.checkpoints]
    payload = {**facts, "audits": reports}
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "projections": [item["projection_infinity_norm"] for item in reports]}, indent=2))


if __name__ == "__main__":
    main()

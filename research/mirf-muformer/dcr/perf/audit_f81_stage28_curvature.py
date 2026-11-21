"""Check the stored diagonal against an independent split score and finite differences.

Step lengths scale with each branch. A positive branch is never stepped across zero.
A zero branch is tested from one side only, and only when that step stays feasible.
"""
import argparse
import json
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np

from .check_f81_joint import data
from .check_f81_split_gradient import Reference
from .f81_joint_optimize import chart
from .f81_zero_boundary import ZeroBoundary


def step_for(length):
    """Return a centered step that stays on the same side of zero."""
    scale = max(abs(float(length)), 1e-6)
    proposal = max(1e-8, min(1e-5, 1e-4 * scale))
    return min(proposal, 0.2 * float(length))


def gradient_of(engine, lengths):
    item = engine.evaluate(lengths, threads=4, gradient=True, diagonal=False)
    sites = len(item.site_log_likelihood)
    return -item.length_gradient / sites, -item.log_likelihood / sites, sites


def centered_diagonal(engine, lengths, edge, step):
    plus, minus = lengths.copy(), lengths.copy()
    plus[edge] += step
    minus[edge] -= step
    right, _, _ = gradient_of(engine, plus)
    left, _, _ = gradient_of(engine, minus)
    return float((right[edge] - left[edge]) / (2 * step))


def onesided_diagonal(engine, lengths, edge, step):
    plus = lengths.copy()
    plus[edge] = step
    right, _, _ = gradient_of(engine, plus)
    base, _, _ = gradient_of(engine, lengths)
    return float((right[edge] - base[edge]) / step)


def block_hessian(engine, lengths, edges):
    base, _, _ = gradient_of(engine, lengths)
    size = len(edges)
    matrix = np.zeros((size, size))
    steps = []
    for column, edge in enumerate(edges):
        step = step_for(lengths[edge]) if lengths[edge] > 0 else max(1e-8, 1e-4)
        steps.append(step)
        if lengths[edge] > 0:
            plus, minus = lengths.copy(), lengths.copy()
            plus[edge] += step
            minus[edge] -= step
            right, _, _ = gradient_of(engine, plus)
            left, _, _ = gradient_of(engine, minus)
            matrix[:, column] = (right[list(edges)] - left[list(edges)]) / (2 * step)
        else:
            plus = lengths.copy()
            plus[edge] = step
            right, _, _ = gradient_of(engine, plus)
            matrix[:, column] = (right[list(edges)] - base[list(edges)]) / step
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    span = max(abs(float(eigenvalues[-1])), 1e-30)
    return {
        "edges": [int(edge) for edge in edges],
        "lengths": [float(lengths[edge]) for edge in edges],
        "steps": steps,
        "matrix": matrix.tolist(),
        "symmetry_max_abs": float(np.max(np.abs(matrix - matrix.T))),
        "eigenvalues": eigenvalues.tolist(),
        "condition_estimate": float(np.linalg.cond(symmetric)),
        "negative_curvature": bool(eigenvalues[0] < -1e-8 * span),
    }


def select_edges(lengths, residual_report, count):
    absolute = np.abs(lengths - np.maximum(0., lengths - residual_report))
    # residual_report here is unused; selections come from the JSON plus local draws.
    del absolute
    rng = np.random.default_rng(20261121)
    positive = np.flatnonzero(lengths > 0)
    near = positive[np.argsort(lengths[positive])[:8]]
    drawn = rng.choice(positive, size=count, replace=False)
    zeros = np.flatnonzero(lengths == 0)
    zero_draw = zeros if len(zeros) <= 8 else rng.choice(zeros, size=8, replace=False)
    return np.unique(drawn), near, zero_draw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-npz", type=Path, required=True)
    parser.add_argument("--residual-json", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--case", default="previous-10000")
    for flag in ("prep", "gate", "library", "checkpoint", "stage26", "boundary", "reference"):
        parser.add_argument("--" + flag, type=Path, required=True)
    args = parser.parse_args()
    engine, _auditor, _point, facts, _digest = data(args)
    guard = ZeroBoundary(engine, args.boundary)
    saved = np.load(args.checkpoint_npz)
    lengths = np.ascontiguousarray(saved["lengths"], dtype=np.float64)
    item = engine.evaluate(lengths, threads=4, gradient=True, diagonal=True)
    sites = len(item.site_log_likelihood)
    gradient = -item.length_gradient / sites
    diagonal = item.negative_log_hessian_diagonal / sites
    positive, active, mixed, mixed_diagonal = chart(lengths, gradient, diagonal)
    analytic_mixed = np.where(lengths > 0, lengths * gradient, gradient)
    analytic_mixed[active] = 0.
    factor = np.where(lengths > 0, lengths, 1.)
    analytic_mixed_diagonal = (diagonal * factor) * factor + np.where(lengths > 0, gradient * factor, 0.)
    analytic_mixed_diagonal[active] = diagonal[active]
    residual = json.loads(args.residual_json.read_text(encoding="utf-8"))
    audit = next(row for row in residual["audits"] if Path(row["checkpoint"]).name == args.checkpoint_npz.name)
    top = audit["top_edges"][:10]
    random_positive, near_zero, zero_edges = select_edges(lengths, gradient, 20)
    split = Reference(engine, lengths, args.reference)
    split_rows = []
    with localcontext() as context:
        context.prec = 80
        for edge in top:
            reference = split.score(int(edge), 256)
            error = abs(Decimal.from_float(float(gradient[edge])) - reference)
            split_rows.append({"edge": int(edge), "length": float(lengths[edge]),
                               "physical_gradient": float(gradient[edge]), "split_256": str(reference),
                               "absolute_difference": str(error)})
    finite = []
    for edge in list(random_positive) + list(near_zero):
        step = step_for(lengths[edge])
        estimate = centered_diagonal(engine, lengths, int(edge), step)
        half = centered_diagonal(engine, lengths, int(edge), step / 2)
        extrapolated = (4 * half - estimate) / 3
        finite.append({"edge": int(edge), "kind": "positive", "length": float(lengths[edge]), "step": step,
                       "stored_diagonal": float(diagonal[edge]), "difference_h": estimate,
                       "difference_h_half": half, "extrapolated": extrapolated,
                       "relative_error": abs(extrapolated - float(diagonal[edge])) / max(1., abs(float(diagonal[edge])))})
    zero_rows = []
    for edge in zero_edges:
        step = 1e-6
        trial = lengths.copy()
        trial[edge] = step
        if not guard.feasible(trial):
            zero_rows.append({"edge": int(edge), "status": "infeasible_step"})
            continue
        estimate = onesided_diagonal(engine, lengths, int(edge), step)
        zero_rows.append({"edge": int(edge), "status": "checked", "step": step,
                          "stored_diagonal": float(diagonal[edge]), "onesided": estimate,
                          "relative_error": abs(estimate - float(diagonal[edge])) / max(1., abs(float(diagonal[edge])))})
    peak = audit["peak_edge"]
    block_edges = [peak["edge_id"], *peak["incident_edges"][:2]]
    report = {
        **{key: facts[key] for key in ("indices_sha256", "reference_profile_sha256", "topology_sha256", "library_sha256")},
        "checkpoint": str(args.checkpoint_npz),
        "mixed_gradient_transform_max_abs": float(np.max(np.abs(mixed - analytic_mixed))),
        "mixed_diagonal_transform_max_abs": float(np.max(np.abs(mixed_diagonal - analytic_mixed_diagonal))),
        "split_top10": split_rows,
        "positive_finite_differences": finite,
        "zero_one_sided": zero_rows,
        "local_block": block_hessian(engine, lengths, block_edges),
        "active_boundary_edges": int(active.sum()),
        "positive_edges": int(positive.sum()),
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), "split_max": max(float(row["absolute_difference"]) for row in split_rows),
                      "block_condition": report["local_block"]["condition_estimate"],
                      "negative_curvature": report["local_block"]["negative_curvature"]}, indent=2))


if __name__ == "__main__":
    main()

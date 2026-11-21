"""Reproducible Stage 2.7 checks on frozen fixed-topology checkpoints."""
import argparse
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .check_f81_edge_conditioning import TangentReference
from .check_f81_observed_bridge import load_case
from .check_f81_recoverability import PATTERNS, TRUTHS, engine_for, law
from .check_f81_split_gradient import Reference
from .f81_joint_optimize import chart, solve
from .f81_zero_boundary import ZeroBoundary


def mathematics(library):
    results = {}
    for name, point in TRUTHS.items():
        mass, first, second = law(point, 2)
        scores = -first/mass[:, None]
        diagonal = scores*scores - np.diagonal(second, axis1=1, axis2=2)/mass[:, None]
        results[name] = {}
        for logged in (False, True):
            g, h, hz = [], [], []
            for i, pattern in enumerate(PATTERNS):
                entry = engine_for(pattern[None, :], library).evaluate(point, diagonal=True, log_domain=logged)
                ng = -entry.length_gradient
                nh = entry.negative_log_hessian_diagonal
                _, _, _, zh = chart(point, ng, nh)
                g.append(ng)
                h.append(nh)
                hz.append(zh)
            factor = np.where(point > 0, point, 1.)
            expected = diagonal * factor**2 + np.where(point > 0, scores*factor, 0.)
            error = {"gradient": float(np.abs(g-scores).max()),
                     "physical_diagonal": float(np.abs(h-diagonal).max()),
                     "mixed_diagonal": float(np.abs(hz-expected).max())}
            assert max(error.values()) < 1e-10
            results[name]["log" if logged else "scaled"] = error
    return {"results": results, "passed": True}


def data(args):
    engine, auditor, facts = load_case(args.prep, args.gate, 10000, args.library)
    earlier = json.loads(args.stage26.read_text())["cases"][args.case]
    for key in ("alignment_weights_sha256", "indices_sha256", "reference_profile_sha256", "topology_sha256"):
        assert facts[key] == earlier[key]
    point = np.load(args.checkpoint, allow_pickle=False)
    digest = hashlib.sha256(point.tobytes()).hexdigest()
    assert digest == earlier["terminal_sha256"]
    return engine, auditor, point, facts, digest


def block(engine, auditor, point, args):
    from decimal import localcontext
    edges = (1310, 10655, 15327)
    ref = TangentReference(auditor, args.dual)
    guard = ZeroBoundary(engine, args.boundary)
    matrix = []
    count = 0
    with localcontext() as context:
        context.prec = 115

        def at(p):
            nonlocal count
            assert guard.feasible(p)
            count += len(edges)
            return [ref.score(p, e, 256) for e in edges]

        start = at(point)
        estimates = []
        for h in (args.block_step, args.block_step/2):
            columns = []
            for e in edges:
                plus, double = point.copy(), point.copy()
                base = Decimal.from_float(point[e])
                plus[e], double[e] = float(base+h), float(base+2*h)
                x = Decimal.from_float(plus[e])-base
                y = Decimal.from_float(double[e])-base
                first, second = at(plus), at(double)
                columns.append([float((y*y*(g-g0)-x*x*(gg-g0))/(x*y*(y-x)))
                                for g0, g, gg in zip(start, first, second)])
            estimates.append(np.array(columns).T)
        matrix = (4*estimates[1]-estimates[0])/3
        exact = Reference(engine, point, args.reference)
        diag = [float(sum(row[4] for row in exact.site_terms(e, 256)[1])/len(engine.pruning.states))
                for e in edges]
    eig = np.linalg.eigvalsh((matrix+matrix.T)/2)
    noise = float(max(np.max(np.abs(estimates[1]-matrix)), np.max(np.abs(matrix-matrix.T))))
    return {"edges": edges, "finite_difference_step": str(args.block_step), "lengths": point[list(edges)].tolist(), "gradient": [str(x) for x in start],
            "matrix": matrix.tolist(), "exact_256bit_diagonal": diag,
            "diagonal_check_max_abs": float(np.max(np.abs(np.diag(matrix)-diag))),
            "two_step_numerical_noise_max": noise, "symmetric_eigenvalues": eig.tolist(),
            "condition_estimate": float(np.linalg.cond(matrix)),
            "smallest_eigenvalue_resolved": bool(min(abs(eig)) > 3*noise),
            "high_precision_tangent_evaluations": count, "high_precision_split_evaluations": len(edges)}


def full(engine, auditor, point, args):
    guard = ZeroBoundary(engine, args.boundary)
    with args.trace.open("x", encoding="utf-8") as let:
        def emit(row):
            let.write(json.dumps(row, allow_nan=False)+"\n")
            let.flush()

        terminal, result = solve(engine, auditor, guard, point, emit)
    np.save(args.out_checkpoint, terminal, allow_pickle=False)
    with localcontext() as context:
        context.prec = 115
        dual = TangentReference(auditor, args.dual)
        reference = Reference(engine, terminal, args.reference)
        edge = result["cross_path_max_edge"]
        a, b = reference.score(edge, 256), dual.score(terminal, edge, 256)
        result["terminal_discrepancy_coordinate"] = {
            "edge": edge, "length": float(terminal[edge]), "split_256": str(a), "dual_256": str(b),
            "abs_difference": str(abs(a-b)), "nearest_binary64_representation_error": str(abs(a-Decimal.from_float(float(a)))),
            "high_precision_split_evaluations": 1, "high_precision_tangent_evaluations": 1,
        }
    result["terminal_sha256"] = hashlib.sha256(terminal.tobytes()).hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--block", action="store_true")
    parser.add_argument("--block-step", type=Decimal, default=Decimal("1.5e-12"))
    parser.add_argument("--case", choices=("short-10000", "warm-10000", "previous-10000"))
    for flag in ("prep", "gate", "library", "checkpoint", "stage26", "boundary", "dual", "reference", "trace", "out-checkpoint"):
        parser.add_argument("--" + flag, type=Path, required=flag == "library")
    args = parser.parse_args()
    started = perf_counter()
    if args.check:
        result = mathematics(args.library)
    else:
        engine, auditor, point, facts, digest = data(args)
        outcome = block(engine, auditor, point, args) if args.block else full(engine, auditor, point, args)
        result = {**facts, "case": args.case, "start_sha256": digest, **outcome}
    result["whole_wall_seconds"] = perf_counter()-started
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

"""Exhaustive small-tree calculus and recovery with anonymous categories."""
from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.optimize import minimize

from .check_f81_split_gradient import Reference
from .f81_optimize import optimize_projected, projected_norm
from .f81_unrooted import UnrootedPruning, UnrootedTopology
from .f81_zero_boundary import ZeroBoundary


EDGES = np.array([[4, 0], [4, 1], [4, 5], [5, 2], [5, 3]])
PI = np.array([0.25, 0.25, 0.5], dtype=np.longdouble)
TRUTHS = {
    "interior": np.array([0.13, 0.21, 0.32, 0.44, 0.58]),
    "zero_internal": np.array([0.13, 0.21, 0., 0.44, 0.58]),
    "zero_pendant": np.array([0., 0.21, 0.32, 0.44, 0.58]),
}
PATTERNS = np.array(list(product(range(3), repeat=4)), dtype=np.uint8)
ASSIGNMENTS = np.array(list(product(range(3), repeat=6))).reshape(len(PATTERNS), 9, 6)
ROOT_WEIGHTS = PI[ASSIGNMENTS[:, :, 4]]
LEFT, RIGHT = ASSIGNMENTS[:, :, EDGES[:, 0]], ASSIGNMENTS[:, :, EDGES[:, 1]]


def law(point, order):
    """Direct products over all internal assignments with differentiated factors."""
    stay = np.exp(-np.asarray(point, dtype=np.longdouble))
    change = -np.expm1(-np.asarray(point, dtype=np.longdouble))
    transition = stay * (LEFT == RIGHT) + change * PI[RIGHT]
    mass = np.sum(ROOT_WEIGHTS * np.prod(transition, axis=-1), axis=1)
    if order == 0:
        return (mass,)
    d_transition = stay * (PI[RIGHT] - (LEFT == RIGHT))
    first = []
    for e in range(len(point)):
        factors = transition.copy()
        factors[:, :, e] = d_transition[:, :, e]
        first.append(np.sum(ROOT_WEIGHTS * np.prod(factors, axis=-1), axis=1))
    first = np.column_stack(first)
    if order == 1:
        return mass, first
    second = np.empty((len(mass), len(point), len(point)), dtype=np.longdouble)
    for e in range(len(point)):
        second[:, e, e] = -first[:, e]
        for f in range(e):
            factors = transition.copy()
            factors[:, :, [e, f]] = d_transition[:, :, [e, f]]
            terms = np.sum(ROOT_WEIGHTS * np.prod(factors, axis=-1), axis=1)
            second[:, e, f] = second[:, f, e] = terms
    return mass, first, second


def engine_for(patterns, library, root=4):
    topology = UnrootedTopology.from_edges(4, EDGES, root=root)
    profiles = np.zeros((len(patterns), 21))
    profiles[:, :3] = PI
    return UnrootedPruning(topology, patterns.T, profiles, library)


def exhaustive_checks(library, reference):
    results = {}
    for label, point in TRUTHS.items():
        mass, first, second = law(point, 2)
        exact_g = -first / mass[:, None]
        exact_h = exact_g[:, :, None] * exact_g[:, None, :] - second / mass[:, None, None]
        expected_h = np.einsum("ijk,i->jk", exact_h, mass)
        assert np.abs(mass.sum() - 1) < 1e-17
        assert np.max(np.abs(np.sum(first, axis=0))) < 1e-17
        assert np.max(np.abs(np.sum(second, axis=0))) < 1e-17
        assert np.max(np.abs(np.sum(mass[:, None] * exact_g, axis=0))) < 1e-17
        eig = np.linalg.eigvalsh(expected_h.astype(float))
        assert eig.min() > 0

        def eval_patterns(values):
            evaluations = [engine_for(x[None, :], library).evaluate(values, threads=1) for x in PATTERNS]
            return np.array([x.log_likelihood for x in evaluations]), -np.array([x.length_gradient for x in evaluations])

        ll, g = eval_patterns(point)
        def gradient_difference(h):
            fd = np.empty_like(exact_h)
            for e in range(len(point)):
                offset = np.eye(len(point))[e] * h
                if point[e] == 0:
                    fd[:, :, e] = (-3*g + 4*eval_patterns(point + offset)[1]
                                   - eval_patterns(point + 2*offset)[1]) / (2*h)
                else:
                    fd[:, :, e] = (eval_patterns(point + offset)[1] - eval_patterns(point - offset)[1]) / (2*h)
            return fd

        coarse, fine = (gradient_difference(h) for h in (2e-6, 1e-6))
        fd = (4 * fine - coarse) / 3
        split_errors = {}
        ref = Reference(engine_for(PATTERNS, library, root=5), point, reference)
        for bits in (192, 256):
            terms = [ref.site_terms(e, bits)[1] for e in range(len(point))]
            p, grad, diag = (np.array([[np.longdouble(str(row[k])) for row in edge]
                                      for edge in terms]).T for k in (2, 3, 4))
            split_errors[str(bits)] = {
                "probability": float(np.max(np.abs(p - mass[:, None]))),
                "gradient": float(np.max(np.abs(grad - exact_g))),
                "diagonal_hessian": float(np.max(np.abs(diag - np.diagonal(exact_h, axis1=1, axis2=2)))),
            }

        err = {"log_likelihood_abs": float(np.max(np.abs(ll - np.log(mass)))),
               "gradient_abs": float(np.max(np.abs(g - exact_g))),
               "full_hessian_fd_abs": float(np.max(np.abs(fd - exact_h))),
               "full_hessian_fd_coarse_abs": float(np.max(np.abs(coarse - exact_h))),
               "full_hessian_fd_fine_abs": float(np.max(np.abs(fine - exact_h))),
               "split_vs_enumeration": split_errors}
        assert err["log_likelihood_abs"] < 1e-12 and err["gradient_abs"] < 1e-10
        assert err["full_hessian_fd_abs"] < 3e-6, (label, err)
        assert all(r["probability"] < 1e-16 and r["gradient"] < 1e-12 and r["diagonal_hessian"] < 1e-11
                   for r in split_errors.values())
        results[label] = {"truth": point.tolist(), "patterns": len(mass), "full_assignments": len(mass)*9,
                          "hessian_eigenvalues_expected_objective": eig.tolist(), **err}

    impossible = engine_for(np.array([[0, 1, 0, 1]], dtype=np.uint8), library).evaluate(np.zeros(5), threads=1)
    assert impossible.log_likelihood == -np.inf and impossible.length_gradient is None
    return {"cases": results, "zero_length_conflict_test": "zero likelihood", "exhaustive_passed": True}


def recovery(library, boundary):
    starts = (np.full(5, .1), np.full(5, .55), np.array([.42, .05, .70, .15, .90]))
    results = []
    for label, truth in TRUTHS.items():
        probability = np.asarray(law(truth, 0)[0], dtype=float)
        for n in (263, 2048, 16384):
            for seed in (713, 811, 947):
                sample = np.random.default_rng(seed).choice(len(PATTERNS), size=n, p=probability)
                empirical = np.bincount(sample, minlength=len(PATTERNS)) / n
                used = empirical > 0

                def exact_objective(point):
                    mass, first = law(point, 1)
                    if np.any(mass[used] <= 0):
                        return np.inf, np.zeros(len(point))
                    weight = empirical[used]
                    return float(-weight @ np.log(mass[used])), np.asarray(-weight @ (first[used] / mass[used, None]), dtype=float)

                oracle = minimize(exact_objective, starts[0], jac=True, method="SLSQP",
                                  bounds=[(0., None)]*5, options={"ftol": 1e-14, "maxiter": 150})
                oracle_residual = projected_norm(oracle.x, exact_objective(oracle.x)[1])
                engine = engine_for(PATTERNS[sample], library)
                auditor = engine_for(PATTERNS[sample], library, root=5)
                guard = ZeroBoundary(engine, boundary)
                for start_id, start in enumerate(starts):
                    events = []
                    point, fit = optimize_projected(engine, start, emit=events.append, threads=4,
                                                    maxiter=150, maxeval=260, gtol=1e-5,
                                                    restrict_history=True, zero_boundary=guard,
                                                    scale_initial_step=True)
                    assert sum(e["event"] == "evaluation" for e in events) == fit["solver_likelihood_gradient_evaluations"]
                    ordinary = -engine.evaluate(point, threads=4).length_gradient / n
                    rerooted = -auditor.evaluate(point, threads=4, log_domain=True).length_gradient / n
                    loss, g = exact_objective(point)
                    result = {
                        "truth_case": label, "n": n, "seed": seed, "start_id": start_id,
                        "start": start.tolist(), "estimate": point.tolist(),
                        "L2_truth_error": float(np.linalg.norm(point-truth)),
                        "L2_sample_optimum_difference": float(np.linalg.norm(point-oracle.x)),
                        "sample_objective_gap": loss - float(oracle.fun),
                        "oracle_fit": oracle.x.tolist(), "oracle_solver_success": bool(oracle.success),
                        "oracle_evaluations": int(oracle.nfev),
                        "oracle_projected_gradient": oracle_residual,
                        "oracle_extra_terminal_evaluations_per_sample": 1,
                        "oracle_terminal_projected_gradient": projected_norm(point, g),
                        "rerooted_terminal_projected_gradient": projected_norm(point, rerooted),
                        "cross_path_gradient_abs": float(np.max(np.abs(ordinary-rerooted))),
                        "ordinary_gradient_abs_vs_oracle": float(np.max(np.abs(ordinary-g))),
                        "rerooted_gradient_abs_vs_oracle": float(np.max(np.abs(rerooted-g))),
                        "solver_evaluations": fit["solver_likelihood_gradient_evaluations"],
                        "solver_terminal_evaluations": fit["additional_terminal_evaluations"],
                        "independent_terminal_objective_or_gradient_evaluations": 3,
                        "status": fit["status"], "solver_passed": fit["passed"],
                    }
                    results.append(result)
    curve = {label: {str(n): float(np.median([r["L2_truth_error"] for r in results
                                             if r["truth_case"] == label and r["n"] == n]))
                     for n in (263, 2048, 16384)} for label in TRUTHS}
    return {"cases": results, "median_L2_truth_error": curve,
            "independent_gradient_gate": 1e-8, "projection_gate": 1e-5,
            "all_passed": all(r["solver_passed"] and r["oracle_terminal_projected_gradient"] <= 1e-5
                              and r["rerooted_terminal_projected_gradient"] <= 1e-5
                              and r["cross_path_gradient_abs"] <= 1e-8 for r in results)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("library", "reference", "boundary"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--exhaustive-only", action="store_true")
    args = parser.parse_args()
    started = perf_counter()
    result = exhaustive_checks(args.library, args.reference)
    if not args.exhaustive_only:
        result["recovery"] = recovery(args.library, args.boundary)
    result["elapsed_seconds"] = perf_counter() - started
    print(json.dumps(result, allow_nan=False, indent=2))


if __name__ == "__main__":
    main()

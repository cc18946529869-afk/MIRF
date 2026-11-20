"""Bound-constrained fixed-topology optimization on unrooted edge coordinates."""
from __future__ import annotations

import math
from time import perf_counter

import numpy as np
from scipy.optimize import minimize

from potts_fit import _two_loop

from .f81_unrooted import UnrootedPruning


class EvaluationLimit(Exception):
    pass


def projected_norm(lengths, derivative):
    residual = lengths - np.maximum(0, lengths - derivative)
    return float(np.max(np.abs(residual)))


def optimize(engine, initial, *, emit, threads=4, maxiter=150, maxeval=260, gtol=1e-5, scale=1.0):
    if not isinstance(engine, UnrootedPruning):
        raise TypeError("Optimization requires unrooted edge coordinates")
    if maxiter < 1 or maxeval < 1 or not math.isfinite(gtol) or gtol <= 0:
        raise ValueError("Invalid iteration, evaluation or gradient limit")
    x = np.asarray(initial, dtype=float)
    if x.shape != engine.topology.child_to_edge.shape or not np.isfinite(x).all() or np.any(x < 0):
        raise ValueError("Invalid initial effective lengths")
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Numerical objective scale must be positive and finite")
    n_sites = len(engine.pruning.states)
    accepted = x.copy()
    calls = iterations = impossible = log_replays = 0
    first = None

    def objective(lengths):
        nonlocal calls, impossible, first, log_replays
        if calls >= maxeval:
            raise EvaluationLimit
        calls += 1
        before = perf_counter()
        item = engine.evaluate(lengths, threads=threads)
        log_replays += item.log_domain_sites
        if not math.isfinite(item.log_likelihood) or item.length_gradient is None:
            impossible += 1
            emit({"event": "evaluation", "call": calls, "finite": False,
                  "seconds": perf_counter() - before})
            if calls == 1:
                raise ValueError("Initial tree has zero likelihood")
            # This trial has infinite cost. The placeholder gradient is
            # returned only to the line search, never used as a convergence check.
            return np.inf, np.zeros_like(lengths)
        value = -item.log_likelihood / n_sites
        derivative = -item.length_gradient / n_sites
        check = projected_norm(lengths, derivative)
        if first is None:
            first = item.log_likelihood
        emit({"event": "evaluation", "call": calls, "finite": True,
              "log_likelihood": item.log_likelihood, "projected_gradient_per_site": check,
              "zero_edges": int(np.count_nonzero(lengths == 0)), "seconds": perf_counter() - before})
        return scale * value, scale * derivative

    def callback(point):
        nonlocal accepted, iterations
        accepted = point.copy()
        iterations += 1
        emit({"event": "accepted", "iteration": iterations, "calls": calls})

    started = perf_counter()
    converged = False
    try:
        fit = minimize(objective, x, method="L-BFGS-B", jac=True, bounds=[(0, None)] * x.size,
                       callback=callback, options={"maxiter": maxiter, "maxfun": maxeval, "maxcor": 10,
                                                   "maxls": 24, "ftol": 0.0, "gtol": gtol * scale})
        if int(fit.nfev) != calls or int(fit.nit) != iterations:
            raise RuntimeError("Iteration ledger does not match SciPy")
        candidate = fit.x
        converged = bool(fit.success)
        status = str(fit.message)
    except EvaluationLimit:
        candidate = accepted
        status = "hard_limit_on_likelihood_gradient_evaluations"
    search_seconds = perf_counter() - started
    before = perf_counter()
    terminal = engine.evaluate(candidate, threads=threads)
    verification_seconds = perf_counter() - before
    if terminal.length_gradient is None or candidate.min() < 0:
        raise RuntimeError("Returned terminal is infeasible")
    norm = projected_norm(candidate, -terminal.length_gradient / n_sites)
    result = {
        "solver": "SciPy L-BFGS-B", "numerical_objective_scale": scale, "criterion": "infinity norm of projected mean negative log-likelihood gradient",
        "gtol_per_site": gtol, "passed": converged and norm <= gtol,
        "solver_success": converged, "status": status,
        "iterations": iterations, "solver_likelihood_gradient_evaluations": calls,
        "additional_terminal_evaluations": 1, "infeasible_trials": impossible, "log_domain_site_replays": log_replays,
        "initial_log_likelihood": first, "terminal_log_likelihood": terminal.log_likelihood,
        "terminal_projected_gradient_per_site": norm,
        "terminal_zero_edges": int(np.count_nonzero(candidate == 0)),
        "search_seconds": search_seconds, "terminal_verification_seconds": verification_seconds,
        "maxiter": maxiter, "maxeval": maxeval,
    }
    return candidate, result


def optimize_projected(engine, initial, *, emit, threads=4, maxiter=150, maxeval=260,
                       gtol=1e-5, restrict_history=False, zero_boundary=None):
    """Projected L-BFGS with Armijo backtracking on the actual feasible trial."""
    if not isinstance(engine, UnrootedPruning):
        raise TypeError("Optimization requires unrooted edge coordinates")
    if maxiter < 1 or maxeval < 1 or not math.isfinite(gtol) or gtol <= 0:
        raise ValueError("Invalid iteration, evaluation or gradient limit")
    point = np.asarray(initial, dtype=float).copy()
    if (point.shape != engine.topology.child_to_edge.shape or not np.isfinite(point).all()
            or np.any(point < 0)):
        raise ValueError("Invalid initial effective lengths")
    n_sites = len(engine.pruning.states)
    calls = iterations = impossible = log_replays = resets = 0

    def objective(lengths):
        nonlocal calls, impossible, log_replays
        if calls >= maxeval:
            raise EvaluationLimit
        calls += 1
        before = perf_counter()
        item = engine.evaluate(lengths, threads=threads)
        log_replays += item.log_domain_sites
        seconds = perf_counter() - before
        if item.length_gradient is None or not math.isfinite(item.log_likelihood):
            impossible += 1
            emit({"event": "evaluation", "call": calls, "finite": False, "seconds": seconds})
            return None
        derivative = -item.length_gradient / n_sites
        emit({"event": "evaluation", "call": calls, "finite": True,
              "log_likelihood": item.log_likelihood,
              "projected_gradient_per_site": projected_norm(lengths, derivative),
              "zero_edges": int(np.count_nonzero(lengths == 0)), "seconds": seconds})
        return -item.log_likelihood / n_sites, derivative

    stats = {
        "line_searches": 0, "ray_limits": 0, "capped_searches": 0, "prechecks": 0,
        "precheck_rejections": 0, "infeasible_full_step_probes": 0,
        "infeasible_likelihood_calls_avoided": 0, "nonfinite_length_trials": 0,
        "precheck_seconds": 0.0, "ray_limit_seconds": 0.0,
    }

    def check(trial):
        before = perf_counter()
        feasible = zero_boundary.feasible(trial)
        stats["prechecks"] += 1
        stats["precheck_rejections"] += not feasible
        stats["precheck_seconds"] += perf_counter() - before
        return feasible

    def search(direction):
        step = 1.0
        if zero_boundary is not None:
            stats["line_searches"] += 1
            before = perf_counter()
            limit = zero_boundary.first_conflict(point, direction)
            step = zero_boundary.bounded_step(limit)
            stats["ray_limits"] += 1
            stats["ray_limit_seconds"] += perf_counter() - before
            if step < 1:
                stats["capped_searches"] += 1
                full_step = np.maximum(0, point + direction)
                if (np.isfinite(full_step).all() and np.dot(derivative, full_step-point) < 0
                        and not check(full_step)):
                    stats["infeasible_full_step_probes"] += 1
                    stats["infeasible_likelihood_calls_avoided"] += 1
                emit({"event": "zero_boundary_cap", "first_conflict": limit, "initial_step": step})
        for _ in range(24):
            trial = np.maximum(0, point + step * direction)
            if zero_boundary is not None and not np.isfinite(trial).all():
                stats["nonfinite_length_trials"] += 1
                step *= 0.5
                continue
            displacement = trial - point
            decrease = float(np.dot(derivative, displacement))
            if decrease < 0:
                if zero_boundary is not None and not check(trial):
                    stats["infeasible_likelihood_calls_avoided"] += 1
                    step *= 0.5
                    continue
                result = objective(trial)
                if result is not None and result[0] <= value + 1e-4 * decrease:
                    return trial, displacement, result
            step *= 0.5
        return None

    if zero_boundary is not None and zero_boundary.topology is not engine.topology:
        raise ValueError("The feasibility guard must use the optimizer's edge coordinates")
    if zero_boundary is not None and not check(point):
        raise ValueError("Initial point has conflicting observations on zero edges")

    started = perf_counter()
    initial_result = objective(point)
    if initial_result is None:
        raise ValueError("Initial tree has zero likelihood")
    value, derivative = initial_result
    first = -value * n_sites
    steps, changes, curvature_history = [], [], []
    status = "iteration_limit"
    try:
        while iterations < maxiter:
            if projected_norm(point, derivative) <= gtol:
                status = "projected_gradient_converged"
                break
            active = (point == 0) & (derivative >= 0)
            free_gradient = np.where(active, 0., derivative)
            if restrict_history and active.any():
                free = ~active
                d_steps, d_changes, d_curvatures = [], [], []
                for step, change in zip(steps, changes):
                    ds, dy = step[free], change[free]
                    sc = float(np.dot(ds, dy))
                    if sc > 1e-10 * np.linalg.norm(ds) * np.linalg.norm(dy):
                        d_steps.append(ds)
                        d_changes.append(dy)
                        d_curvatures.append(sc)
                direction = np.zeros_like(point)
                direction[free] = _two_loop(derivative[free], d_steps, d_changes, d_curvatures)
            else:
                direction = _two_loop(free_gradient, steps, changes, curvature_history)
            direction[(point == 0) & (direction < 0)] = 0
            answer = search(direction)
            if answer is None and steps:
                steps.clear()
                changes.clear()
                curvature_history.clear()
                resets += 1
                answer = search(-free_gradient)
            if answer is None:
                status = "no_finite_armijo_step"
                break
            trial, displacement, (new_value, new_derivative) = answer
            change = new_derivative - derivative
            curvature = float(np.dot(displacement, change))
            bound = 1e-10 * np.linalg.norm(displacement) * np.linalg.norm(change)
            if curvature > bound:
                if len(steps) == 10:
                    steps.pop(0)
                    changes.pop(0)
                    curvature_history.pop(0)
                steps.append(displacement)
                changes.append(change)
                curvature_history.append(curvature)
            point, value, derivative = trial, new_value, new_derivative
            iterations += 1
            emit({"event": "accepted", "iteration": iterations, "calls": calls})
        if projected_norm(point, derivative) <= gtol:
            status = "projected_gradient_converged"
    except EvaluationLimit:
        status = "hard_limit_on_likelihood_gradient_evaluations"
    search_seconds = perf_counter() - started
    before = perf_counter()
    terminal = engine.evaluate(point, threads=threads)
    verification_seconds = perf_counter() - before
    if terminal.length_gradient is None:
        raise RuntimeError("Accepted terminal is infeasible")
    norm = projected_norm(point, -terminal.length_gradient / n_sites)
    passed = status == "projected_gradient_converged" and norm <= gtol
    return point, {
        "solver": "Projected L-BFGS with feasible Armijo backtracking",
        "restricted_curvature_to_free_coordinates": restrict_history,
        "criterion": "infinity norm of projected mean negative log-likelihood gradient",
        "gtol_per_site": gtol, "passed": passed, "solver_success": status == "projected_gradient_converged",
        "status": status, "iterations": iterations, "solver_likelihood_gradient_evaluations": calls,
        "additional_terminal_evaluations": 1, "infeasible_trials": impossible,
        "log_domain_site_replays": log_replays, "history_resets": resets,
        "initial_log_likelihood": first, "terminal_log_likelihood": terminal.log_likelihood,
        "terminal_projected_gradient_per_site": norm,
        "terminal_zero_edges": int(np.count_nonzero(point == 0)),
        "search_seconds": search_seconds, "terminal_verification_seconds": verification_seconds,
        "maxiter": maxiter, "maxeval": maxeval,
        **({"zero_boundary": stats} if zero_boundary is not None else {}),
    }

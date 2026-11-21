"""Bound-constrained fixed-topology optimization on unrooted edge coordinates."""
from __future__ import annotations

import math
from time import perf_counter

import numpy as np
from scipy.optimize import minimize

from potts_fit import _two_loop

from .f81_unrooted import UnrootedPruning
from .f81_zero_boundary import FRACTION_TO_CONFLICT


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


ARMIJO_C1 = 1e-4


def projected_scale_step(point, direction, gradient, value, step):
    """Cap the initial predicted decrease at 1% of the current objective."""
    if not value > 0:
        return step
    target = 0.01 * value
    adjustments = 0
    while step > 0:
        with np.errstate(over="ignore", invalid="ignore"):
            displacement = np.maximum(0, point + step * direction) - point
            predicted = -ARMIJO_C1 * float(gradient @ displacement)
        if np.isfinite(predicted):
            if predicted <= target:
                return step
            ratio = min(0.95 if adjustments < 8 else 0.5, target / predicted)
        else:
            # Bring an out-of-range displacement or inner product into range
            # before computing a binary64 prediction from the actual trial.
            ratio = 2.0 ** -16
        step = min(step * ratio, math.nextafter(step, 0.0))
        adjustments += 1
    return step


def expansion_steps(anchor, conflict):
    """Probe separated scales without assuming a connected Armijo interval."""
    bound = (min(FRACTION_TO_CONFLICT * conflict, math.nextafter(conflict, 0.0))
             if math.isfinite(conflict) else math.inf)
    previous = anchor
    for exponent in (2, 7, 13, 20, 26):
        step = min(math.ldexp(anchor, exponent), bound)
        if step > previous:
            yield step
            previous = step


def normalized_negative_gradient(free_gradient):
    """Limit each free component to unit magnitude, preserving its sign."""
    return -free_gradient / np.maximum(1., np.abs(free_gradient))


def stale_direction(stale_pairs, gamma, descent_cosine):
    return (stale_pairs >= 8 and gamma is not None and gamma <= 1e-20
            and descent_cosine is not None and descent_cosine <= 1e-8)


def optimize_projected(engine, initial, *, emit, threads=4, maxiter=150, maxeval=260,
                       gtol=1e-5, restrict_history=False, zero_boundary=None,
                       search_diagnostics=False, terminal_diagnostic=None,
                       scale_initial_step=False, expand_steps=False, recover_history=False,
                       normalized_only=False, iteration_diagnostics=False):
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
    stale_pairs = accepted_pairs = recovery_resets = 0
    expansion_trials = expansion_evaluations = expansion_selections = 0

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

    search_number = 0
    last_directions = []

    def trial_at(direction, step, attempt, phase):
        diagnostic = {"event": "line_trial", "search": search_number, "attempt": attempt,
                      "step": step, "current_value": value} if search_diagnostics else None
        if diagnostic is not None and expand_steps:
            diagnostic["phase"] = phase

        def finish(outcome, **fields):
            if diagnostic is not None:
                emit({**diagnostic, **fields, "outcome": outcome})

        trial = np.maximum(0, point + step * direction)
        if zero_boundary is not None and not np.isfinite(trial).all():
            stats["nonfinite_length_trials"] += 1
            finish("nonfinite_lengths")
            return None
        displacement = trial - point
        decrease = float(np.dot(derivative, displacement))
        if diagnostic is not None:
            diagnostic.update(projected_directional_derivative=decrease,
                              displacement_inf=float(np.max(np.abs(displacement))),
                              moved_edges=int(np.count_nonzero(displacement)))
        if not decrease < 0:
            finish("no_movement" if not np.any(displacement) else "not_descent_after_projection")
            return None
        if zero_boundary is not None and not check(trial):
            stats["infeasible_likelihood_calls_avoided"] += 1
            finish("boundary_conflict")
            return None
        armijo = value + ARMIJO_C1 * decrease
        if scale_initial_step and armijo < 0:
            finish("armijo_below_objective_bound", armijo_rhs=armijo)
            return None
        if calls >= maxeval:
            finish("evaluation_limit")
        result = objective(trial)
        if result is None:
            finish("zero_likelihood")
            return None
        passed = result[0] <= armijo
        if passed:
            outcome = "armijo_candidate" if expand_steps else "accepted"
        else:
            outcome = "armijo_rejected"
        finish(outcome, trial_value=result[0], armijo_rhs=armijo, armijo_gap=result[0] - armijo)
        if passed:
            return trial, displacement, result, calls
        return None

    def search(direction, method):
        nonlocal search_number, expansion_trials, expansion_evaluations, expansion_selections
        step, limit = 1.0, math.inf
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

        search_number += 1
        if scale_initial_step:
            initial_step = step
            step = projected_scale_step(point, direction, derivative, value, step)
            emit({"event": "projected_scale", "search": search_number,
                  "unscaled_step": initial_step, "scaled_step": step})
        if terminal_diagnostic is not None:
            last_directions.append((method, step, direction.copy()))
        if search_diagnostics:
            emit({"event": "line_search", "search": search_number, "method": method,
                  "iteration": iterations, "start_step": step, "history": len(steps),
                  "direction_l2": float(np.linalg.norm(direction)),
                  "direction_inf": float(np.max(np.abs(direction))),
                  "directional_derivative": float(np.dot(derivative, direction))})
        for attempt in range(1, 25):
            answer = trial_at(direction, step, attempt, "backtrack")
            if answer is None:
                step *= 0.5
                continue
            if (not expand_steps or attempt != 1 or iterations % 8
                    or np.max(np.abs(answer[1])) > 1e-5):
                return answer

            # Spend at most five of the same search's 24 trial slots. Keep a
            # passing incumbent if the global evaluation limit arrives here.
            best = answer
            for probe, larger in enumerate(expansion_steps(step, limit), start=2):
                if calls >= maxeval:
                    break
                expansion_trials += 1
                before = calls
                candidate = trial_at(direction, larger, probe, "expansion")
                expansion_evaluations += calls - before
                if candidate is not None and candidate[2][0] < best[2][0]:
                    best = candidate
            expansion_selections += best[3] != answer[3]
            return best
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
            used_changes, used_curvatures = changes, curvature_history
            if normalized_only:
                direction = normalized_negative_gradient(free_gradient)
            elif restrict_history and active.any():
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
                used_changes, used_curvatures = d_changes, d_curvatures
            else:
                direction = _two_loop(free_gradient, steps, changes, curvature_history)
            direction[(point == 0) & (direction < 0)] = 0
            method = "normalized_gradient" if normalized_only else "lbfgs"
            if recover_history or iteration_diagnostics:
                gamma = (used_curvatures[-1] / float(used_changes[-1] @ used_changes[-1])
                         if used_curvatures else None)
                product = float(np.linalg.norm(free_gradient) * np.linalg.norm(direction))
                cosine = float(free_gradient @ direction) / product if product else None
                if iteration_diagnostics:
                    emit({"event": "direction_state", "iteration": iterations, "method": method,
                          "gamma": gamma, "stale_pairs": stale_pairs, "free_gradient_cosine": cosine,
                          "active_constraints": int(np.count_nonzero(active))})
                if recover_history and stale_direction(stale_pairs, gamma, None if cosine is None else -cosine):
                    steps.clear()
                    changes.clear()
                    curvature_history.clear()
                    resets += 1
                    recovery_resets += 1
                    emit({"event": "history_recovery", "iteration": iterations,
                          "gamma": gamma, "stale_pairs": stale_pairs, "free_gradient_cosine": cosine})
                    stale_pairs = 0
                    direction = normalized_negative_gradient(free_gradient)
                    method = "normalized_gradient_after_stale_history"
            last_directions.clear()
            answer = search(direction, method)
            if answer is None and steps:
                steps.clear()
                changes.clear()
                curvature_history.clear()
                resets += 1
                stale_pairs = 0
                fallback = normalized_negative_gradient(free_gradient) if recover_history else -free_gradient
                method = "normalized_gradient_after_failed_search" if recover_history else "projected_gradient_after_reset"
                answer = search(fallback, method)
            if answer is None:
                status = "no_finite_armijo_step"
                break
            trial, displacement, (new_value, new_derivative), selected_call = answer
            change = new_derivative - derivative
            curvature = float(np.dot(displacement, change))
            bound = 1e-10 * np.linalg.norm(displacement) * np.linalg.norm(change)
            if not normalized_only and curvature > bound:
                accepted_pairs += 1
                stale_pairs = 0
                if len(steps) == 10:
                    steps.pop(0)
                    changes.pop(0)
                    curvature_history.pop(0)
                steps.append(displacement)
                changes.append(change)
                curvature_history.append(curvature)
            else:
                stale_pairs += 1
            point, value, derivative = trial, new_value, new_derivative
            iterations += 1
            emit({"event": "accepted", "iteration": iterations, "calls": selected_call})
        if projected_norm(point, derivative) <= gtol:
            status = "projected_gradient_converged"
    except EvaluationLimit:
        status = "hard_limit_on_likelihood_gradient_evaluations"
    search_seconds = perf_counter() - started
    before = perf_counter()
    terminal = engine.evaluate(point, threads=threads)
    verification_seconds = perf_counter() - before
    if terminal_diagnostic is not None and status != "projected_gradient_converged":
        terminal_diagnostic(point, derivative, terminal, last_directions, status)
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
        "curvature_pairs_kept": accepted_pairs, "stale_history_resets": recovery_resets,
        "expansion_trials": expansion_trials, "expansion_evaluations": expansion_evaluations,
        "expansion_selections": expansion_selections,
        "initial_log_likelihood": first, "terminal_log_likelihood": terminal.log_likelihood,
        "terminal_projected_gradient_per_site": norm,
        "terminal_zero_edges": int(np.count_nonzero(point == 0)),
        "search_seconds": search_seconds, "terminal_verification_seconds": verification_seconds,
        "maxiter": maxiter, "maxeval": maxeval,
        **({"zero_boundary": stats} if zero_boundary is not None else {}),
    }

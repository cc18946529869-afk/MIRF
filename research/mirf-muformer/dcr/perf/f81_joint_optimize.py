"""Joint mixed-coordinate optimization for a fixed F81 topology."""
import hashlib
import json
import math
import os
from pathlib import Path
from time import perf_counter

import numpy as np

from .f81_optimize import ARMIJO_C1, projected_norm


TRUST_LOG = 4.
TRUST_ZERO = .25


def chart(point, physical_gradient, physical_diagonal):
    positive = point > 0
    active = ~positive & (physical_gradient >= 0)
    factor = np.where(positive, point, 1.)
    grad = physical_gradient * factor
    # Multiplying by s in two steps avoids premature underflow of s squared.
    diag = (physical_diagonal * factor) * factor + np.where(positive, grad, 0.)
    grad[active] = 0.
    return positive, active, grad, diag


def direction(gradient, diagonal, radius, history):
    # Use the current diagonal as the initial inverse Hessian, rather than
    # deriving one global scalar from the most recent secant pair.
    scale = np.maximum(np.abs(diagonal), np.abs(gradient)/radius)
    q = gradient.copy()
    alpha = []
    for dx, dg, sy in reversed(history):
        a = float(dx @ q)/sy
        alpha.append(a)
        q -= a*dg
    response = np.divide(q, scale, out=np.zeros_like(q), where=scale != 0)
    for (dx, dg, sy), a in zip(history, reversed(alpha)):
        response += dx*(a-float(dg @ response)/sy)
    return np.clip(-response, -radius, radius)


def curvature_pair(dx, dg):
    sy = float(dx @ dg)
    threshold = 1e-10 * np.linalg.norm(dx) * np.linalg.norm(dg)
    return (dx, dg, sy) if sy > threshold and math.isfinite(sy) else None


def save_checkpoint(directory, name, *, lengths, physical_gradient, hessian_diagonal,
                    mixed_gradient, objective, projection, iteration, objective_calls):
    """Write one full parameter state and a JSON sidecar containing its SHA-256."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    arrays = {
        "lengths": np.ascontiguousarray(lengths, dtype=np.float64),
        "physical_gradient": np.ascontiguousarray(physical_gradient, dtype=np.float64),
        "hessian_diagonal": np.ascontiguousarray(hessian_diagonal, dtype=np.float64),
        "mixed_gradient": np.ascontiguousarray(mixed_gradient, dtype=np.float64),
    }
    temporary = directory / f".{name}.{os.getpid()}.npz"
    np.savez(temporary, **arrays, objective=np.float64(objective), projection=np.float64(projection),
             iteration=np.int64(iteration), objective_calls=np.int64(objective_calls))
    target = directory / f"{name}.npz"
    os.replace(temporary, target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    meta = {
        "name": name,
        "sha256": digest,
        "lengths_sha256": hashlib.sha256(arrays["lengths"].tobytes()).hexdigest(),
        "objective": float(objective),
        "projection": float(projection),
        "iteration": int(iteration),
        "objective_calls": int(objective_calls),
        "edges": int(arrays["lengths"].size),
    }
    (directory / f"{name}.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta


def candidate(point, positive, step, alpha, snap):
    trial = np.empty_like(point)
    trial[positive] = point[positive]*np.exp(alpha*step[positive])
    trial[~positive] = np.maximum(0., alpha*step[~positive])
    trial[snap & (alpha*step <= -.5)] = 0.
    return trial


def solve(engine, auditor, guard, initial, emit, *, maxiter=600, maxeval=1500, threads=4, archive=None):
    """Budget charged per actual objective call, including rejected trial points."""
    point = np.array(initial, dtype=float, copy=True)
    assert guard.feasible(point)
    start = perf_counter()
    counts = dict(objective=0, gradient=0, diagonal=0, line_search_trials=0,
                  independent_objective=0, independent_gradient=0)
    resets = accepted = pairs = 0
    history = []
    stopped = "iteration_limit"

    def evaluate(lengths, derivatives):
        counts["objective"] += 1
        counts["gradient"] += derivatives
        counts["diagonal"] += derivatives
        return engine.evaluate(lengths, threads=threads, gradient=derivatives, diagonal=derivatives)

    def independent(lengths):
        counts["independent_objective"] += 1
        counts["independent_gradient"] += 1
        item = auditor.evaluate(lengths, threads=threads, log_domain=True)
        return -item.length_gradient / n, item.log_likelihood

    current = evaluate(point, True)
    n = len(current.site_log_likelihood)
    gradient = -current.length_gradient/n
    diagonal = current.negative_log_hessian_diagonal/n
    objective_value = -current.log_likelihood/n
    emit({"event": "initial", "objective": objective_value,
          "projection": projected_norm(point, gradient)})
    best_projection = projected_norm(point, gradient)
    best_objective = objective_value
    checkpoint_meta = {}

    def remember(name):
        if archive is None:
            return
        _, _, mixed, _ = chart(point, gradient, diagonal)
        checkpoint_meta[name] = save_checkpoint(
            archive, name, lengths=point, physical_gradient=gradient, hessian_diagonal=diagonal,
            mixed_gradient=mixed, objective=objective_value, projection=projected_norm(point, gradient),
            iteration=accepted, objective_calls=counts["objective"])

    remember("initial")
    remember("best_projection")
    remember("best_objective")
    no_new_pair = 0
    while accepted < maxiter:
        if projected_norm(point, gradient) <= 1e-5:
            stopped = "candidate_stationary"
            break
        if counts["objective"]+1 >= maxeval:
            stopped = "objective_budget"
            break
        positive, active, g, h = chart(point, gradient, diagonal)
        radius = np.where(positive, TRUST_LOG, TRUST_ZERO)
        snap = positive & (gradient > 0) & (gradient >= point*np.abs(diagonal))
        found = False
        had_history = bool(history)
        for fallback in (False, True) if had_history else (False,):
            if fallback:
                history.clear()
                resets += 1
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                d = direction(g, h, radius, history)
            d[active] = 0.
            if not np.isfinite(d).all() or float(d @ g) >= 0:
                continue
            for attempt in range(24):
                if counts["objective"]+1 >= maxeval:
                    break
                alpha = math.ldexp(1., -attempt)
                counts["line_search_trials"] += 1
                trial = candidate(point, positive, d, alpha, snap)
                # A positive coordinate can hit zero only through an explicit
                # boundary proposal, never by rounding a log update to zero.
                requested_zero = snap & (alpha*d <= -.5)
                slope = float(gradient @ (trial-point))
                row = {"event": "trial", "alpha": alpha, "fallback": fallback,
                       "slope": slope if math.isfinite(slope) else None}
                if (not math.isfinite(slope) or slope >= 0 or not np.isfinite(trial).all()
                        or np.any(positive & (trial == 0) & ~requested_zero)):
                    emit({**row, "status": "invalid_step"})
                    continue
                if not guard.feasible(trial):
                    emit({**row, "status": "zero_conflict"})
                    continue
                try:
                    test = evaluate(trial, False)
                    if not math.isfinite(test.log_likelihood):
                        emit({**row, "status": "nonfinite_likelihood"})
                        continue
                    change = -math.fsum(test.site_log_likelihood-current.site_log_likelihood)/n
                    if change <= ARMIJO_C1*slope:
                        following = evaluate(trial, True)
                    else:
                        following = None
                except FloatingPointError:
                    emit({**row, "status": "numerical_range"})
                    continue
                found = False
                if following is not None and following.length_gradient is not None:
                    change = -math.fsum(following.site_log_likelihood-current.site_log_likelihood)/n
                    found = change <= ARMIJO_C1*slope
                emit({**row, "objective_change": change, "status": "accepted" if found else "rejected"})
                if found:
                    break
            if found:
                break
        if not found:
            stopped = "objective_budget" if counts["objective"]+1 >= maxeval else "line_search_stopped"
            break

        new_gradient = -following.length_gradient/n
        new_diagonal = following.negative_log_hessian_diagonal/n
        new_positive, new_active, new_g, _ = chart(trial, new_gradient, new_diagonal)
        flips = int(np.count_nonzero(new_positive != positive))
        # Chart or face changes invalidate those coordinates, not secants
        # on the remaining free subspace. Recheck curvature after masking.
        stable = (positive == new_positive) & ~active & ~new_active
        if flips or np.any(new_active != active):
            restricted = []
            for dx, dg, _ in history:
                pair = curvature_pair(np.where(stable, dx, 0.), np.where(stable, dg, 0.))
                if pair is not None:
                    restricted.append(pair)
            resets += len(history)-len(restricted)
            history = restricted
        dx = np.where(stable, trial-point, 0.)
        log_slots = stable & positive
        dx[log_slots] = np.log(trial[log_slots])-np.log(point[log_slots])
        pair = curvature_pair(dx, np.where(stable, new_g-g, 0.))
        if pair is not None:
            history.append(pair)
            history = history[-10:]
            pairs += 1
            no_new_pair = 0
        else:
            no_new_pair += 1
            if no_new_pair >= 8 and history:
                history.clear()
                resets += 1
                no_new_pair = 0

        changed_edges = int(np.count_nonzero(trial != point))
        point, current, gradient, diagonal = trial, following, new_gradient, new_diagonal
        accepted += 1
        objective_value = -current.log_likelihood/n
        p = projected_norm(point, gradient)
        if p < best_projection:
            best_projection = p
            remember("best_projection")
        if objective_value < best_objective:
            best_objective = objective_value
            remember("best_objective")
        emit({"event": "iterate", "accepted": accepted, "objective": objective_value,
              "projection": p, "sign_changes": flips, "history": len(history),
              "changed_edges": changed_edges,
              "counts": counts.copy(), "elapsed_seconds": perf_counter()-start})
        if accepted % 25 == 0:
            audit, ll = independent(point)
            emit({"event": "independent", "iteration": accepted, "projection": projected_norm(point, audit),
                  "log_likelihood": ll, "gradient_max_abs": float(np.abs(audit-gradient).max())})

    remember("final")
    audit, ll = independent(point)
    p, q = projected_norm(point, gradient), projected_norm(point, audit)
    discrepancy = np.abs(audit-gradient)
    feasible = guard.feasible(point)
    return point, {"status": stopped, "accepted": accepted, "history_pairs": pairs, "history_clear_or_discard_count": resets,
                   "objective": -current.log_likelihood/n, "independent_objective": -ll/n,
                   "projection": p, "independent_projection": q, "best_ordinary_projection": best_projection,
                   "cross_path_max": float(discrepancy.max()), "cross_path_max_edge": int(discrepancy.argmax()),
                   "zero_edges": int(np.count_nonzero(point == 0)), "feasible": feasible,
                   "both_gates_passed": bool(feasible and max(p, q) <= 1e-5 and discrepancy.max() <= 1e-8),
                   "counts": counts, "maxeval": maxeval, "maxiter": maxiter,
                   "elapsed_seconds": perf_counter()-start, "checkpoints": checkpoint_meta}

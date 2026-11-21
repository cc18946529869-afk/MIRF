"""Truncated Newton on the high-residual tail of a fixed F81 tree.

Stage 2.7 kept a mixed-coordinate L-BFGS step on every free edge. Accepted
steps moved about 16,000 edges at once, and the mean objective could fall
while the infinity-norm projection rose. At the recovered terminal state the
largest residuals sit on positive edges whose adjacent cross derivative is as
large as the diagonal, so the local Hessian has a stiff combination and a
near-null combination. An undamped Newton step follows that near-null
direction by a length of order one and leaves the positive orthant.

This solver therefore does four things the previous one does not:

- it frees only the residual tail and the positive edges that share a vertex
  with that tail, and it leaves every other edge fixed;
- the search direction is the verified diagonal step, clipped separately on
  each edge, not a secant step and not one global rescaling;
- one Hessian-vector product then sets a separate length on each edge. A
  single length is fixed by the weakly coupled majority. On a path whose
  cross curvature equals its diagonal, that length is several times the
  joint Newton step, and the shared gradient on the path changes sign.
- a trial is accepted when it stays feasible and satisfies the original
  Armijo condition. The infinity-norm projection is recorded for the
  convergence test. It is not a per-step acceptance gate: one edge can
  improve while another becomes the new maximum.

No L-BFGS pair is read or stored. A product that disagrees with the verified
diagonal aborts the run before any trial step.
"""
import math
from time import perf_counter

import numpy as np

from .f81_joint_optimize import chart, save_checkpoint
from .f81_optimize import ARMIJO_C1, projected_norm


class _Halt(Exception):
    def __init__(self, status):
        super().__init__(status)
        self.status = status


def incident_edges(pairs):
    """Map each edge to the other edges that share one of its vertices."""
    pairs = np.asarray(pairs)
    nodes = {}
    for edge, (child, parent) in enumerate(pairs.tolist()):
        nodes.setdefault(int(child), []).append(edge)
        nodes.setdefault(int(parent), []).append(edge)
    adjacency = []
    for edge, (child, parent) in enumerate(pairs.tolist()):
        seen = []
        for node in (int(child), int(parent)):
            for other in nodes[node]:
                if other != edge and other not in seen:
                    seen.append(other)
        adjacency.append(seen)
    return adjacency


def free_edges(lengths, gradient, adjacency):
    """Residual tail plus the positive edges adjacent to it.

    A zero edge with a nonnegative physical gradient is a bound constraint.
    It is not freed, even when it touches a high-residual edge.
    """
    residual = np.abs(lengths - np.maximum(0., lengths - gradient))
    peak = float(residual.max()) if residual.size else 0.
    threshold = max(1e-4, 0.05 * peak)
    seeds = np.flatnonzero((lengths > 0) & (residual >= threshold))
    if seeds.size == 0:
        positive = np.flatnonzero(lengths > 0)
        order = positive[np.argsort(residual[positive])[::-1]]
        seeds = order[: min(32, order.size)]
    chosen = {int(edge) for edge in seeds}
    for edge in list(chosen):
        for other in adjacency[edge]:
            if lengths[other] > 0:
                chosen.add(int(other))
    return np.array(sorted(chosen), dtype=int), threshold, int(seeds.size)


def _step_length(lengths, direction):
    """Centered probe that stays on the feasible side of every moved edge."""
    nz = np.flatnonzero(direction != 0)
    if nz.size == 0:
        return 0.
    span = float(np.max(np.abs(direction[nz])))
    rooms = 0.2 * lengths[nz] / np.abs(direction[nz])
    eps = min(1e-6 / span, float(np.min(rooms)))
    return max(eps, 0.)


def hessian_product(lengths, direction, gradient_at):
    eps = _step_length(lengths, direction)
    if eps == 0.:
        return np.zeros_like(lengths)
    plus = lengths + eps * direction
    minus = lengths - eps * direction
    if np.any(plus < 0) or np.any(minus < 0):
        raise FloatingPointError("Hessian product left the nonnegative orthant")
    right = gradient_at(plus)
    left = gradient_at(minus)
    return (right - left) / (2. * eps)


def _cap_tau(delta, direction, cap):
    tau = math.inf
    for moved, component, limit in zip(delta, direction, cap):
        if component == 0.:
            continue
        room = (limit - moved) / component if component > 0 else (-limit - moved) / component
        tau = min(tau, room)
    if tau is math.inf:
        return 0.
    return max(0., float(tau))


def capped_diagonal_step(gradient, diagonal, product, index, cap):
    """Clip each diagonal step, then shorten each edge by the joint curvature.

    A single truncated-Newton direction in all free coordinates is not usable
    here. A few edges have a diagonal step far larger than their length cap.
    Shrinking the whole vector until the worst edge fits also shrinks the
    stiff edges, whose useful step is only a few thousandths of a substitution
    per site. Each edge is therefore clipped on its own. Edges with a
    nonpositive stored diagonal are left fixed: moving them is the negative
    curvature the undamped Newton step would follow.

    One Hessian-vector product of this clipped direction supplies ``(Hd)_i``.
    Edge ``i`` then takes the fraction ``min(1, max(0, -g_i / (Hd)_i))`` of
    its clipped step. Where the cross curvature equals the diagonal, as on
    the measured paths, ``(Hd)_i`` is about ``n`` times ``H_ii d_i`` and the
    fraction is about ``1/n``. A single fraction cannot do that: the Rayleigh
    quotient of a large free set stays near one, so every edge, including the
    path, keeps most of its diagonal step. An edge whose own curvature along
    the joint direction is not positive stays fixed for this iteration.
    """
    slope_gradient = gradient[index]
    curvature = diagonal[index]
    raw = np.divide(-slope_gradient, np.maximum(curvature, 1e-3),
                    out=np.zeros(index.size), where=np.isfinite(curvature))
    raw[curvature <= 0] = 0.
    direction = np.clip(raw, -cap, cap)
    info = {"hit": "clipped_diagonal", "model_alpha": 0., "rayleigh": None,
            "global_alpha": None,
            "clipped_edges": int(np.count_nonzero(np.abs(raw) > cap)),
            "frozen_nonpositive_diagonal": int(np.count_nonzero(curvature <= 0))}
    if not np.isfinite(direction).all() or float(slope_gradient @ direction) >= 0.:
        info["hit"] = "not_descent"
        return np.zeros(index.size), info
    full = np.zeros_like(gradient)
    full[index] = direction
    product_direction = product(full)[index]
    directional = float(direction @ product_direction)
    slope = float(slope_gradient @ direction)
    metric = float(direction @ (np.maximum(curvature, 1e-3) * direction))
    info["rayleigh"] = directional / metric if metric > 0 else None
    if directional > 0.:
        info["global_alpha"] = min(1., max(0., -slope / directional))
    else:
        info["global_alpha"] = 1.
    component = direction * product_direction
    scales = np.zeros(index.size)
    usable = component > 0.
    scales[usable] = np.clip(
        -(slope_gradient[usable] * direction[usable]) / component[usable], 0., 1.)
    step = scales * direction
    moved = direction != 0.
    if not np.isfinite(step).all() or not np.any(moved) or float(slope_gradient @ step) >= 0.:
        info["hit"] = "not_descent"
        return np.zeros(index.size), info
    info["hit"] = "coordinate_hessian_length"
    info["model_alpha"] = float(np.median(scales[moved]))
    info["coordinate_alpha_min"] = float(scales[moved].min())
    info["coordinate_alpha_max"] = float(scales[moved].max())
    info["model_slope"] = float(slope_gradient @ step)
    return step, info


def verify_products(lengths, gradient, diagonal, adjacency, gradient_at):
    """Compare products on the worst edges with the verified analytic diagonal."""
    index, _threshold, _seeds = free_edges(lengths, gradient, adjacency)
    residual = np.abs(lengths - np.maximum(0., lengths - gradient))
    order = index[np.argsort(residual[index])[::-1]]
    checked = []
    worst = 0.
    for edge in order[:8]:
        direction = np.zeros_like(lengths)
        direction[edge] = 1.
        column = hessian_product(lengths, direction, gradient_at)
        stored = float(diagonal[edge])
        scale = max(abs(stored), 1e-8)
        error = abs(float(column[edge]) - stored) / scale
        worst = max(worst, error)
        others = [other for other in adjacency[int(edge)] if lengths[other] > 0]
        neighbor = max(others, key=lambda other: abs(float(column[other])), default=None)
        checked.append({
            "edge": int(edge),
            "stored_diagonal": stored,
            "product_diagonal": float(column[edge]),
            "relative_error": error,
            "largest_neighbor": None if neighbor is None else int(neighbor),
            "neighbor_product": None if neighbor is None else float(column[neighbor]),
        })
    return {"checked": checked, "max_relative_error": worst, "accepted": worst <= 1e-3}


def solve_subspace_trust(engine, auditor, guard, initial, emit, adjacency, *,
                         archive=None, maxiter=400, maxeval=3000, threads=4,
                         wall_seconds=45 * 60, max_cg=6):
    """One run from ``initial``. Every ordinary objective call counts toward ``maxeval``."""
    point = np.array(initial, dtype=float, copy=True)
    if not guard.feasible(point):
        raise ValueError("initial lengths are not feasible")
    start = perf_counter()
    deadline = start + wall_seconds
    counts = dict(objective=0, gradient=0, diagonal=0, hessian_products=0,
                  line_search_trials=0, independent_objective=0, independent_gradient=0)
    accepted = 0
    stopped = "iteration_limit"
    trust = 1.
    checkpoint_meta = {}

    def evaluate(lengths, derivatives):
        if perf_counter() >= deadline:
            raise _Halt("wall_clock")
        if counts["objective"] + 1 > maxeval:
            raise _Halt("objective_budget")
        counts["objective"] += 1
        counts["gradient"] += int(derivatives)
        counts["diagonal"] += int(derivatives)
        return engine.evaluate(lengths, threads=threads, gradient=derivatives, diagonal=derivatives)

    def physical(lengths, derivatives):
        item = evaluate(lengths, derivatives)
        sites = len(item.site_log_likelihood)
        grad = None if item.length_gradient is None else -item.length_gradient / sites
        diag = None if item.negative_log_hessian_diagonal is None else item.negative_log_hessian_diagonal / sites
        return item, grad, diag, sites

    current, gradient, diagonal, n = physical(point, True)

    def gradient_at(lengths):
        if perf_counter() >= deadline:
            raise _Halt("wall_clock")
        if counts["objective"] + 1 > maxeval:
            raise _Halt("objective_budget")
        counts["objective"] += 1
        counts["gradient"] += 1
        counts["hessian_products"] += 1
        item = engine.evaluate(lengths, threads=threads, gradient=True, diagonal=False)
        return -item.length_gradient / n

    def independent(lengths):
        counts["independent_objective"] += 1
        counts["independent_gradient"] += 1
        item = auditor.evaluate(lengths, threads=threads, log_domain=True)
        return -item.length_gradient / n, item.log_likelihood

    objective_value = -current.log_likelihood / n
    best_projection = projected_norm(point, gradient)
    best_objective = objective_value

    def remember(name):
        if archive is None:
            return
        _positive, _active, mixed, _diag = chart(point, gradient, diagonal)
        checkpoint_meta[name] = save_checkpoint(
            archive, name, lengths=point, physical_gradient=gradient, hessian_diagonal=diagonal,
            mixed_gradient=mixed, objective=objective_value, projection=projected_norm(point, gradient),
            iteration=accepted, objective_calls=counts["objective"])

    emit({"event": "initial", "objective": objective_value, "projection": best_projection,
          "lbfgs_history": "discarded"})
    remember("initial")
    remember("best_projection")
    remember("best_objective")
    verification = verify_products(point, gradient, diagonal, adjacency, gradient_at)
    emit({"event": "hvp_verification", **{k: verification[k] for k in ("max_relative_error", "accepted")},
          "checked": verification["checked"]})
    if not verification["accepted"]:
        stopped = "hvp_verification_failed"
        remember("final")
        return _finish(point, current, gradient, stopped, accepted, counts, best_projection,
                       checkpoint_meta, start, guard, independent, verification, trust)

    try:
        while accepted < maxiter:
            projection = projected_norm(point, gradient)
            if projection <= 1e-5:
                stopped = "candidate_stationary"
                break
            if trust < 1e-3:
                stopped = "trust_collapsed"
                break
            index, threshold, seeds = free_edges(point, gradient, adjacency)
            if index.size == 0:
                stopped = "empty_free_set"
                break
            cap = trust * np.minimum(0.25 * point[index], 0.01)

            def product(direction):
                return hessian_product(point, direction, gradient_at)

            try:
                step, info = capped_diagonal_step(gradient, diagonal, product, index, cap)
            except FloatingPointError:
                trust *= 0.5
                emit({"event": "direction", "status": "infeasible_product", "trust": trust})
                continue
            full = np.zeros_like(point)
            full[index] = step
            slope = float(gradient @ full)
            emit({"event": "direction", "free_edges": int(index.size), "seeds": seeds,
                  "threshold": threshold, "trust": trust, "slope": slope, **info})
            if not math.isfinite(slope) or slope >= 0. or not np.isfinite(full).all():
                trust *= 0.5
                emit({"event": "trust_shrink", "trust": trust, "reasons": ["not_descent"],
                      "criterion": "directional derivative is not a strict decrease"})
                continue
            found = False
            reasons = []
            for attempt in range(6):
                alpha = math.ldexp(1., -attempt)
                trial = point.copy()
                trial[index] = point[index] + alpha * step
                counts["line_search_trials"] += 1
                row = {"event": "trial", "alpha": alpha, "slope": slope * alpha}
                if np.any(trial < 0) or not np.isfinite(trial).all() or float(gradient @ (trial - point)) >= 0:
                    reasons.append("invalid_step")
                    emit({**row, "status": "invalid_step"})
                    continue
                if not guard.feasible(trial):
                    reasons.append("zero_conflict")
                    emit({**row, "status": "zero_conflict"})
                    continue
                test, _grad, _diag, _sites = physical(trial, False)
                if not math.isfinite(test.log_likelihood):
                    reasons.append("nonfinite_likelihood")
                    emit({**row, "status": "nonfinite_likelihood"})
                    continue
                change = -math.fsum(test.site_log_likelihood - current.site_log_likelihood) / n
                # Armijo compares the actual objective change with c1 times the
                # predicted directional derivative. There is no second ratio gate.
                if change > ARMIJO_C1 * slope * alpha:
                    reasons.append("armijo")
                    emit({**row, "objective_change": change, "status": "rejected"})
                    continue
                following, new_gradient, new_diagonal, _sites = physical(trial, True)
                new_projection = projected_norm(trial, new_gradient)
                change = -math.fsum(following.site_log_likelihood - current.site_log_likelihood) / n
                if change > ARMIJO_C1 * slope * alpha:
                    reasons.append("armijo")
                    emit({**row, "objective_change": change, "projection": new_projection,
                          "status": "rejected"})
                    continue
                changed = int(np.count_nonzero(np.abs(trial - point) > 0))
                point, current, gradient, diagonal = trial, following, new_gradient, new_diagonal
                objective_value = -current.log_likelihood / n
                accepted += 1
                found = True
                remember(f"accepted-{accepted:04d}")
                if new_projection < best_projection:
                    best_projection = new_projection
                    remember("best_projection")
                if objective_value < best_objective:
                    best_objective = objective_value
                    remember("best_objective")
                emit({**row, "objective_change": change, "projection": new_projection,
                      "status": "accepted", "accepted": accepted, "objective": objective_value,
                      "changed_edges": changed,
                      "counts": counts.copy(), "elapsed_seconds": perf_counter() - start})
                trust = min(1., trust * 1.25)
                break
            if not found:
                trust *= 0.5
                emit({"event": "trust_shrink", "trust": trust, "reasons": reasons,
                      "criterion": "no trial was feasible, finite, and an Armijo decrease"})
    except _Halt as halt:
        stopped = halt.status

    remember("final")
    return _finish(point, current, gradient, stopped, accepted, counts, best_projection,
                   checkpoint_meta, start, guard, independent, verification, trust)


def _finish(point, current, gradient, stopped, accepted, counts, best_projection,
            checkpoint_meta, start, guard, independent, verification, trust):
    n = len(current.site_log_likelihood)
    audit, ll = independent(point)
    ordinary = projected_norm(point, gradient)
    checked = projected_norm(point, audit)
    discrepancy = np.abs(audit - gradient)
    feasible = bool(guard.feasible(point))
    return point, {
        "status": stopped,
        "method": "per-edge capped diagonal step; one Hessian-vector product sets a separate length on each edge",
        "lbfgs_history": "discarded",
        "accepted": accepted,
        "objective": -current.log_likelihood / n,
        "independent_objective": -ll / n,
        "projection": ordinary,
        "independent_projection": checked,
        "best_ordinary_projection": best_projection,
        "cross_path_max": float(discrepancy.max()),
        "cross_path_max_edge": int(discrepancy.argmax()),
        "zero_edges": int(np.count_nonzero(point == 0)),
        "feasible": feasible,
        "both_gates_passed": bool(feasible and max(ordinary, checked) <= 1e-5 and discrepancy.max() <= 1e-8),
        "trust": trust,
        "counts": counts,
        "elapsed_seconds": perf_counter() - start,
        "checkpoints": checkpoint_meta,
        "hvp_verification": verification,
    }

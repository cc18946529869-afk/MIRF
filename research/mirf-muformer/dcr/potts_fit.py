"""Numerical solver for one asymmetric L2 Potts conditional model.

The frozen point-estimate configuration is L-BFGS with a strong Wolfe line
search, a zero start, and the unnormalized weighted objective. Only a gradient
infinity norm at or below the optimality tolerance is a successful stop.
"""

from __future__ import annotations

import ctypes
import json
import math
import multiprocessing
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

Q = 21
LAMBDA_H = 0.01
LAMBDA_J_PRIME = 0.005
OPT_TOL = 1e-5
PROG_TOL = 1e-9
CORRECTIONS = 100
MAX_ITER = 500
MAX_FUN_EVALS = 1000
LINE_SEARCH_EVALS = 25
C1 = 1e-4
C2 = 0.9
OFFICIAL_M_EFF = 47899.73496814515
BOOTSTRAP_WEIGHT_TOTAL = 47900.0

_DIRECTORY = Path(__file__).resolve().parent
_SOURCE = _DIRECTORY / "conditional.c"
_LIBRARY = _DIRECTORY / "conditional_native.so"
_COMPILED: ctypes.CDLL | None = None


@dataclass(frozen=True)
class FitResult:
    """One site, including a stop that did not meet the gradient test."""

    status: str
    theta: np.ndarray
    h: np.ndarray
    couplings: np.ndarray
    objective: float
    nll: float
    penalty: float
    grad_inf: float
    iterations: int
    evaluations: int
    message: str

    @property
    def converged(self) -> bool:
        return self.status == "converged"


def parameter_count(length: int, q: int = Q) -> int:
    """Return the length of one site's parameter vector."""
    return q + q * q * (length - 1)


def unpack_parameters(theta: np.ndarray, length: int, q: int = Q) -> tuple[np.ndarray, np.ndarray]:
    """Split a vector into fields and directional couplings.

    Couplings use C order ``(predicted state, partner, partner state)``.
    Partners are the other columns, in increasing column index.
    """
    expected = parameter_count(length, q)
    if theta.shape != (expected,):
        raise ValueError(f"expected {expected} parameters, got {theta.shape}")
    fields = theta[:q]
    couplings = theta[q:].reshape(q, length - 1, q)
    return fields, couplings


def stopping_status(
    *,
    finite: bool,
    grad_inf: float,
    step_inf: float | None,
    function_change: float | None,
    direction_derivative: float | None,
    opt_tol: float,
    prog_tol: float,
    evaluations: int,
    max_fun_evals: int,
    iteration: int,
    max_iter: int,
) -> str | None:
    """Return a stop label, or None when the iteration may continue.

    The gradient test is decided before the progress test. A step or a
    function change below ``prog_tol`` does not become a successful stop.
    """
    if not finite:
        return "nonfinite"
    if grad_inf <= opt_tol:
        return "converged"
    if direction_derivative is not None and direction_derivative > -prog_tol:
        return "stagnation_direction"
    if step_inf is not None and step_inf <= prog_tol:
        return "stagnation_step"
    if function_change is not None and function_change < prog_tol:
        return "stagnation_function"
    if evaluations >= max_fun_evals:
        return "max_eval"
    if iteration >= max_iter:
        return "max_iter"
    return None


def _as_alignment(msa: np.ndarray, q: int) -> np.ndarray:
    alignment = np.ascontiguousarray(msa)
    if alignment.ndim != 2:
        raise ValueError("the alignment must be a matrix")
    if alignment.dtype != np.int8:
        if not np.issubdtype(alignment.dtype, np.integer):
            raise ValueError("alignment codes must be integers")
        if alignment.min() < -128 or alignment.max() > 127:
            raise ValueError("alignment codes do not fit in int8")
        alignment = np.ascontiguousarray(alignment, dtype=np.int8)
    if alignment.size and (int(alignment.min()) < 0 or int(alignment.max()) >= q):
        raise ValueError(f"alignment codes must lie in 0..{q - 1}")
    return alignment


def _as_weights(weights: np.ndarray, n_rows: int) -> np.ndarray:
    vector = np.ascontiguousarray(weights, dtype=np.float64)
    if vector.shape != (n_rows,):
        raise ValueError(f"expected {n_rows} weights, got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError("weights must be finite")
    return vector


def site_terms_numpy(
    theta: np.ndarray,
    msa: np.ndarray,
    weights: np.ndarray,
    site: int,
    weight_total: float,
    lambda_h: float = LAMBDA_H,
    lambda_j_prime: float = LAMBDA_J_PRIME,
    q: int = Q,
) -> tuple[float, float, np.ndarray]:
    """Return the negative log pseudolikelihood, the penalty, and the total gradient."""
    alignment = _as_alignment(msa, q)
    n_rows, length = alignment.shape
    if not 0 <= int(site) < length:
        raise ValueError("site is outside the alignment")
    vector = np.ascontiguousarray(theta, dtype=np.float64)
    fields, couplings = unpack_parameters(vector, length, q)
    row_weights = _as_weights(weights, n_rows)
    grad_h = np.zeros(q, dtype=np.float64)
    grad_j = np.zeros_like(couplings)
    nll = 0.0
    if n_rows:
        partners = np.concatenate((np.arange(site), np.arange(site + 1, length)))
        partner_codes = alignment[:, partners]
        logits = np.zeros((q, n_rows), dtype=np.float64)
        for partner, codes in enumerate(partner_codes.T):
            logits += couplings[:, partner, codes]
        logits += fields[:, None]
        maximum = np.max(logits, axis=0)
        shifted = np.exp(logits - maximum)
        normalizer = shifted.sum(axis=0)
        probabilities = shifted / normalizer
        observed = alignment[:, site]
        log_probability = logits[observed, np.arange(n_rows)] - (maximum + np.log(normalizer))
        nll = -float(np.dot(row_weights, log_probability))
        signed = probabilities * row_weights
        signed[observed, np.arange(n_rows)] -= row_weights
        grad_h = signed.sum(axis=1)
        for partner in range(length - 1):
            indicators = np.zeros((n_rows, q), dtype=np.float64)
            indicators[np.arange(n_rows), partner_codes[:, partner]] = 1.0
            grad_j[:, partner, :] = signed @ indicators
    penalty = float(weight_total) * (
        float(lambda_h) * float(np.dot(fields, fields))
        + float(lambda_j_prime) * float(np.dot(couplings.ravel(), couplings.ravel()))
    )
    grad_h = grad_h + (2.0 * float(weight_total) * float(lambda_h)) * fields
    grad_j = grad_j + (2.0 * float(weight_total) * float(lambda_j_prime)) * couplings
    gradient = np.concatenate((grad_h, grad_j.ravel()))
    return nll, penalty, gradient


def _compile_library() -> ctypes.CDLL:
    global _COMPILED
    if _COMPILED is not None and _LIBRARY.exists() and _LIBRARY.stat().st_mtime >= _SOURCE.stat().st_mtime:
        return _COMPILED
    subprocess.run(
        [
            "gcc",
            "-O3",
            "-fPIC",
            "-shared",
            "-Wall",
            "-Wextra",
            "-fopenmp",
            str(_SOURCE),
            "-o",
            str(_LIBRARY),
            "-lm",
        ],
        check=True,
    )
    library = ctypes.CDLL(str(_LIBRARY))
    library.site_objective.argtypes = [
        ctypes.POINTER(ctypes.c_int8),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    library.site_objective.restype = ctypes.c_int
    library.site_logit_stats.argtypes = [
        ctypes.POINTER(ctypes.c_int8),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_int),
    ]
    library.site_logit_stats.restype = ctypes.c_int
    _COMPILED = library
    return library


def set_objective_threads(threads: int) -> None:
    """Set the OpenMP team size used by the conditional objective."""
    ctypes.CDLL("libgomp.so.1").omp_set_num_threads(int(threads))


def site_terms_c(
    theta: np.ndarray,
    msa: np.ndarray,
    weights: np.ndarray,
    site: int,
    weight_total: float,
    lambda_h: float = LAMBDA_H,
    lambda_j_prime: float = LAMBDA_J_PRIME,
    q: int = Q,
) -> tuple[float, float, np.ndarray]:
    """Evaluate the same objective as ``site_terms_numpy`` through the C kernel."""
    alignment = _as_alignment(msa, q)
    n_rows, length = alignment.shape
    vector = np.ascontiguousarray(theta, dtype=np.float64)
    unpack_parameters(vector, length, q)
    row_weights = _as_weights(weights, n_rows)
    if not 0 <= site < length:
        raise ValueError("site is outside the alignment")
    gradient = np.zeros(parameter_count(length, q), dtype=np.float64)
    nll = ctypes.c_double()
    penalty = ctypes.c_double()
    library = _compile_library()
    code = library.site_objective(
        alignment.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        row_weights.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        vector.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        n_rows,
        length,
        q,
        site,
        float(weight_total),
        float(lambda_h),
        float(lambda_j_prime),
        ctypes.byref(nll),
        ctypes.byref(penalty),
        gradient.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if code != 0:
        raise RuntimeError("the conditional objective could not be evaluated")
    return float(nll.value), float(penalty.value), gradient


def site_logit_stats(
    theta: np.ndarray,
    msa: np.ndarray,
    site: int,
    q: int = Q,
) -> dict[str, float | int]:
    """Return logit range and softmax checks for one fitted site.

    Probabilities are the stable softmax used by the objective. The reported
    extremes are independent of the order in which rows are added.
    """
    alignment = _as_alignment(msa, q)
    n_rows, length = alignment.shape
    vector = np.ascontiguousarray(theta, dtype=np.float64)
    unpack_parameters(vector, length, q)
    if not 0 <= int(site) < length:
        raise ValueError("site is outside the alignment")
    logit_min = ctypes.c_double()
    logit_max = ctypes.c_double()
    max_row_range = ctypes.c_double()
    max_prob_sum_error = ctypes.c_double()
    min_probability = ctypes.c_double()
    nonfinite_rows = ctypes.c_int()
    library = _compile_library()
    code = library.site_logit_stats(
        alignment.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        vector.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        n_rows,
        length,
        q,
        int(site),
        ctypes.byref(logit_min),
        ctypes.byref(logit_max),
        ctypes.byref(max_row_range),
        ctypes.byref(max_prob_sum_error),
        ctypes.byref(min_probability),
        ctypes.byref(nonfinite_rows),
    )
    if code != 0:
        raise RuntimeError("logit statistics could not be evaluated")
    return {
        "logit_min": float(logit_min.value),
        "logit_max": float(logit_max.value),
        "max_row_range": float(max_row_range.value),
        "max_prob_sum_error": float(max_prob_sum_error.value),
        "min_probability": float(min_probability.value),
        "nonfinite_rows": int(nonfinite_rows.value),
    }


def site_terms(
    theta: np.ndarray,
    msa: np.ndarray,
    weights: np.ndarray,
    site: int,
    weight_total: float,
    lambda_h: float = LAMBDA_H,
    lambda_j_prime: float = LAMBDA_J_PRIME,
    q: int = Q,
) -> tuple[float, float, np.ndarray]:
    """Evaluate one conditional objective. The C kernel is the production path."""
    return site_terms_c(
        theta,
        msa,
        weights,
        site,
        weight_total,
        lambda_h,
        lambda_j_prime,
        q,
    )


def conditional_probabilities(
    msa: np.ndarray,
    fields: np.ndarray,
    couplings: np.ndarray,
    site: int,
) -> np.ndarray:
    """Return P(state at site | other sites) for every row. Rows sum to 1."""
    alignment = np.asarray(msa)
    q = fields.shape[0]
    n_rows, length = alignment.shape
    partners = np.concatenate((np.arange(site), np.arange(site + 1, length)))
    logits = np.array(fields, dtype=np.float64, copy=True)[:, None] * np.ones(n_rows)
    for partner, column in enumerate(partners):
        logits += couplings[:, partner, alignment[:, column]]
    maximum = np.max(logits, axis=0)
    shifted = np.exp(logits - maximum)
    return (shifted / shifted.sum(axis=0)).T


def _cubic_minimizer(left, f_left, slope_left, right, f_right, slope_right) -> float:
    if left == right or not np.isfinite(f_right):
        return 0.5 * (left + right)
    d1 = slope_left + slope_right - 3.0 * (f_left - f_right) / (left - right)
    discriminant = d1 * d1 - slope_left * slope_right
    if discriminant < 0.0:
        return 0.5 * (left + right)
    d2 = math.copysign(math.sqrt(discriminant), right - left)
    denominator = slope_right - slope_left + 2.0 * d2
    if denominator == 0.0:
        return 0.5 * (left + right)
    step = right - (right - left) * (slope_right + d2 - d1) / denominator
    if not np.isfinite(step) or (step - left) * (step - right) >= 0.0:
        return 0.5 * (left + right)
    return step


def _evaluate_step(objective, position, direction, alpha: float):
    trial = position + alpha * direction
    value, gradient = objective(trial)
    if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
        return None
    return float(value), gradient


def _note(events, **payload) -> None:
    if events is not None:
        payload.setdefault("event", payload.get("stage", "line_search"))
        events.append({key: (float(value) if isinstance(value, float) else value) for key, value in payload.items()})


def _zoom(objective, position, direction, low, high, f_low, f_high, g_low, g_high, f0, slope0, c1, c2, evaluations, max_evals, events=None):
    while evaluations < max_evals:
        slope_low = float(np.dot(g_low, direction))
        slope_high = float(np.dot(g_high, direction)) if g_high is not None else None
        if slope_high is None or not np.isfinite(f_high):
            alpha = 0.5 * (low + high)
        else:
            alpha = _cubic_minimizer(low, f_low, slope_low, high, f_high, slope_high)
        width = abs(high - low)
        if width == 0.0 or not np.isfinite(alpha):
            return None
        if min(abs(alpha - low), abs(alpha - high)) < 0.05 * width:
            alpha = 0.5 * (low + high)
        probed = _evaluate_step(objective, position, direction, alpha)
        evaluations += 1
        if probed is None or probed[0] > f0 + c1 * alpha * slope0 or probed[0] >= f_low:
            _note(
                events,
                stage="zoom_trial",
                alpha=float(alpha),
                finite=probed is not None,
                armijo=False if probed is None else bool(probed[0] <= f0 + c1 * alpha * slope0),
                reason="nonfinite" if probed is None else "armijo_or_bracket",
            )
            high = alpha
            f_high = math.inf if probed is None else probed[0]
            g_high = None if probed is None else probed[1]
            continue
        value, gradient = probed
        slope = float(np.dot(gradient, direction))
        if abs(slope) <= c2 * abs(slope0):
            _note(events, stage="zoom_trial", alpha=float(alpha), finite=True, armijo=True, curvature=True, reason="accepted")
            return alpha, value, gradient, evaluations
        if slope * (high - low) >= 0.0:
            high = low
            f_high = f_low
            g_high = g_low
        low = alpha
        f_low = value
        g_low = gradient
        _note(events, stage="zoom_trial", alpha=float(alpha), finite=True, armijo=True, curvature=False, slope=slope, reason="curvature")
    _note(events, stage="zoom", reason="budget")
    return None


def strong_wolfe(objective, position, f0, g0, direction, alpha0, c1=C1, c2=C2, max_evals=LINE_SEARCH_EVALS, events=None):
    """Return (alpha, value, gradient, evaluations) or None when no Wolfe point is found."""
    slope0 = float(np.dot(g0, direction))
    if not np.isfinite(slope0) or not np.isfinite(alpha0) or alpha0 <= 0.0:
        _note(
            events,
            stage="line_search",
            reason="invalid_slope_or_step",
            slope0=float(slope0) if np.isfinite(slope0) else math.nan,
            alpha0=float(alpha0) if np.isfinite(alpha0) else math.nan,
        )
        return None
    previous = 0.0
    f_previous = f0
    g_previous = g0
    alpha = float(alpha0)
    finite_trials = 0
    evaluations = 0
    while evaluations < max_evals:
        if alpha > 1e8:
            _note(events, stage="line_search", reason="alpha_cap", alpha=float(alpha))
            return None
        probed = _evaluate_step(objective, position, direction, alpha)
        evaluations += 1
        if probed is None:
            _note(events, stage="trial", alpha=float(alpha), finite=False, reason="nonfinite")
            alpha *= 0.5
            if alpha <= np.finfo(np.float64).tiny:
                _note(events, stage="line_search", reason="nonfinite_step_collapsed")
                return None
            continue
        value, gradient = probed
        slope = float(np.dot(gradient, direction))
        armijo = bool(value <= f0 + c1 * alpha * slope0)
        curvature = bool(abs(slope) <= c2 * abs(slope0))
        _note(
            events,
            stage="trial",
            alpha=float(alpha),
            finite=True,
            f=float(value),
            slope=slope,
            armijo=armijo,
            curvature=curvature,
        )
        if (not armijo) or (finite_trials > 0 and value >= f_previous):
            _note(events, stage="zoom_enter", reason="armijo" if not armijo else "previous_value")
            found = _zoom(
                objective,
                position,
                direction,
                previous,
                alpha,
                f_previous,
                value,
                g_previous,
                gradient,
                f0,
                slope0,
                c1,
                c2,
                evaluations,
                max_evals,
                events,
            )
            if found is None:
                _note(events, stage="line_search", reason="zoom_failed")
            return found
        if curvature:
            return alpha, value, gradient, evaluations
        if slope >= 0.0:
            _note(events, stage="zoom_enter", reason="nonnegative_slope")
            found = _zoom(
                objective,
                position,
                direction,
                alpha,
                previous,
                value,
                f_previous,
                gradient,
                g_previous,
                f0,
                slope0,
                c1,
                c2,
                evaluations,
                max_evals,
                events,
            )
            if found is None:
                _note(events, stage="line_search", reason="zoom_failed")
            return found
        previous = alpha
        f_previous = value
        g_previous = gradient
        finite_trials += 1
        _note(events, stage="expand", alpha_next=float(alpha * 2.0), slope=slope, reason="curvature")
        alpha *= 2.0
    _note(events, stage="line_search", reason="evaluation_budget")
    return None


def _two_loop(gradient: np.ndarray, steps: list[np.ndarray], changes: list[np.ndarray], curvatures: list[float]) -> np.ndarray:
    if not steps:
        return -gradient
    q_vector = gradient.copy()
    alphas = []
    for step, change, curvature in zip(reversed(steps), reversed(changes), reversed(curvatures)):
        alpha = float(np.dot(step, q_vector)) / curvature
        q_vector -= alpha * change
        alphas.append(alpha)
    latest_step = steps[-1]
    latest_change = changes[-1]
    gamma = curvatures[-1] / float(np.dot(latest_change, latest_change))
    direction = gamma * q_vector
    for step, change, curvature, alpha in zip(steps, changes, curvatures, reversed(alphas)):
        beta = float(np.dot(change, direction)) / curvature
        direction += step * (alpha - beta)
    return -direction


@dataclass(frozen=True)
class OptimizerResult:
    status: str
    x: np.ndarray
    f: float
    grad_inf: float
    iterations: int
    evaluations: int
    message: str
    trace: tuple = ()


def lbfgs(
    objective,
    x0: np.ndarray,
    *,
    corrections: int = CORRECTIONS,
    opt_tol: float = OPT_TOL,
    prog_tol: float = PROG_TOL,
    max_iter: int = MAX_ITER,
    max_fun_evals: int = MAX_FUN_EVALS,
    c1: float = C1,
    c2: float = C2,
) -> OptimizerResult:
    """Minimize with limited-memory BFGS and a strong Wolfe line search.

    The first trial step is ``min(1, 1 / ||g||_1)``. Later trial steps start at 1.
    Those lengths come from minFunc's L-BFGS initialization. They are not
    optimality tolerances. ``corrections`` is the number of stored vector pairs.
    """
    position = np.array(x0, dtype=np.float64, copy=True)
    value, gradient = objective(position)
    evaluations = 1
    iterations = 0
    if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
        return OptimizerResult("nonfinite", position, float(value), math.inf, 0, evaluations, "non-finite value at the start")
    grad_inf = float(np.max(np.abs(gradient)))
    status = stopping_status(
        finite=True,
        grad_inf=grad_inf,
        step_inf=None,
        function_change=None,
        direction_derivative=None,
        opt_tol=opt_tol,
        prog_tol=prog_tol,
        evaluations=evaluations,
        max_fun_evals=max_fun_evals,
        iteration=iterations,
        max_iter=max_iter,
    )
    if status == "converged":
        return OptimizerResult(status, position, float(value), grad_inf, 0, evaluations, "gradient infinity norm at the start")
    steps: list[np.ndarray] = []
    changes: list[np.ndarray] = []
    curvatures: list[float] = []
    trace: list[dict] = []
    while True:
        direction = _two_loop(gradient, steps, changes, curvatures)
        slope = float(np.dot(gradient, direction))
        direction_inf = float(np.max(np.abs(direction))) if direction.size else 0.0
        if not np.all(np.isfinite(direction)) or not np.isfinite(slope):
            trace.append({"event": "stop", "reason": "nonfinite_direction", "iteration": iterations})
            return OptimizerResult("nonfinite", position, float(value), grad_inf, iterations, evaluations, "non-finite search direction", tuple(trace))
        gradient_mass = float(np.sum(np.abs(gradient)))
        alpha0 = min(1.0, 1.0 / gradient_mass) if iterations == 0 else 1.0
        trace.append(
            {
                "event": "direction",
                "iteration": iterations,
                "gtd": slope,
                "grad_inf": grad_inf,
                "direction_inf": direction_inf,
                "alpha0": alpha0,
                "descent": bool(slope < 0.0),
            }
        )
        if slope > -prog_tol:
            trace.append({"event": "stop", "reason": "stagnation_direction", "iteration": iterations, "gtd": slope})
            return OptimizerResult("stagnation_direction", position, float(value), grad_inf, iterations, evaluations, "directional derivative reached progTol", tuple(trace))
        found = strong_wolfe(objective, position, float(value), gradient, direction, alpha0, c1, c2, LINE_SEARCH_EVALS, trace)
        if found is None:
            trace.append({"event": "stop", "reason": "line_search", "iteration": iterations, "gtd": slope})
            return OptimizerResult("line_search", position, float(value), grad_inf, iterations, evaluations, "strong Wolfe line search failed", tuple(trace))
        alpha, new_value, new_gradient, used = found
        evaluations += used
        step = alpha * direction
        step_inf = float(np.max(np.abs(step)))
        change = abs(new_value - float(value))
        new_grad_inf = float(np.max(np.abs(new_gradient)))
        finite = bool(np.isfinite(new_value) and np.all(np.isfinite(new_gradient)) and np.isfinite(step_inf))
        iterations += 1
        status = stopping_status(
            finite=finite,
            grad_inf=new_grad_inf,
            step_inf=step_inf,
            function_change=change,
            direction_derivative=None,
            opt_tol=opt_tol,
            prog_tol=prog_tol,
            evaluations=evaluations,
            max_fun_evals=max_fun_evals,
            iteration=iterations,
            max_iter=max_iter,
        )
        y_change = new_gradient - gradient
        curvature = float(np.dot(y_change, step))
        trace.append(
            {
                "event": "accepted",
                "iteration": iterations,
                "alpha": float(alpha),
                "step_inf": step_inf,
                "function_change": float(change),
                "curvature_sy": curvature,
                "curvature_positive": bool(curvature > 0.0),
                "new_grad_inf": new_grad_inf,
                "stop": status,
            }
        )
        if finite and curvature > 0.0:
            steps.append(step.copy())
            changes.append(y_change)
            curvatures.append(curvature)
            if len(steps) > corrections:
                del steps[0]
                del changes[0]
                del curvatures[0]
        position = position + step
        value = new_value
        gradient = new_gradient
        grad_inf = new_grad_inf
        if status is not None:
            messages = {
                "converged": "gradient infinity norm reached optTol",
                "stagnation_step": "parameter step reached progTol before the gradient test",
                "stagnation_function": "objective change reached progTol before the gradient test",
                "nonfinite": "non-finite value during the iteration",
                "max_eval": "function-evaluation limit reached",
                "max_iter": "iteration limit reached",
            }
            return OptimizerResult(status, position, float(value), grad_inf, iterations, evaluations, messages[status], tuple(trace))


def fit_site(
    msa: np.ndarray,
    weights: np.ndarray,
    site: int,
    weight_total: float,
    *,
    lambda_h: float = LAMBDA_H,
    lambda_j_prime: float = LAMBDA_J_PRIME,
    q: int = Q,
    opt_tol: float = OPT_TOL,
    prog_tol: float = PROG_TOL,
    corrections: int = CORRECTIONS,
    max_iter: int = MAX_ITER,
    max_fun_evals: int = MAX_FUN_EVALS,
) -> FitResult:
    """Fit one site from the zero vector. ``weight_total`` scales only the penalty."""
    alignment = _as_alignment(msa, q)
    length = alignment.shape[1]
    if not 0 <= int(site) < length:
        raise ValueError("site is outside the alignment")
    row_weights = _as_weights(weights, alignment.shape[0])
    if not np.isfinite(weight_total):
        raise ValueError("weight_total must be finite")
    theta0 = np.zeros(parameter_count(length, q), dtype=np.float64)

    def objective(theta: np.ndarray):
        nll, penalty, gradient = site_terms(
            theta,
            alignment,
            row_weights,
            int(site),
            float(weight_total),
            lambda_h,
            lambda_j_prime,
            q,
        )
        return nll + penalty, gradient

    solved = lbfgs(
        objective,
        theta0,
        corrections=corrections,
        opt_tol=opt_tol,
        prog_tol=prog_tol,
        max_iter=max_iter,
        max_fun_evals=max_fun_evals,
    )
    nll, penalty, _gradient = site_terms(
        solved.x,
        alignment,
        row_weights,
        int(site),
        float(weight_total),
        lambda_h,
        lambda_j_prime,
        q,
    )
    fields, couplings = unpack_parameters(solved.x, length, q)
    return FitResult(
        status=solved.status,
        theta=solved.x,
        h=fields.copy(),
        couplings=couplings.copy(),
        objective=float(nll + penalty),
        nll=nll,
        penalty=penalty,
        grad_inf=solved.grad_inf,
        iterations=solved.iterations,
        evaluations=solved.evaluations,
        message=solved.message,
    )


def load_official_alignment(prep_dir: Path | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    """Load the 209644 by 263 matrix and weights 1/m_i.

    The returned total is the float64 sum of those weights. It is checked
    against the recorded official value.
    """
    directory = Path(__file__).resolve().parents[1] / "prep" if prep_dir is None else Path(prep_dir)
    alignment = np.load(directory / "mature_codes.npy")
    counts = np.loadtxt(directory / "neighbor_counts.tsv", skiprows=1, usecols=1, dtype=np.int64)
    if alignment.shape != (counts.shape[0], 263):
        raise ValueError(f"alignment shape {alignment.shape} does not match {counts.shape[0]} by 263")
    weights = np.float64(1.0) / counts
    total = float(weights.sum())
    if total != OFFICIAL_M_EFF:
        raise ValueError(f"weight sum {total!r} is not the official M_eff")
    return np.ascontiguousarray(alignment), np.ascontiguousarray(weights), total


_OFFICIAL: tuple[np.ndarray, np.ndarray, float] | None = None


def _worker_init() -> None:
    ctypes.CDLL("libgomp.so.1").omp_set_num_threads(int(os.environ["POTTS_THREADS"]))
    global _OFFICIAL
    _OFFICIAL = load_official_alignment()


def _fit_official_site(site: int) -> dict[str, float | int | str]:
    if _OFFICIAL is None:
        raise RuntimeError("the worker alignment was not loaded")
    alignment, weights, total = _OFFICIAL
    output = Path(os.environ["POTTS_OUTPUT"])
    started = time.perf_counter()
    result = fit_site(alignment, weights, site, total)
    elapsed = time.perf_counter() - started
    np.savez(
        output / f"site_{site:03d}.npz",
        h=result.h,
        couplings=result.couplings,
        status=np.array(result.status),
        objective=np.float64(result.objective),
        nll=np.float64(result.nll),
        penalty=np.float64(result.penalty),
        grad_inf=np.float64(result.grad_inf),
        iterations=np.int32(result.iterations),
        evaluations=np.int32(result.evaluations),
        message=np.array(result.message),
        seconds=np.float64(elapsed),
    )
    return {
        "site": site,
        "status": result.status,
        "converged": result.converged,
        "grad_inf": result.grad_inf,
        "objective": result.objective,
        "nll": result.nll,
        "penalty": result.penalty,
        "iterations": result.iterations,
        "evaluations": result.evaluations,
        "seconds": elapsed,
        "message": result.message,
    }


def run_official_point_estimate(output_dir: Path, workers: int = 12, threads: int = 4) -> None:
    """Fit all 263 sites. A progTol stop is saved and is not counted as success."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    os.environ["POTTS_THREADS"] = str(threads)
    os.environ["POTTS_OUTPUT"] = str(output)
    pending = [site for site in range(263) if not (output / f"site_{site:03d}.npz").exists()]
    summary_path = output / "summary.json"
    existing = []
    if summary_path.exists():
        existing = json.loads(summary_path.read_text())["sites"]
    by_site = {row["site"]: row for row in existing}
    context = multiprocessing.get_context("spawn")
    with context.Pool(workers, initializer=_worker_init) as pool:
        for row in pool.imap_unordered(_fit_official_site, pending, chunksize=1):
            by_site[int(row["site"])] = row
            rows = [by_site[site] for site in sorted(by_site)]
            counts: dict[str, int] = {}
            for item in rows:
                counts[str(item["status"])] = counts.get(str(item["status"]), 0) + 1
            summary_path.write_text(
                json.dumps(
                    {
                        "weight_total": OFFICIAL_M_EFF,
                        "lambda_h": LAMBDA_H,
                        "lambda_j_prime": LAMBDA_J_PRIME,
                        "opt_tol": OPT_TOL,
                        "prog_tol": PROG_TOL,
                        "corrections": CORRECTIONS,
                        "finished_sites": len(rows),
                        "converged_sites": sum(1 for item in rows if item["converged"]),
                        "status_counts": counts,
                        "sites": rows,
                    },
                    indent=2,
                )
            )
            print(
                f"site {int(row['site']):3d}  {row['status']}  grad {row['grad_inf']:.3e}  "
                f"{len(rows)}/263",
                flush=True,
            )


if __name__ == "__main__":
    run_official_point_estimate(_DIRECTORY / "point_estimate")

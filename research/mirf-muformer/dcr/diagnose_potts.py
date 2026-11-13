"""Diagnostics for the full-MSA Potts solve. They do not change the model or the tolerances."""

from __future__ import annotations

import json
import multiprocessing
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import potts_fit

OUTPUT = Path(__file__).resolve().parent / "point_estimate"
DIAGNOSTICS = OUTPUT / "diagnostics"
LINE_SEARCH_SITES = (12, 70, 89, 90, 94, 103, 133, 169, 172, 202, 229)
FUNCTION_SITE = 208
MEDIAN_SITE = 256
MAXIMUM_SITE = 95
SERIAL_SITES = (LINE_SEARCH_SITES[0], FUNCTION_SITE)
EPSILONS = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8)


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not np.isfinite(number):
            return None
        return number
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2) + "\n")


def _saved_theta(site: int) -> np.ndarray:
    blob = np.load(OUTPUT / f"site_{site:03d}.npz")
    return np.concatenate((np.asarray(blob["h"], dtype=np.float64), np.asarray(blob["couplings"], dtype=np.float64).ravel()))


def _objective(alignment, weights, total, site):
    def objective(theta: np.ndarray):
        nll, penalty, gradient = potts_fit.site_terms(theta, alignment, weights, site, total)
        return nll + penalty, gradient

    return objective


def penalty_split() -> None:
    rows = []
    for site in range(263):
        blob = np.load(OUTPUT / f"site_{site:03d}.npz")
        fields = np.asarray(blob["h"], dtype=np.float64)
        couplings = np.asarray(blob["couplings"], dtype=np.float64)
        field_penalty = potts_fit.OFFICIAL_M_EFF * potts_fit.LAMBDA_H * float(np.dot(fields, fields))
        coupling_penalty = potts_fit.OFFICIAL_M_EFF * potts_fit.LAMBDA_J_PRIME * float(np.dot(couplings.ravel(), couplings.ravel()))
        saved = float(blob["penalty"])
        rows.append(
            {
                "site": site,
                "status": str(blob["status"]),
                "nll": float(blob["nll"]),
                "field_penalty": field_penalty,
                "coupling_penalty": coupling_penalty,
                "penalty_saved": saved,
                "penalty_gap": saved - (field_penalty + coupling_penalty),
                "max_abs_h": float(np.max(np.abs(fields))),
                "max_abs_J": float(np.max(np.abs(couplings))),
                "grad_inf": float(blob["grad_inf"]),
                "objective_finite": bool(np.isfinite(blob["objective"])),
            }
        )
    _write(DIAGNOSTICS / "penalty_split.json", rows)
    gaps = np.array([row["penalty_gap"] for row in rows])
    print(
        f"penalty split max |gap| {np.max(np.abs(gaps)):.3e}  "
        f"max |h| {max(row['max_abs_h'] for row in rows):.3e}  "
        f"max |J| {max(row['max_abs_J'] for row in rows):.3e}",
        flush=True,
    )


def logit_scan() -> None:
    potts_fit.set_objective_threads(16)
    alignment, _weights, _total = potts_fit.load_official_alignment()
    rows = []
    for site in range(263):
        theta = _saved_theta(site)
        stats = potts_fit.site_logit_stats(theta, alignment, site)
        stats["site"] = site
        rows.append(stats)
        if site % 20 == 0:
            print(f"logit {site}", flush=True)
    _write(DIAGNOSTICS / "logit_stats.json", rows)
    print(
        "logit range max",
        max(row["max_row_range"] for row in rows),
        "nonfinite",
        sum(row["nonfinite_rows"] for row in rows),
        "max prob error",
        max(row["max_prob_sum_error"] for row in rows),
        "min probability",
        min(row["min_probability"] for row in rows),
        flush=True,
    )


def _directions(gradient: np.ndarray, seed: int) -> list[tuple[str, np.ndarray]]:
    generator = np.random.default_rng(seed)
    gradient_norm = float(np.linalg.norm(gradient))
    coordinate = np.zeros_like(gradient)
    coordinate[int(np.argmax(np.abs(gradient)))] = 1.0
    random = generator.normal(size=gradient.shape)
    random /= np.linalg.norm(random)
    return [
        ("gradient", gradient / gradient_norm),
        ("max_coordinate", coordinate),
        ("random", random),
    ]


def directional_derivatives() -> None:
    potts_fit.set_objective_threads(1)
    alignment, weights, total = potts_fit.load_official_alignment()
    reports = []
    for label, site in (("minimum", FUNCTION_SITE), ("median", MEDIAN_SITE), ("maximum", MAXIMUM_SITE)):
        theta = _saved_theta(site)
        objective = _objective(alignment, weights, total, site)
        value, gradient = objective(theta)
        value_again, gradient_again = objective(theta)
        site_report = {
            "label": label,
            "site": site,
            "objective": value,
            "grad_inf": float(np.max(np.abs(gradient))),
            "objective_repeat_gap": value_again - value,
            "gradient_repeat_max_abs": float(np.max(np.abs(gradient_again - gradient))),
            "directions": [],
        }
        for name, direction in _directions(gradient, site):
            directional = float(np.dot(gradient, direction))
            steps = []
            for epsilon in EPSILONS:
                forward = objective(theta + epsilon * direction)[0]
                backward = objective(theta - epsilon * direction)[0]
                estimate = (forward - backward) / (2.0 * epsilon)
                steps.append(
                    {
                        "epsilon": epsilon,
                        "directional_difference": estimate,
                        "absolute_error": estimate - directional,
                        "relative_error": (estimate - directional) / directional if directional != 0.0 else None,
                    }
                )
            site_report["directions"].append(
                {
                    "name": name,
                    "analytic": directional,
                    "steps": steps,
                }
            )
        reports.append(site_report)
        best = min(abs(step["absolute_error"]) for item in site_report["directions"] if item["name"] == "gradient" for step in item["steps"])
        print(f"fd site {site} best |error| along gradient {best:.3e}", flush=True)
    _write(DIAGNOSTICS / "directional_derivatives.json", reports)


def _fit_from_zero(site: int, threads: int) -> dict:
    potts_fit.set_objective_threads(threads)
    alignment, weights, total = potts_fit.load_official_alignment()
    solved = potts_fit.lbfgs(
        _objective(alignment, weights, total, site),
        np.zeros(potts_fit.parameter_count(alignment.shape[1]), dtype=np.float64),
    )
    saved = _saved_theta(site)
    blob = np.load(OUTPUT / f"site_{site:03d}.npz")
    return {
        "site": site,
        "threads": threads,
        "status": solved.status,
        "saved_status": str(blob["status"]),
        "grad_inf": solved.grad_inf,
        "saved_grad_inf": float(blob["grad_inf"]),
        "objective": solved.f,
        "saved_objective": float(blob["objective"]),
        "iterations": solved.iterations,
        "evaluations": solved.evaluations,
        "message": solved.message,
        "max_abs_parameter_difference": float(np.max(np.abs(solved.x - saved))),
        "trace": list(solved.trace),
    }


def _retrace_worker(site: int) -> int:
    os.environ["OMP_NUM_THREADS"] = "4"
    report = _fit_from_zero(site, 4)
    _write(DIAGNOSTICS / "line_search" / f"site_{site:03d}.json", report)
    print(f"retrace {site} {report['status']} saved {report['saved_status']} grad {report['grad_inf']:.3e}", flush=True)
    return site


def retrace_line_search() -> None:
    DIAGNOSTICS.joinpath("line_search").mkdir(parents=True, exist_ok=True)
    context = multiprocessing.get_context("spawn")
    with context.Pool(len(LINE_SEARCH_SITES)) as pool:
        pool.map(_retrace_worker, LINE_SEARCH_SITES, chunksize=1)


def _serial_worker(site: int) -> int:
    os.environ["OMP_NUM_THREADS"] = "1"
    report = _fit_from_zero(site, 1)
    _write(DIAGNOSTICS / "serial" / f"site_{site:03d}.json", report)
    print(
        f"serial {site} {report['status']} saved {report['saved_status']} "
        f"param {report['max_abs_parameter_difference']:.3e}",
        flush=True,
    )
    return site


def serial_refits() -> None:
    DIAGNOSTICS.joinpath("serial").mkdir(parents=True, exist_ok=True)
    context = multiprocessing.get_context("spawn")
    with context.Pool(len(SERIAL_SITES)) as pool:
        pool.map(_serial_worker, SERIAL_SITES, chunksize=1)


def thread_point_comparison() -> None:
    alignment, weights, total = potts_fit.load_official_alignment()
    rows = []
    for site in (FUNCTION_SITE, MEDIAN_SITE, MAXIMUM_SITE, LINE_SEARCH_SITES[0]):
        theta = _saved_theta(site)
        potts_fit.set_objective_threads(1)
        first = potts_fit.site_terms(theta, alignment, weights, site, total)
        second = potts_fit.site_terms(theta, alignment, weights, site, total)
        potts_fit.set_objective_threads(4)
        parallel = potts_fit.site_terms(theta, alignment, weights, site, total)
        rows.append(
            {
                "site": site,
                "serial_repeat_nll_gap": first[0] - second[0],
                "serial_repeat_gradient_max_abs": float(np.max(np.abs(first[2] - second[2]))),
                "serial_minus_four_thread_nll": first[0] - parallel[0],
                "serial_minus_four_thread_gradient_max_abs": float(np.max(np.abs(first[2] - parallel[2]))),
                "serial_grad_inf": float(np.max(np.abs(first[2]))),
                "four_thread_grad_inf": float(np.max(np.abs(parallel[2]))),
            }
        )
        print(f"thread compare site {site} nll gap {rows[-1]['serial_minus_four_thread_nll']:.3e}", flush=True)
    _write(DIAGNOSTICS / "thread_comparison.json", rows)


def main() -> None:
    phase = sys.argv[1] if len(sys.argv) > 1 else "prepare"
    if phase in ("prepare", "all"):
        penalty_split()
        logit_scan()
        thread_point_comparison()
        directional_derivatives()
    if phase in ("retrace", "all"):
        retrace_line_search()
    if phase in ("serial", "all"):
        serial_refits()


if __name__ == "__main__":
    main()

"""Compare the original Potts kernel with a transposed working copy.

Version A compiles the production source. Version B uses the transposed
working layout and the same compiler flags. Version C adds -march=native to
that transposed source only. External parameter order is unchanged.
"""

from __future__ import annotations

import ctypes
import json
import os
import subprocess
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT.parent / "prep"
SOURCE_A = ROOT / "conditional.c"
SOURCE_T = Path(__file__).resolve().parent / "conditional_transpose.c"
BUILD = Path("/tmp/potts_layout_bench")
Q = 21
SITES = 263
WORKERS = 40
EVALUATIONS = 111.72
ITERATIONS = 102.70

BASE_FLAGS = [
    "gcc",
    "-O3",
    "-fPIC",
    "-shared",
    "-Wall",
    "-Wextra",
    "-fopenmp",
]


def compile_library(source: Path, output: Path, native: bool) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    command = list(BASE_FLAGS)
    if native:
        command.append("-march=native")
    command.extend([str(source), "-o", str(output), "-lm"])
    subprocess.run(command, check=True)


def bind(library: ctypes.CDLL) -> None:
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
    library.site_direction_cache.argtypes = [
        ctypes.POINTER(ctypes.c_int8),
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    library.site_direction_cache.restype = ctypes.c_int


def set_threads(count: int) -> None:
    os.environ["OMP_NUM_THREADS"] = str(count)
    ctypes.CDLL("libgomp.so.1").omp_set_num_threads(int(count))


def call_objective(library, msa, weights, theta, site, total, grad):
    nll = ctypes.c_double()
    penalty = ctypes.c_double()
    code = library.site_objective(
        msa.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        weights.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        theta.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        msa.shape[0],
        msa.shape[1],
        Q,
        int(site),
        float(total),
        0.01,
        0.005,
        ctypes.byref(nll),
        ctypes.byref(penalty),
        grad.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if code != 0:
        raise RuntimeError(f"site_objective failed for site {site}")
    return float(nll.value), float(penalty.value)


def call_cache(library, msa, theta, direction, site, probability, velocity):
    code = library.site_direction_cache(
        msa.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
        theta.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        direction.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        msa.shape[0],
        msa.shape[1],
        Q,
        int(site),
        probability.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        velocity.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
    )
    if code != 0:
        raise RuntimeError(f"site_direction_cache failed for site {site}")


def max_abs(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(left - right)))


def check_pair(reference, candidate, msa, weights, theta, direction, total, sites) -> dict:
    set_threads(1)
    report = []
    for site in sites:
        grad_a = np.empty(theta.size, dtype=np.float64)
        grad_b = np.empty(theta.size, dtype=np.float64)
        nll_a, penalty_a = call_objective(reference, msa, weights, theta, site, total, grad_a)
        nll_b, penalty_b = call_objective(candidate, msa, weights, theta, site, total, grad_b)
        probability_a = np.empty((msa.shape[0], Q), dtype=np.float64)
        velocity_a = np.empty_like(probability_a)
        probability_b = np.empty_like(probability_a)
        velocity_b = np.empty_like(probability_a)
        call_cache(reference, msa, theta, direction, site, probability_a, velocity_a)
        call_cache(candidate, msa, theta, direction, site, probability_b, velocity_b)
        report.append(
            {
                "site": int(site),
                "nll_abs": abs(nll_a - nll_b),
                "penalty_abs": abs(penalty_a - penalty_b),
                "gradient_abs": max_abs(grad_a, grad_b),
                "probability_abs": max_abs(probability_a, probability_b),
                "velocity_abs": max_abs(velocity_a, velocity_b),
            }
        )
    return {
        "nll_abs": max(row["nll_abs"] for row in report),
        "penalty_abs": max(row["penalty_abs"] for row in report),
        "gradient_abs": max(row["gradient_abs"] for row in report),
        "probability_abs": max(row["probability_abs"] for row in report),
        "velocity_abs": max(row["velocity_abs"] for row in report),
        "sites": report,
    }


def time_once(function) -> float:
    started = time.perf_counter()
    function()
    return time.perf_counter() - started


def single_process(library, msa, weights, theta, direction, total) -> dict:
    set_threads(1)
    grad = np.empty(theta.size, dtype=np.float64)
    probability = np.empty((msa.shape[0], Q), dtype=np.float64)
    velocity = np.empty_like(probability)

    def objective():
        call_objective(library, msa, weights, theta, 0, total, grad)

    def cache():
        call_cache(library, msa, theta, direction, 0, probability, velocity)

    def both():
        objective()
        cache()

    objective()
    cache()
    samples = {
        "objective": [time_once(objective) for _ in range(5)],
        "direction_cache": [time_once(cache) for _ in range(5)],
        "both": [time_once(both) for _ in range(5)],
    }
    return {name: {"mean": float(np.mean(values)), "min": float(np.min(values))} for name, values in samples.items()}


SHARED: dict[str, object] = {}


def worker_job(site: int) -> float:
    os.environ["OMP_NUM_THREADS"] = "1"
    ctypes.CDLL("libgomp.so.1").omp_set_num_threads(1)
    active = SHARED["library"]
    kind = SHARED["kind"]
    msa = SHARED["msa"]
    weights = SHARED["weights"]
    theta = SHARED["theta"]
    direction = SHARED["direction"]
    total = SHARED["total"]
    chosen = int(site) if SHARED["vary_site"] else 0
    started = time.perf_counter()
    if kind in {"objective", "both"}:
        grad = np.empty(theta.size, dtype=np.float64)
        call_objective(active, msa, weights, theta, chosen, total, grad)
    if kind in {"direction_cache", "both"}:
        probability = np.empty((msa.shape[0], Q), dtype=np.float64)
        velocity = np.empty_like(probability)
        call_cache(active, msa, theta, direction, chosen, probability, velocity)
    return time.perf_counter() - started


def concurrent(library: ctypes.CDLL, kind: str, msa, weights, theta, direction, total, vary_site: bool = False) -> dict:
    set_threads(1)
    SHARED["library"] = library
    SHARED["kind"] = kind
    SHARED["msa"] = msa
    SHARED["weights"] = weights
    SHARED["theta"] = theta
    SHARED["direction"] = direction
    SHARED["total"] = total
    SHARED["vary_site"] = vary_site
    context = get_context("fork")
    started = time.perf_counter()
    with context.Pool(WORKERS) as pool:
        samples = pool.map(worker_job, list(range(WORKERS)))
    wall = time.perf_counter() - started
    values = np.array(samples, dtype=np.float64)
    return {
        "wall": float(wall),
        "worker_mean": float(values.mean()),
        "worker_min": float(values.min()),
        "worker_max": float(values.max()),
    }


def project_hours(objective_seconds: float, cache_seconds: float) -> dict:
    per_site = EVALUATIONS * objective_seconds + ITERATIONS * cache_seconds
    one_fit = SITES * per_site / WORKERS
    return {
        "per_site_seconds": per_site,
        "one_fit_hours": one_fit / 3600.0,
        "hundred_fit_hours": 100.0 * one_fit / 3600.0,
        "objective_hundred_hours": 100.0 * SITES * EVALUATIONS * objective_seconds / WORKERS / 3600.0,
        "cache_hundred_hours": 100.0 * SITES * ITERATIONS * cache_seconds / WORKERS / 3600.0,
    }


def seven_batch_hours(objective_wall: float, cache_wall: float) -> dict:
    """Ceiling of 263/40 is 7. Each batch costs one measured 40-process wall."""
    seconds = 7.0 * (EVALUATIONS * objective_wall + ITERATIONS * cache_wall)
    return {"one_fit_hours": seconds / 3600.0, "hundred_fit_hours": 100.0 * seconds / 3600.0}


def main() -> None:
    BUILD.mkdir(parents=True, exist_ok=True)
    libraries = {
        "A": BUILD / "libA.so",
        "B": BUILD / "libB.so",
        "C": BUILD / "libC.so",
    }
    compile_library(SOURCE_A, libraries["A"], native=False)
    compile_library(SOURCE_T, libraries["B"], native=False)
    compile_library(SOURCE_T, libraries["C"], native=True)
    loaded = {}
    for name, path in libraries.items():
        loaded[name] = ctypes.CDLL(str(path))
        bind(loaded[name])

    msa = np.ascontiguousarray(np.load(PREP / "mature_codes.npy"), dtype=np.int8)
    weights = np.ascontiguousarray(np.loadtxt(PREP / "neighbor_counts.tsv", skiprows=1, usecols=2))
    total = float(weights.sum())
    count = Q + Q * (msa.shape[1] - 1) * Q
    generator = np.random.default_rng(20261120)
    theta = np.ascontiguousarray(generator.normal(0.0, 0.01, size=count))
    direction = np.ascontiguousarray(generator.normal(0.0, 0.01, size=count))
    zero = np.zeros(count, dtype=np.float64)
    tiny = np.ascontiguousarray(generator.integers(0, Q, size=(6, 5)), dtype=np.int8)
    tiny_weights = np.ones(tiny.shape[0], dtype=np.float64)
    tiny_count = Q + Q * (tiny.shape[1] - 1) * Q
    tiny_theta = np.ascontiguousarray(generator.normal(size=tiny_count))
    tiny_direction = np.ascontiguousarray(generator.normal(size=tiny_count))
    tiny_total = float(tiny_weights.sum())
    numeric_sites = (0, 12, 100, 200)
    numeric = {
        "zero": {
            "B": check_pair(loaded["A"], loaded["B"], msa, weights, zero, direction, total, numeric_sites),
            "C": check_pair(loaded["A"], loaded["C"], msa, weights, zero, direction, total, numeric_sites),
        },
        "random": {
            "B": check_pair(loaded["A"], loaded["B"], msa, weights, theta, direction, total, numeric_sites),
            "C": check_pair(loaded["A"], loaded["C"], msa, weights, theta, direction, total, numeric_sites),
        },
        "tiny": {
            "B": check_pair(loaded["A"], loaded["B"], tiny, tiny_weights, tiny_theta, tiny_direction, tiny_total, (0, 2, 4)),
            "C": check_pair(loaded["A"], loaded["C"], tiny, tiny_weights, tiny_theta, tiny_direction, tiny_total, (0, 2, 4)),
        },
    }
    single = {name: single_process(library, msa, weights, theta, direction, total) for name, library in loaded.items()}
    parallel = {
        name: {
            kind: concurrent(library, kind, msa, weights, theta, direction, total)
            for kind in ("objective", "direction_cache", "both")
        }
        for name, library in loaded.items()
    }
    zero_parallel = {
        name: {
            kind: concurrent(loaded[name], kind, msa, weights, zero, direction, total)
            for kind in ("objective", "direction_cache")
        }
        for name in ("A", "B")
    }
    estimates = {}
    for name, rows in parallel.items():
        estimates[name] = {
            "from_worker_mean": project_hours(rows["objective"]["worker_mean"], rows["direction_cache"]["worker_mean"]),
            "from_batch_wall": project_hours(rows["objective"]["wall"], rows["direction_cache"]["wall"]),
            "from_seven_batches": seven_batch_hours(rows["objective"]["wall"], rows["direction_cache"]["wall"]),
        }
    payload = {
        "parameter_case_for_timing": "normal(0, 0.01), seed 20261120",
        "evaluations_per_site": EVALUATIONS,
        "direction_caches_per_site": ITERATIONS,
        "numeric": numeric,
        "single_process": single,
        "forty_processes_nonzero": parallel,
        "forty_processes_zero": zero_parallel,
        "hundred_fit_extrapolation_nonzero": estimates,
    }
    output = Path(__file__).resolve().parent / "layout_results.json"
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Draw bootstrap alignments and refit them with the frozen Solver v2 site fit.

The formal replicate seed is intentionally absent. The verification entry uses
its own seed and is not a formal replicate. Each finished replicate is written
before the next one starts.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import multiprocessing
import os
import subprocess
import time
from pathlib import Path

import numpy as np

import bootstrap_gauge
import potts_fit
import solver_v2

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "bootstrap"
FULL_FIT = ROOT / "full_fit"
PREP = ROOT.parent / "prep"
DRAW_COUNT = 47900
SITE_COUNT = 263
WORKERS = 40
WEIGHT_TOTAL = 47900.0
VERIFICATION_SEED = 1
_SAMPLE: np.ndarray | None = None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def code_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def draw_indices(generator: np.random.Generator, weights: np.ndarray, total: float, count: int = DRAW_COUNT) -> np.ndarray:
    """Draw row indices with replacement using ``p = w / M_eff``."""
    probabilities = np.asarray(weights, dtype=np.float64) / float(total)
    if probabilities.shape != (weights.shape[0],) or not np.isclose(float(probabilities.sum()), 1.0):
        raise ValueError("sampling probabilities must be the normalized official weights")
    return generator.choice(weights.shape[0], size=int(count), replace=True, p=probabilities)


def replicate_sample(alignment: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Keep every drawn row, including repeated draws of the same source row."""
    return np.ascontiguousarray(alignment[np.asarray(indices, dtype=np.int64)])


def data_record() -> dict:
    alignment_path = PREP / "mature_codes.npy"
    counts_path = PREP / "neighbor_counts.tsv"
    return {
        "alignment_sha256": file_sha256(alignment_path),
        "neighbor_counts_sha256": file_sha256(counts_path),
        "official_m_eff": potts_fit.OFFICIAL_M_EFF,
        "draw_count": DRAW_COUNT,
        "weight_total": WEIGHT_TOTAL,
        "code_revision": code_revision(),
    }


def _init_worker(sample: np.ndarray) -> None:
    global _SAMPLE
    os.environ["POTTS_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    ctypes.CDLL("libgomp.so.1").omp_set_num_threads(1)
    potts_fit.set_objective_threads(1)
    _SAMPLE = sample


def _fit_worker(site: int) -> dict:
    if _SAMPLE is None:
        raise RuntimeError("the bootstrap sample was not loaded")
    if _SAMPLE.shape != (DRAW_COUNT, SITE_COUNT):
        raise RuntimeError("a bootstrap fit expects the 47900 by 263 sample")
    weights = np.ones(DRAW_COUNT, dtype=np.float64)
    started = time.perf_counter()
    result = potts_fit.fit_site(_SAMPLE, weights, int(site), WEIGHT_TOTAL)
    finite = bool(np.all(np.isfinite(result.h)) and np.all(np.isfinite(result.couplings)))
    passed = bool(
        result.status == "converged"
        and result.converged
        and np.isfinite(result.grad_inf)
        and result.grad_inf <= potts_fit.OPT_TOL
        and np.isfinite(result.objective)
        and result.iterations <= potts_fit.MAX_ITER
        and result.evaluations <= potts_fit.MAX_FUN_EVALS
        and finite
    )
    return {
        "site": int(site),
        "status": result.status,
        "converged": bool(result.converged),
        "grad_inf": float(result.grad_inf),
        "objective": float(result.objective),
        "iterations": int(result.iterations),
        "evaluations": int(result.evaluations),
        "finite_parameters": finite,
        "passed": passed,
        "seconds": time.perf_counter() - started,
        "h": result.h,
        "couplings": result.couplings,
    }


def _write_site(directory: Path, row: dict) -> None:
    np.savez(
        directory / f"site_{row['site']:03d}.npz",
        h=row["h"],
        couplings=row["couplings"],
        status=np.array(row["status"]),
        grad_inf=np.float64(row["grad_inf"]),
        objective=np.float64(row["objective"]),
        iterations=np.int32(row["iterations"]),
        evaluations=np.int32(row["evaluations"]),
        passed=np.bool_(row["passed"]),
    )


def _read_site(path: Path, site: int) -> dict:
    saved = np.load(path)
    return {
        "site": site,
        "status": str(saved["status"]),
        "grad_inf": float(saved["grad_inf"]),
        "objective": float(saved["objective"]),
        "iterations": int(saved["iterations"]),
        "evaluations": int(saved["evaluations"]),
        "passed": bool(saved["passed"]),
        "finite_parameters": bool(np.all(np.isfinite(saved["h"])) and np.all(np.isfinite(saved["couplings"]))),
        "h": saved["h"],
        "couplings": saved["couplings"],
    }


def fit_sample(sample: np.ndarray, directory: Path, workers: int = WORKERS) -> list[dict]:
    """Fit every site once, writing each site before the pool returns the next."""
    directory.mkdir(parents=True, exist_ok=True)
    pending = []
    rows = []
    for site in range(SITE_COUNT):
        path = directory / f"site_{site:03d}.npz"
        if path.is_file() and path.stat().st_size > 0:
            rows.append(_read_site(path, site))
        else:
            pending.append(site)
    if pending:
        os.environ["POTTS_THREADS"] = "1"
        os.environ["OMP_NUM_THREADS"] = "1"
        potts_fit._compile_library()
        context = multiprocessing.get_context("spawn")
        with context.Pool(workers, initializer=_init_worker, initargs=(sample,)) as pool:
            for row in pool.imap_unordered(_fit_worker, pending):
                _write_site(directory, row)
                print(
                    f"site {row['site']:3d}  {row['status']}  grad {row['grad_inf']:.3e}  "
                    f"iter {row['iterations']}  eval {row['evaluations']}  sec {row['seconds']:.1f}",
                    flush=True,
                )
                rows.append({key: value for key, value in row.items() if key not in {"h", "couplings"}})
        rows = [_read_site(directory / f"site_{site:03d}.npz", site) for site in range(SITE_COUNT)]
    rows.sort(key=lambda row: int(row["site"]))
    return rows


def write_replicate(directory: Path, sample: np.ndarray, indices: np.ndarray, rows: list[dict], metadata: dict) -> dict:
    """Assemble the gauge parameters and write the replicate manifest."""
    fields, directional = bootstrap_gauge.load_directional(directory, SITE_COUNT)
    recovered, averaged = bootstrap_gauge.gauge_parameters(fields, directional)
    weights = np.ones(DRAW_COUNT, dtype=np.float64)
    background = solver_v2._background(sample, weights, WEIGHT_TOTAL, potts_fit.Q)
    np.save(directory / "indices.npy", np.asarray(indices, dtype=np.int64))
    np.save(directory / "fields.npy", fields)
    np.save(directory / "directional.npy", directional)
    np.save(directory / "h_hat.npy", recovered)
    np.save(directory / "j_star.npy", averaged)
    np.save(directory / "q.npy", background)
    failed = [int(row["site"]) for row in rows if not row["passed"]]
    manifest = {
        **metadata,
        "sites": SITE_COUNT,
        "passed_sites": SITE_COUNT - len(failed),
        "failed_sites": failed,
        "replicate_success": len(failed) == 0,
        "entered_formal_statistics": False,
        "both_triangles": bool(np.array_equal(averaged, np.transpose(averaged, (1, 0, 3, 2)))),
        "workers": WORKERS,
        "threads": 1,
    }
    (directory / "manifest.json").write_text(json.dumps(solver_v2._json_ready(manifest), indent=2) + "\n")
    diagnostics = [{key: value for key, value in row.items() if key not in {"h", "couplings"}} for row in rows]
    (directory / "diagnostics.json").write_text(json.dumps(solver_v2._json_ready(diagnostics)) + "\n")
    return manifest


def build_reference(destination: Path | None = None) -> dict:
    """Gauge the accepted full-fit parameters without refitting them."""
    destination = OUTPUT / "reference" if destination is None else destination
    destination.mkdir(parents=True, exist_ok=True)
    fields, directional = bootstrap_gauge.load_directional(FULL_FIT, SITE_COUNT)
    recovered, averaged = bootstrap_gauge.gauge_parameters(fields, directional)
    np.save(destination / "fields.npy", fields)
    np.save(destination / "directional.npy", directional)
    np.save(destination / "h_hat.npy", recovered)
    np.save(destination / "j_star.npy", averaged)
    manifest = {
        "role": "reference",
        "source": str(FULL_FIT),
        "refit": False,
        "sites": SITE_COUNT,
        "both_triangles": bool(np.array_equal(averaged, np.transpose(averaged, (1, 0, 3, 2)))),
        **data_record(),
    }
    (destination / "manifest.json").write_text(json.dumps(solver_v2._json_ready(manifest), indent=2) + "\n")
    return manifest


def run_verification(seed: int = VERIFICATION_SEED, workers: int = WORKERS) -> dict:
    """Fit one 263-site sample. This replicate has no formal ID."""
    if int(seed) == 20261114:
        raise ValueError("the timing seed is not a verification seed")
    directory = OUTPUT / "verification" / "run"
    manifest_path = directory / "manifest.json"
    if manifest_path.is_file():
        saved = json.loads(manifest_path.read_text())
        if int(saved["verification_seed"]) != int(seed):
            raise RuntimeError("the verification run already used another seed")
        return saved
    alignment, weights, total = potts_fit.load_official_alignment()
    if alignment.shape != (weights.shape[0], SITE_COUNT):
        raise RuntimeError("the official alignment does not have 263 columns")
    generator = np.random.default_rng(int(seed))
    indices = draw_indices(generator, weights, total)
    sample = replicate_sample(alignment, indices)
    build_reference()
    rows = fit_sample(sample, directory, workers)
    manifest = write_replicate(
        directory,
        sample,
        indices,
        rows,
        {
            "role": "verification",
            "verification_seed": int(seed),
            "formal_seed": None,
            "formal_replicate_id": None,
            **data_record(),
        },
    )
    print(
        f"verification passed_sites {manifest['passed_sites']}/{SITE_COUNT} "
        f"replicate_success {manifest['replicate_success']}",
        flush=True,
    )
    return manifest


def formal_seed_sequence(seed: int, replicate_id: int) -> np.random.Generator:
    """Return the generator for one preassigned formal replicate.

    Replicate ``r`` does not depend on drawing replicates ``1..r-1``. The
    formal seed value is supplied by the caller; this module has no default.
    """
    if int(replicate_id) < 1:
        raise ValueError("formal replicate ids start at 1")
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(replicate_id)]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("verification", "reference"))
    parser.add_argument("--seed", type=int, default=VERIFICATION_SEED)
    args = parser.parse_args()
    if args.command == "reference":
        manifest = build_reference()
    else:
        manifest = run_verification(args.seed)
    print(json.dumps(solver_v2._json_ready({"role": manifest["role"], "code_revision": manifest["code_revision"]})) , flush=True)


if __name__ == "__main__":
    main()

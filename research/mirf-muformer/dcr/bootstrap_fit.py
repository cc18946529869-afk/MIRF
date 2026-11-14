"""Draw bootstrap alignments and refit them with the frozen Solver v2 site fit.

The formal seed has no default. A verification run is not a formal replicate.
Sampling indices and the replicate identity are written before any site is fit.
A site file is reused only when it belongs to that same sample. One failed site
keeps the replicate out of the gauge parameters and out of the formal statistic.
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
import bootstrap_stats
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
TIMING_SEED = 20261114
NEIGHBOR_THRESHOLD = 211
GAUGE_FILES = ("fields.npy", "directional.npy", "h_hat.npy", "j_star.npy", "q.npy")
_SAMPLE: np.ndarray | None = None
_DATA_RECORD: dict | None = None


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


def data_record() -> dict:
    global _DATA_RECORD
    if _DATA_RECORD is None:
        _DATA_RECORD = {
            "alignment_sha256": file_sha256(PREP / "mature_codes.npy"),
            "neighbor_counts_sha256": file_sha256(PREP / "neighbor_counts.tsv"),
            "official_m_eff": potts_fit.OFFICIAL_M_EFF,
            "draw_count": DRAW_COUNT,
            "weight_total": WEIGHT_TOTAL,
            "code_revision": code_revision(),
        }
    return dict(_DATA_RECORD)


def indices_sha256(indices: np.ndarray) -> str:
    array = np.ascontiguousarray(indices, dtype=np.int64)
    return hashlib.sha256(array.tobytes()).hexdigest()


def _token(payload: dict) -> str:
    body = {
        "role": payload["role"],
        "replicate_id": payload["replicate_id"],
        "seed": payload["seed"],
        "alignment_sha256": payload["alignment_sha256"],
        "neighbor_counts_sha256": payload["neighbor_counts_sha256"],
        "code_revision": payload["code_revision"],
        "indices_sha256": payload["indices_sha256"],
        "draw_count": payload["draw_count"],
        "weight_total": payload["weight_total"],
    }
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def make_identity(role: str, seed: int, replicate_id: int | None, indices: np.ndarray) -> dict:
    """Bind one replicate to its seed, sample, data files, and code revision."""
    data = data_record()
    payload = {
        "role": role,
        "replicate_id": None if replicate_id is None else int(replicate_id),
        "seed": int(seed),
        "alignment_sha256": data["alignment_sha256"],
        "neighbor_counts_sha256": data["neighbor_counts_sha256"],
        "code_revision": data["code_revision"],
        "indices_sha256": indices_sha256(indices),
        "draw_count": int(indices.shape[0]),
        "weight_total": float(WEIGHT_TOTAL),
    }
    if int(indices.shape[0]) != int(DRAW_COUNT):
        raise ValueError("a bootstrap sample has 47900 draws")
    payload["identity_token"] = _token(payload)
    return payload


def draw_indices(generator: np.random.Generator, weights: np.ndarray, total: float, count: int = DRAW_COUNT) -> np.ndarray:
    """Draw row indices with replacement using ``p = w / M_eff``."""
    probabilities = np.asarray(weights, dtype=np.float64) / float(total)
    if probabilities.shape != (weights.shape[0],) or not np.isclose(float(probabilities.sum()), 1.0):
        raise ValueError("sampling probabilities must be the normalized official weights")
    return generator.choice(weights.shape[0], size=int(count), replace=True, p=probabilities)


def replicate_sample(alignment: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Keep every drawn row, including repeated draws of the same source row."""
    return np.ascontiguousarray(alignment[np.asarray(indices, dtype=np.int64)])


def _sequence_keys(alignment: np.ndarray) -> np.ndarray:
    packed = np.ascontiguousarray(alignment)
    if packed.ndim != 2 or packed.shape[1] == 0:
        raise ValueError("an alignment needs rows and columns")
    width = packed.dtype.itemsize * packed.shape[1]
    return packed.view(np.dtype((np.void, width))).ravel()


def identical_sequences_share_counts(alignment: np.ndarray, counts: np.ndarray) -> int:
    """Return the number of duplicated sequences. Disagreeing counts raise."""
    keys = _sequence_keys(alignment)
    if keys.shape != counts.shape:
        raise ValueError("neighbor counts do not have one entry per alignment row")
    order = np.argsort(keys, kind="mergesort")
    sorted_keys = keys[order]
    groups = 0
    mismatched = 0
    start = 0
    for index in range(1, keys.size + 1):
        if index < keys.size and sorted_keys[index] == sorted_keys[start]:
            continue
        if index - start > 1:
            groups += 1
            chosen = counts[order[start:index]]
            if np.any(chosen != chosen[0]):
                mismatched += 1
        start = index
    if mismatched:
        raise RuntimeError(f"{mismatched} identical sequences do not share one neighbor count")
    return groups


def recount_neighbors(alignment: np.ndarray, row: int, threshold: int = NEIGHBOR_THRESHOLD) -> int:
    """Count rows, including itself, with at least ``threshold`` matching columns."""
    matches = np.count_nonzero(alignment == alignment[int(row)], axis=1)
    return int(np.count_nonzero(matches >= int(threshold)))


def require_msa_row_order(row_index: np.ndarray, row_count: int) -> None:
    """The weight table's first column must be the MSA row number, in that order."""
    expected = np.arange(int(row_count), dtype=np.int64)
    if row_index.shape != expected.shape or not np.array_equal(row_index, expected):
        raise RuntimeError("neighbor_counts.tsv row ids are not the MSA row order")


def verify_official_weight_rows() -> dict:
    """Require the weight file's row ids to be the MSA order, then test that claim.

    The table has a row number and no sequence header. Identical MSA rows must
    share ``m_i``. A fixed set of rows is recounted against ``mature_codes.npy``.
    This does not repeat the full pair scan.
    """
    alignment, weights, total = potts_fit.load_official_alignment()
    table = np.loadtxt(PREP / "neighbor_counts.tsv", skiprows=1, usecols=(0, 1))
    row_index = table[:, 0].astype(np.int64)
    counts = table[:, 1].astype(np.int64)
    if alignment.shape != (counts.shape[0], SITE_COUNT):
        raise RuntimeError("the alignment and neighbor counts do not have the same rows")
    require_msa_row_order(row_index, alignment.shape[0])
    if np.any(counts < 1):
        raise RuntimeError("a neighbor count is below one")
    if not np.array_equal(weights, np.float64(1.0) / counts.astype(np.float64)):
        raise RuntimeError("official weights are not 1/m_i in MSA row order")
    if float(total) != float(potts_fit.OFFICIAL_M_EFF):
        raise RuntimeError("the official weight sum changed")
    summary = json.loads((PREP / "neighbor_summary.json").read_text())
    if int(summary["n_rows"]) != int(alignment.shape[0]) or int(summary["query_row"]) != 0:
        raise RuntimeError("neighbor_summary.json does not describe this alignment")
    if int(summary["query_m"]) != int(counts[0]):
        raise RuntimeError("the query neighbor count does not match the summary")
    if int(summary["sum_m"]) != int(counts.sum()):
        raise RuntimeError("the neighbor-count sum does not match the summary")
    if int(summary["sum_m_minus_1"]) != int(np.sum(counts - 1)):
        raise RuntimeError("the neighbor-pair total does not match the summary")
    if int(summary["sum_m_minus_1"]) != 2 * int(summary["neighbor_pairs"]):
        raise RuntimeError("the summary pair count is not half the sum of m_i-1")
    if float(summary["M_eff"]) != float(potts_fit.OFFICIAL_M_EFF):
        raise RuntimeError("the summary M_eff is not the official value")
    groups = identical_sequences_share_counts(alignment, counts)
    recount_rows = np.unique(np.linspace(0, alignment.shape[0] - 1, 32, dtype=np.int64))
    recounted = []
    for row in recount_rows.tolist():
        observed = recount_neighbors(alignment, int(row))
        if observed != int(counts[int(row)]):
            raise RuntimeError(f"row {int(row)} recount {observed} does not match stored m_i {int(counts[int(row)])}")
        recounted.append({"row": int(row), "m_i": observed})
    return {
        "row_index_is_msa_order": True,
        "identical_sequences_share_counts": True,
        "duplicate_groups": groups,
        "recount_rows": recounted,
        "full_pair_recount": False,
        "query_row": 0,
        "query_m": int(counts[0]),
    }


def atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, payload: dict) -> None:
    atomic_text(path, json.dumps(solver_v2._json_ready(payload), indent=2) + "\n")


def atomic_npy(path: Path, array: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npy")
    np.save(temporary, array)
    os.replace(temporary, path)


def atomic_savez(path: Path, **arrays) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez(temporary, **arrays)
    os.replace(temporary, path)


def ensure_indices(directory: Path, indices: np.ndarray) -> None:
    """Write the draw before fitting. A different saved draw stops the run."""
    array = np.ascontiguousarray(indices, dtype=np.int64)
    path = directory / "indices.npy"
    if path.exists():
        if not np.array_equal(np.load(path), array):
            raise RuntimeError("saved sampling indices do not match this sample")
        return
    atomic_npy(path, array)


def _read_token(path: Path) -> str:
    try:
        with np.load(path) as saved:
            if "identity_token" not in saved.files:
                raise RuntimeError(f"{path.name} has no sample identity and will not be reused")
            return str(saved["identity_token"].item())
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"{path.name} cannot be read and will not be reused") from exc


def _require_existing_sites(directory: Path, token: str) -> None:
    for site in range(SITE_COUNT):
        path = directory / f"site_{site:03d}.npz"
        if not path.exists():
            continue
        if path.stat().st_size == 0:
            raise RuntimeError(f"{path.name} is empty and will not be reused")
        found = _read_token(path)
        if found != token:
            raise RuntimeError(f"{path.name} belongs to another sample")


def _gauge_names_present(directory: Path) -> list[str]:
    return [name for name in GAUGE_FILES if (directory / name).exists()]


def _require_finished_files(directory: Path, success: bool) -> None:
    present = _gauge_names_present(directory)
    if success and tuple(present) != GAUGE_FILES:
        raise RuntimeError("a successful replicate is missing its gauge parameters")
    if not success and present:
        raise RuntimeError("a failed replicate still has gauge parameters")


def pending_sites(directory: Path, token: str) -> list[int]:
    """Return unfinished sites. A file from another sample is never a resume."""
    _require_existing_sites(directory, token)
    return [site for site in range(SITE_COUNT) if not (directory / f"site_{site:03d}.npz").exists()]


def open_replicate(directory: Path, identity: dict, indices: np.ndarray, manifest_extra: dict) -> str:
    """Record the sample, then accept only site files carrying the same token."""
    directory.mkdir(parents=True, exist_ok=True)
    token = identity["identity_token"]
    index_path = directory / "indices.npy"
    array = np.ascontiguousarray(indices, dtype=np.int64)
    if index_path.exists() and not np.array_equal(np.load(index_path), array):
        raise RuntimeError("saved sampling indices do not match this sample")
    _require_existing_sites(directory, token)
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text())
        if saved.get("identity_token") != token:
            raise RuntimeError(
                "this directory is bound to another sample, or was written before sample identity was recorded"
            )
        if saved.get("state") == "complete":
            _require_finished_files(directory, bool(saved["replicate_success"]))
            return "complete"
    if not index_path.exists():
        atomic_npy(index_path, array)
    atomic_json(
        manifest_path,
        {
            **identity,
            **manifest_extra,
            "state": "running",
            "sites": SITE_COUNT,
        },
    )
    return "running"


def _site_accepted(status: str, grad_inf: float, objective: float, iterations: int, evaluations: int, finite_parameters: bool) -> bool:
    """Accept one site. Fallback and curvature counts are not part of this rule."""
    return bool(
        status == "converged"
        and np.isfinite(grad_inf)
        and float(grad_inf) <= potts_fit.OPT_TOL
        and np.isfinite(objective)
        and int(iterations) <= potts_fit.MAX_ITER
        and int(evaluations) <= potts_fit.MAX_FUN_EVALS
        and bool(finite_parameters)
    )


def _search_diagnostics(result) -> dict:
    exits = [row.get("line_search_exit") for row in result.trace if row.get("line_search_exit")]
    curvature = [float(value) for value in result.curvature_sy]
    return {
        "n_strong_wolfe": int(result.n_strong_wolfe),
        "n_armijo_fallback": int(result.n_armijo_fallback),
        "n_curvature_skipped": int(result.n_curvature_skipped),
        "curvature_sy": curvature,
        "curvature_sy_min": min(curvature) if curvature else None,
        "curvature_sy_max": max(curvature) if curvature else None,
        "line_search_exits": exits,
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
    passed = _site_accepted(
        result.status,
        result.grad_inf,
        result.objective,
        result.iterations,
        result.evaluations,
        finite,
    )
    row = {
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
    row.update(_search_diagnostics(result))
    return row


def _write_site(directory: Path, row: dict, token: str) -> None:
    atomic_savez(
        directory / f"site_{int(row['site']):03d}.npz",
        h=np.asarray(row["h"], dtype=np.float64),
        couplings=np.asarray(row["couplings"], dtype=np.float64),
        status=np.array(row["status"]),
        grad_inf=np.float64(row["grad_inf"]),
        objective=np.float64(row["objective"]),
        iterations=np.int32(row["iterations"]),
        evaluations=np.int32(row["evaluations"]),
        passed=np.bool_(row["passed"]),
        seconds=np.float64(row["seconds"]),
        n_strong_wolfe=np.int32(row["n_strong_wolfe"]),
        n_armijo_fallback=np.int32(row["n_armijo_fallback"]),
        n_curvature_skipped=np.int32(row["n_curvature_skipped"]),
        curvature_sy=np.asarray(row["curvature_sy"], dtype=np.float64),
        line_search_exits=np.asarray(row["line_search_exits"], dtype="U128"),
        identity_token=np.asarray(token),
    )


def _read_site(path: Path, site: int) -> dict:
    with np.load(path) as saved:
        fields = np.array(saved["h"], copy=True)
        couplings = np.array(saved["couplings"], copy=True)
        curvature = np.array(saved["curvature_sy"], dtype=np.float64, copy=True)
        exits = [str(item) for item in np.asarray(saved["line_search_exits"]).tolist()]
        finite = bool(np.all(np.isfinite(fields)) and np.all(np.isfinite(couplings)))
        return {
            "site": site,
            "status": str(saved["status"]),
            "grad_inf": float(saved["grad_inf"]),
            "objective": float(saved["objective"]),
            "iterations": int(saved["iterations"]),
            "evaluations": int(saved["evaluations"]),
            "passed": bool(saved["passed"]),
            "finite_parameters": finite,
            "seconds": float(saved["seconds"]),
            "n_strong_wolfe": int(saved["n_strong_wolfe"]),
            "n_armijo_fallback": int(saved["n_armijo_fallback"]),
            "n_curvature_skipped": int(saved["n_curvature_skipped"]),
            "curvature_sy": curvature.tolist(),
            "curvature_sy_min": None if curvature.size == 0 else float(curvature.min()),
            "curvature_sy_max": None if curvature.size == 0 else float(curvature.max()),
            "line_search_exits": exits,
            "identity_token": str(saved["identity_token"].item()),
            "h": fields,
            "couplings": couplings,
        }


def fit_sample(sample: np.ndarray, directory: Path, token: str, workers: int = WORKERS) -> list[dict]:
    """Fit unfinished sites once. Each finished site is replaced into place immediately."""
    if sample.shape != (DRAW_COUNT, SITE_COUNT):
        raise RuntimeError("a bootstrap fit expects the 47900 by 263 sample")
    pending = pending_sites(directory, token)
    if pending:
        os.environ["POTTS_THREADS"] = "1"
        os.environ["OMP_NUM_THREADS"] = "1"
        potts_fit._compile_library()
        context = multiprocessing.get_context("spawn")
        with context.Pool(workers, initializer=_init_worker, initargs=(sample,)) as pool:
            for row in pool.imap_unordered(_fit_worker, pending):
                _write_site(directory, row, token)
                print(
                    f"site {row['site']:3d}  {row['status']}  grad {row['grad_inf']:.3e}  "
                    f"iter {row['iterations']}  eval {row['evaluations']}  sec {row['seconds']:.1f}",
                    flush=True,
                )
    rows = [_read_site(directory / f"site_{site:03d}.npz", site) for site in range(SITE_COUNT)]
    for row in rows:
        if row["identity_token"] != token:
            raise RuntimeError(f"site {row['site']} belongs to another sample")
    rows.sort(key=lambda row: int(row["site"]))
    return rows


def _ordered_rows(rows: list[dict]) -> list[dict]:
    if len(rows) != SITE_COUNT:
        raise RuntimeError("a replicate needs one diagnostic for every site")
    by_site = {int(row["site"]): row for row in rows}
    if set(by_site) != set(range(SITE_COUNT)):
        raise RuntimeError("a replicate is missing a site diagnostic")
    ordered = [by_site[site] for site in range(SITE_COUNT)]
    for row in ordered:
        accepted = _site_accepted(
            str(row["status"]),
            float(row["grad_inf"]),
            float(row["objective"]),
            int(row["iterations"]),
            int(row["evaluations"]),
            bool(row["finite_parameters"]),
        )
        if bool(row["passed"]) != accepted:
            raise RuntimeError(f"site {row['site']} pass flag does not match its diagnostics")
    return ordered


def _write_diagnostics(directory: Path, rows: list[dict]) -> None:
    diagnostics = [
        {key: value for key, value in row.items() if key not in {"h", "couplings", "identity_token"}}
        for row in rows
    ]
    atomic_json(directory / "diagnostics.json", diagnostics)


def _discard_gauge(directory: Path) -> None:
    for name in GAUGE_FILES:
        path = directory / name
        if path.exists():
            path.unlink()


def finalize_replicate(
    directory: Path,
    sample: np.ndarray | None,
    indices: np.ndarray,
    rows: list[dict],
    identity: dict,
    manifest_extra: dict,
) -> dict:
    """Write the replicate. Gauge parameters are written only when every site passed."""
    directory.mkdir(parents=True, exist_ok=True)
    if indices_sha256(indices) != identity["indices_sha256"]:
        raise RuntimeError("sampling indices do not match the replicate identity")
    ensure_indices(directory, indices)
    ordered = _ordered_rows(rows)
    failed = [int(row["site"]) for row in ordered if not row["passed"]]
    manifest = {
        **identity,
        **manifest_extra,
        "state": "complete",
        "sites": SITE_COUNT,
        "passed_sites": SITE_COUNT - len(failed),
        "failed_sites": failed,
        "replicate_success": len(failed) == 0,
        "entered_formal_statistics": identity["role"] == "formal" and len(failed) == 0,
        "both_triangles": None,
        "threads": 1,
    }
    if failed:
        _discard_gauge(directory)
        _write_diagnostics(directory, ordered)
        atomic_json(directory / "manifest.json", manifest)
        return manifest
    fields, directional = bootstrap_gauge.load_directional(directory, SITE_COUNT)
    recovered, averaged = bootstrap_gauge.gauge_parameters(fields, directional)
    both_triangles = bool(np.array_equal(averaged, np.transpose(averaged, (1, 0, 3, 2))))
    if not both_triangles:
        raise RuntimeError("J* does not keep both triangles")
    if sample is None or sample.shape != (DRAW_COUNT, SITE_COUNT):
        raise RuntimeError("a successful replicate needs its 47900 by 263 sample")
    background = solver_v2._background(sample, np.ones(DRAW_COUNT, dtype=np.float64), WEIGHT_TOTAL, potts_fit.Q)
    atomic_npy(directory / "fields.npy", fields)
    atomic_npy(directory / "directional.npy", directional)
    atomic_npy(directory / "h_hat.npy", recovered)
    atomic_npy(directory / "j_star.npy", averaged)
    atomic_npy(directory / "q.npy", background)
    manifest["both_triangles"] = True
    _write_diagnostics(directory, ordered)
    atomic_json(directory / "manifest.json", manifest)
    return manifest


def _metadata(role: str, seed: int, replicate_id: int | None, pairing: dict, workers: int) -> dict:
    formal = role == "formal"
    return {
        "verification_seed": None if formal else int(seed),
        "formal_seed": int(seed) if formal else None,
        "formal_replicate_id": int(replicate_id) if formal else None,
        "weight_pairing": pairing,
        "workers": int(workers),
        "threads": 1,
    }


def execute_replicate(
    *,
    role: str,
    seed: int,
    replicate_id: int | None,
    directory: Path,
    alignment: np.ndarray,
    weights: np.ndarray,
    total: float,
    workers: int,
    pairing: dict,
) -> dict:
    if role == "verification":
        generator = np.random.default_rng(int(seed))
    elif role == "formal":
        generator = formal_seed_sequence(int(seed), int(replicate_id))
    else:
        raise ValueError("the replicate role must be verification or formal")
    indices = draw_indices(generator, weights, total)
    sample = replicate_sample(alignment, indices)
    identity = make_identity(role, int(seed), replicate_id, indices)
    extra = _metadata(role, int(seed), replicate_id, pairing, workers)
    state = open_replicate(directory, identity, indices, extra)
    if state == "complete":
        return json.loads((directory / "manifest.json").read_text())
    rows = fit_sample(sample, directory, identity["identity_token"], workers)
    return finalize_replicate(directory, sample, indices, rows, identity, extra)


def build_reference(destination: Path | None = None) -> dict:
    """Gauge the accepted full-fit parameters without refitting them."""
    destination = OUTPUT / "reference" if destination is None else destination
    destination.mkdir(parents=True, exist_ok=True)
    fields, directional = bootstrap_gauge.load_directional(FULL_FIT, SITE_COUNT)
    recovered, averaged = bootstrap_gauge.gauge_parameters(fields, directional)
    both_triangles = bool(np.array_equal(averaged, np.transpose(averaged, (1, 0, 3, 2))))
    if not both_triangles:
        raise RuntimeError("the reference J* does not keep both triangles")
    atomic_npy(destination / "fields.npy", fields)
    atomic_npy(destination / "directional.npy", directional)
    atomic_npy(destination / "h_hat.npy", recovered)
    atomic_npy(destination / "j_star.npy", averaged)
    manifest = {
        "role": "reference",
        "source": str(FULL_FIT),
        "refit": False,
        "sites": SITE_COUNT,
        "both_triangles": True,
        **data_record(),
    }
    atomic_json(destination / "manifest.json", manifest)
    return manifest


def run_verification(seed: int = VERIFICATION_SEED, workers: int = WORKERS) -> dict:
    """Fit one 263-site sample. This replicate has no formal ID."""
    if int(seed) == TIMING_SEED:
        raise ValueError("the timing seed is not a verification seed")
    alignment, weights, total = potts_fit.load_official_alignment()
    if alignment.shape != (weights.shape[0], SITE_COUNT):
        raise RuntimeError("the official alignment does not have 263 columns")
    pairing = verify_official_weight_rows()
    if not (OUTPUT / "reference" / "manifest.json").exists():
        build_reference()
    manifest = execute_replicate(
        role="verification",
        seed=int(seed),
        replicate_id=None,
        directory=OUTPUT / "verification" / "run",
        alignment=alignment,
        weights=weights,
        total=total,
        workers=workers,
        pairing=pairing,
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


def _formal_directory(replicate_id: int) -> Path:
    return OUTPUT / "formal" / f"replicate_{int(replicate_id):03d}"


def lock_formal_seed(seed: int, data: dict) -> None:
    """Keep every formal replicate on the seed and alignment recorded first."""
    if int(seed) == TIMING_SEED:
        raise ValueError("the timing seed is not the formal seed")
    root = OUTPUT / "formal"
    root.mkdir(parents=True, exist_ok=True)
    path = root / "seed.json"
    payload = {
        "formal_seed": int(seed),
        "alignment_sha256": data["alignment_sha256"],
        "neighbor_counts_sha256": data["neighbor_counts_sha256"],
    }
    if path.exists():
        saved = json.loads(path.read_text())
        if (
            int(saved["formal_seed"]) != int(seed)
            or saved["alignment_sha256"] != payload["alignment_sha256"]
            or saved["neighbor_counts_sha256"] != payload["neighbor_counts_sha256"]
        ):
            raise RuntimeError("the formal run is already bound to another seed or alignment")
        return
    atomic_json(path, payload)


def _require_prefix(seed: int, last: int) -> None:
    for replicate_id in range(1, int(last) + 1):
        path = _formal_directory(replicate_id) / "manifest.json"
        if not path.exists():
            raise RuntimeError(f"replicate {replicate_id} is absent; later replicates do not redraw it")
        saved = json.loads(path.read_text())
        if saved.get("state") != "complete":
            raise RuntimeError(f"replicate {replicate_id} is not finished")
        if int(saved.get("formal_seed", -1)) != int(seed):
            raise RuntimeError(f"replicate {replicate_id} used another formal seed")


def _stage_complete(stage: int, seed: int) -> bool:
    try:
        _require_prefix(seed, stage)
    except RuntimeError:
        return False
    return True


def _assert_both_triangles(array: np.ndarray, replicate_id: int) -> None:
    if array.ndim != 4:
        return
    if not np.array_equal(array, np.transpose(array, (1, 0, 3, 2))):
        raise RuntimeError(f"replicate {replicate_id} J* does not keep both triangles")


def load_formal_samples(stage: int, seed: int, data: dict) -> tuple[list[bool], dict, dict]:
    """Load successful gauge parameters. A failed replicate cannot contribute arrays."""
    if stage not in bootstrap_stats.STAGE_MINIMUM:
        raise ValueError("the stage must be 100 or 200")
    required = (
        "state",
        "role",
        "formal_seed",
        "formal_replicate_id",
        "identity_token",
        "alignment_sha256",
        "neighbor_counts_sha256",
        "code_revision",
        "indices_sha256",
        "replicate_success",
        "entered_formal_statistics",
    )
    success: list[bool] = []
    fields: dict[int, np.ndarray] = {}
    couplings: dict[int, np.ndarray] = {}
    revisions: list[str] = []
    for replicate_id in range(1, int(stage) + 1):
        directory = _formal_directory(replicate_id)
        manifest_path = directory / "manifest.json"
        if not manifest_path.exists():
            raise RuntimeError(f"replicate {replicate_id} is not on disk")
        saved = json.loads(manifest_path.read_text())
        missing = [key for key in required if key not in saved]
        if missing:
            raise RuntimeError(f"replicate {replicate_id} manifest is missing {missing[0]}")
        if saved["state"] != "complete" or saved["role"] != "formal":
            raise RuntimeError(f"replicate {replicate_id} is not a finished formal replicate")
        if int(saved["formal_seed"]) != int(seed) or int(saved["formal_replicate_id"]) != replicate_id:
            raise RuntimeError(f"replicate {replicate_id} does not match the requested formal run")
        if saved["alignment_sha256"] != data["alignment_sha256"] or saved["neighbor_counts_sha256"] != data["neighbor_counts_sha256"]:
            raise RuntimeError(f"replicate {replicate_id} does not match the alignment used for this decision")
        saved_indices = np.load(directory / "indices.npy")
        if indices_sha256(saved_indices) != saved["indices_sha256"]:
            raise RuntimeError(f"replicate {replicate_id} indices do not match its manifest")
        revisions.append(str(saved["code_revision"]))
        replicate_success = bool(saved["replicate_success"])
        if bool(saved["entered_formal_statistics"]) != replicate_success:
            raise RuntimeError(f"replicate {replicate_id} does not agree on whether it enters the statistic")
        if replicate_success:
            if tuple(_gauge_names_present(directory)) != GAUGE_FILES:
                raise RuntimeError(f"replicate {replicate_id} is missing gauge parameters")
            loaded_fields = np.load(directory / "h_hat.npy")
            loaded_couplings = np.load(directory / "j_star.npy")
            _assert_both_triangles(loaded_couplings, replicate_id)
            fields[replicate_id] = loaded_fields
            couplings[replicate_id] = loaded_couplings
        elif _gauge_names_present(directory):
            raise RuntimeError(f"replicate {replicate_id} failed but still has gauge parameters")
        success.append(replicate_success)
    if len(set(revisions)) != 1:
        raise RuntimeError("formal replicates do not share one code revision")
    return success, fields, couplings


def decide_formal_stage(stage: int, seed: int, field_reference: np.ndarray, coupling_reference: np.ndarray, data: dict) -> dict:
    """Apply the existing decision table to one finished formal stage."""
    success, fields, couplings = load_formal_samples(stage, seed, data)
    return bootstrap_stats.evaluate_stage(success, fields, couplings, field_reference, coupling_reference, stage)


def write_stage_decision(stage: int, seed: int) -> dict:
    reference = OUTPUT / "reference"
    if not (reference / "manifest.json").exists():
        build_reference()
    outcome = decide_formal_stage(
        int(stage),
        int(seed),
        np.load(reference / "h_hat.npy"),
        np.load(reference / "j_star.npy"),
        data_record(),
    )
    saved = {**outcome, "formal_seed": int(seed)}
    atomic_json(OUTPUT / "formal" / f"stage_{int(stage)}.json", saved)
    return outcome


def run_formal(seed: int, start: int = 1, stop: int = 100, workers: int = WORKERS) -> dict:
    """Fit preassigned formal replicates. Extension does not redraw earlier IDs."""
    if int(seed) == TIMING_SEED:
        raise ValueError("the timing seed is not the formal seed")
    start = int(start)
    stop = int(stop)
    if not 1 <= start <= stop <= 200:
        raise ValueError("formal replicate ids run from 1 through 200")
    alignment, weights, total = potts_fit.load_official_alignment()
    pairing = verify_official_weight_rows()
    recorded = data_record()
    lock_formal_seed(int(seed), recorded)
    if start > 1:
        _require_prefix(int(seed), start - 1)
    replicate_ids = []
    for replicate_id in range(start, stop + 1):
        manifest = execute_replicate(
            role="formal",
            seed=int(seed),
            replicate_id=replicate_id,
            directory=_formal_directory(replicate_id),
            alignment=alignment,
            weights=weights,
            total=total,
            workers=workers,
            pairing=pairing,
        )
        replicate_ids.append(replicate_id)
        print(
            f"replicate {replicate_id} passed_sites {manifest['passed_sites']}/{SITE_COUNT} "
            f"replicate_success {manifest['replicate_success']}",
            flush=True,
        )
    decisions = {}
    if stop >= 100 and _stage_complete(100, int(seed)):
        decisions["100"] = write_stage_decision(100, int(seed))
    if stop >= 200 and _stage_complete(200, int(seed)):
        decisions["200"] = write_stage_decision(200, int(seed))
    return {
        "role": "formal",
        "formal_seed": int(seed),
        "replicate_ids": replicate_ids,
        "decisions": decisions,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("verification", "reference", "formal", "decide"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--stop", type=int, default=100)
    parser.add_argument("--stage", type=int, choices=(100, 200))
    args = parser.parse_args(argv)
    if args.command == "reference":
        manifest = build_reference()
        summary = {"role": manifest["role"], "code_revision": manifest["code_revision"]}
    elif args.command == "formal":
        if args.seed is None:
            parser.error("formal requires --seed")
        result = run_formal(args.seed, args.start, args.stop)
        summary = {
            "role": "formal",
            "formal_seed": result["formal_seed"],
            "replicate_ids": result["replicate_ids"],
            "decisions": {key: value["action"] for key, value in result["decisions"].items()},
        }
    elif args.command == "decide":
        if args.seed is None or args.stage is None:
            parser.error("decide requires --seed and --stage")
        outcome = write_stage_decision(args.stage, args.seed)
        summary = {"role": "decide", "stage": args.stage, "action": outcome["action"]}
    else:
        seed = VERIFICATION_SEED if args.seed is None else args.seed
        manifest = run_verification(seed)
        summary = {"role": manifest["role"], "code_revision": manifest["code_revision"]}
    print(json.dumps(solver_v2._json_ready(summary)), flush=True)


if __name__ == "__main__":
    main()

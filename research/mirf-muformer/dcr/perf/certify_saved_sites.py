"""Read-only interval certification of saved sites. Does not invoke L-BFGS."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import ctypes
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import resource
import subprocess
import tempfile
from time import perf_counter

import numpy as np

import potts_fit
from .check_top263_synthetic import Intervals, point, ZERO
from .fit_kernel_b import BoundBKernel, build

RAW_SOURCE = Path(__file__).with_name("certification_grid.c")
ROW_GRID_BITS = 45
I64_MAX = (1 << 63) - 1
INPUT = None
KERNEL = None
GRID = None


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def same_hash(path, expected):
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"Frozen source hash differs: {path.name}")
    return actual


class Sources:
    def __init__(self, root):
        self.root = root
        self.prep = root / "prep"
        self.full = root / "dcr/full_fit"
        self.reference = root / "dcr/bootstrap/reference"
        path = self.reference / "manifest.json"
        self.manifest = json.loads(path.read_text())
        if not self.manifest["accepted_point_estimate"] or self.manifest["refit"]:
            raise ValueError("Expected the accepted point estimate and original reference")
        self.identity = {"manifest_sha256": sha256(path)}
        for filename, key in (("mature_codes.npy", "alignment_sha256"), ("neighbor_counts.tsv", "neighbor_counts_sha256")):
            self.identity[key] = same_hash(self.prep / filename, self.manifest[key])
        report = self.full / "summary.json"
        self.identity["summary_sha256"] = same_hash(report, self.manifest["source_sha256"]["summary.json"])
        summary = json.loads(report.read_text())
        if not summary["accepted_point_estimate"] or summary["n_passed"] != 263:
            raise ValueError("Saved full-fit summary was not accepted")
        self.matrix, self.weights, self.total = potts_fit.load_official_alignment(self.prep)
        if self.matrix.shape != (209644, 263) or self.total != self.manifest["official_m_eff"]:
            raise ValueError("Unexpected official shape or total penalty weight")

    def theta(self, site):
        path = self.full / f"site_{site:03d}.npz"
        digest = same_hash(path, self.manifest["source_sha256"][path.name])
        with np.load(path, allow_pickle=False) as data:
            theta = data["theta"]
            h, j = potts_fit.unpack_parameters(theta, 263)
            if (
                theta.dtype != np.float64
                or not np.isfinite(theta).all()
                or not np.array_equal(data["h"], h)
                or not np.array_equal(data["couplings"], j)
                or data["status"].item() != "converged"
            ):
                raise ValueError(f"Invalid accepted parameters for site {site}")
        return theta, digest


class Grid:
    def __init__(self, directory):
        lib = directory / "certification_grid.so"
        subprocess.run(
            ["gcc", "-std=c11", "-O3", "-shared", "-fPIC", "-Wall", "-Wextra", "-Werror",
             str(RAW_SOURCE), "-o", str(lib)],
            check=True,
        )
        self.library = ctypes.CDLL(str(lib))
        self.identity = {"source_sha256": sha256(RAW_SOURCE), "artifact_sha256": sha256(lib)}
        for name in ("grid_logits", "grid_gradients"):
            function = getattr(self.library, name)
            function.argtypes = [
                np.ctypeslib.ndpointer(dtype=np.int8, ndim=2, flags="C_CONTIGUOUS"),
                *[np.ctypeslib.ndpointer(dtype=np.int64, flags="C_CONTIGUOUS")] * 2,
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                *[np.ctypeslib.ndpointer(dtype=np.int64, flags="C_CONTIGUOUS")] * 2,
            ]
            function.restype = None

    @staticmethod
    def _safe(lower, upper, terms):
        bound = I64_MAX // terms
        return (
            int(lower.min()) >= -bound and int(lower.max()) <= bound
            and int(upper.min()) >= -bound and int(upper.max()) <= bound
        )

    def logits(self, matrix, site, theta, q):
        n, length = matrix.shape
        packed = np.r_[theta[q:].reshape(q, length - 1, q).transpose(1, 2, 0).ravel(), theta[:q]]
        for bits in (52, 48, 44, 40):
            scaled = np.ldexp(packed, bits)
            # Scaling upward by a power of two is exact if finite and in range.
            if not np.isfinite(scaled).all() or np.max(np.abs(scaled)) >= 2.0**62:
                continue
            lower = np.floor(scaled).astype(np.int64)
            upper = np.ceil(scaled).astype(np.int64)
            if self._safe(lower, upper, length):
                lo, hi = (np.empty((n, q), dtype=np.int64) for _ in range(2))
                self.library.grid_logits(matrix, lower, upper, n, length, q, site, lo, hi)
                return lo, hi, bits
        raise OverflowError("Cannot safely sum the dyadic logits")

    def gradients(self, matrix, q, site, lower, upper):
        n, length = matrix.shape
        if not self._safe(lower, upper, n):
            raise OverflowError("Cannot safely sum the dyadic gradient contributions")
        size = potts_fit.parameter_count(length, q)
        lo, hi = (np.zeros(size, dtype=np.int64) for _ in range(2))
        self.library.grid_gradients(matrix, lower, upper, n, length, q, site, lo, hi)
        return tuple(np.r_[x[-q:], x[:-q].reshape(length - 1, q, q).transpose(2, 0, 1).ravel()] for x in (lo, hi))


def row_residual_bounds(iv, log_lo, log_hi, bits, states, weights):
    """Decimal softmax for every row, output rounded outward to a dyadic grid."""
    low, high = iv.lo, iv.hi
    exp, prev, next_ = iv.nearest.exp, low.next_minus, high.next_plus
    step = Decimal.from_float(2.0**-bits)
    factor = Decimal(1 << ROW_GRID_BITS)
    exact_weights = {float(x): Decimal.from_float(float(x)) for x in np.unique(weights)}
    lo_arr, hi_arr = (np.empty_like(log_lo) for _ in range(2))
    for r, (il, ih, state, weight) in enumerate(zip(log_lo.tolist(), log_hi.tolist(), states.tolist(), weights.tolist())):
        shift = max(ih)
        exp_lo, exp_hi = [], []
        dl, du = ZERO, ZERO
        for l, u in zip(il, ih):
            z = low.multiply(Decimal(l - shift), step)
            zu = high.multiply(Decimal(u - shift), step)
            width = high.subtract(zu, z)
            if not ZERO <= width < 1:
                raise ArithmeticError("Invalid logit enclosure width")
            value = exp(z)
            lo = max(ZERO, prev(value))
            # e**width <= 1/(1-width) for 0 <= width < 1.
            hi = high.divide(next_(value), low.subtract(Decimal(1), width))
            exp_lo.append(lo)
            exp_hi.append(hi)
            dl, du = low.add(dl, lo), high.add(du, hi)
        if dl <= ZERO:
            raise ArithmeticError("Cannot certify a positive softmax denominator")
        w = exact_weights[weight]
        for a, (l, h) in enumerate(zip(exp_lo, exp_hi)):
            lg, hg = low.divide(l, du), high.divide(h, dl)
            if a == state:
                lg, hg = low.subtract(lg, Decimal(1)), high.subtract(hg, Decimal(1))
            lo_scalar = low.multiply(low.multiply(lg, w), factor)
            hi_scalar = high.multiply(high.multiply(hg, w), factor)
            lo_arr[r, a] = int(lo_scalar.to_integral_value(rounding=ROUND_FLOOR))
            hi_arr[r, a] = int(hi_scalar.to_integral_value(rounding=ROUND_CEILING))
    return lo_arr, hi_arr


def gradient_bound(grid, matrix, weights, total, site, theta, q):
    if (
        matrix.shape[0] == 0 or matrix.ndim != 2 or matrix.dtype != np.int8
        or not matrix.flags.c_contiguous or int(matrix.min()) < 0 or int(matrix.max()) >= q
        or not 0 <= site < matrix.shape[1] or weights.shape != (len(matrix),)
        or not np.isfinite(weights).all() or float(weights.min()) < 0
        or not np.isfinite(total) or total <= 0 or not np.isfinite(theta).all()
        or theta.shape != (potts_fit.parameter_count(matrix.shape[1], q),)
        or float(weights.max()) > math.ldexp(1.0, 62 - ROW_GRID_BITS)
    ):
        raise ValueError("Inputs fall outside the exact grid contract")
    iv = Intervals()
    start = perf_counter()
    lo, hi, bits = grid.logits(matrix, site, theta, q)
    logits_seconds = perf_counter() - start
    start = perf_counter()
    rlo, rhi = row_residual_bounds(iv, lo, hi, bits, matrix[:, site], weights)
    decimal_seconds = perf_counter() - start
    start = perf_counter()
    glo, ghi = grid.gradients(matrix, q, site, rlo, rhi)
    unit = Decimal.from_float(2.0**-ROW_GRID_BITS)
    scale = iv.scale_positive(point(total), Decimal(2))
    field = iv.mul(scale, point(potts_fit.LAMBDA_H))
    coupling = iv.mul(scale, point(potts_fit.LAMBDA_J_PRIME))
    result = [
        iv.add((iv.lo.multiply(Decimal(l), unit), iv.hi.multiply(Decimal(h), unit)),
               iv.mul(point(t), field if i < q else coupling))
        for i, (l,h,t) in enumerate(zip(glo.tolist(),ghi.tolist(),theta))
    ]
    return iv, result, min(field[0], coupling[0]), bits, {
        "logits": logits_seconds, "decimal_rows": decimal_seconds,
        "gradient_sum_and_regularizer": perf_counter() - start,
    }


def evaluate_site(site):
    start = perf_counter()
    theta, digest = INPUT.theta(site)
    iv, enclosure, kappa, bits, times = gradient_bound(GRID, INPUT.matrix, INPUT.weights, INPUT.total, site, theta, 21)
    start_b = perf_counter()
    library = KERNEL.library_for_worker()
    _, _, g = potts_fit.site_terms_c(theta, INPUT.matrix, INPUT.weights, site, INPUT.total, library=library)
    if not np.isfinite(g).all() or KERNEL.objective_team_size() != 1:
        raise RuntimeError("Could not compare the B floating point gradient")
    times["b_gradient"] = perf_counter() - start_b
    eta = max(iv.discrepancy(x, ivl) for x, ivl in zip(g, enclosure))
    bound = iv.sqrt(iv.sum(iv.square(x) for x in enclosure))[1]
    return {
        "site": site, "source_parameter_sha256": digest, "theta_grid_bits": bits,
        "decimal_precision": iv.nearest.prec, "contribution_grid_bits": ROW_GRID_BITS,
        "b_gradient_inf": float(np.max(np.abs(g))),
        "eta_upper": str(eta), "real_gradient_l2_upper": str(bound),
        "strong_convexity_lower": str(kappa), "parameter_radius_upper": str(iv.hi.divide(bound, kappa)),
        "seconds": {**times, "worker_total": perf_counter() - start},
        "process_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }


def main():
    global INPUT, KERNEL, GRID
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Read-only mirf-muformer source root")
    parser.add_argument("--output", type=Path, required=True, help="New JSON file outside the source root")
    parser.add_argument("--sites", required=True, help="Comma-separated zero-based sites, or all")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--progress", type=Path, help="New scratch subdirectory for per-site bound records")
    args = parser.parse_args()
    sites = list(range(263)) if args.sites == "all" else [int(x) for x in args.sites.split(",")]
    if not sites or not set(sites) <= set(range(263)) or len(set(sites)) != len(sites) or not 1 <= args.workers <= len(sites):
        parser.error("Invalid site list or worker count")
    if args.output.exists() or args.output.resolve().is_relative_to(args.source.resolve()):
        parser.error("Output must be new and outside the source")
    if args.progress is not None:
        if args.progress.exists() or args.progress.resolve().is_relative_to(args.source.resolve()):
            parser.error("Progress directory must be new and outside the source")
        args.progress.mkdir()
    INPUT = Sources(args.source.resolve())
    start = perf_counter()
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp)
        metadata = build(path / "b")
        KERNEL = BoundBKernel(path / "b", expected_sha256=metadata["artifact_sha256"])
        GRID = Grid(path)
        results = []
        def collect(row):
            results.append(row)
            if args.progress is not None:
                path = args.progress / f"site_{row['site']:03d}.json"
                record = {
                    "frozen_identity": INPUT.identity,
                    "weights_total": INPUT.total,
                    "b_sha256": metadata["artifact_sha256"],
                    "grid": GRID.identity,
                    "sites": [row],
                }
                pending = path.with_suffix(".pending")
                pending.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
                pending.replace(path)
                print(f"{len(results)}/{len(sites)} saved: site {row['site']:03d}", flush=True)
        if args.workers == 1:
            for site in sites:
                collect(evaluate_site(site))
        else:
            with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("fork")) as pool:
                for future in as_completed([pool.submit(evaluate_site, site) for site in sites]):
                    collect(future.result())
        results.sort(key=lambda row: row["site"])
    result = {
        "scope": "Read-only real point-estimate bounds; no optimizer, refit, or modified weights",
        "shape": list(INPUT.matrix.shape), "frozen_identity": INPUT.identity,
        "weights_total": INPUT.total, "worker_count": args.workers,
        "b_sha256": metadata["artifact_sha256"], "grid": GRID.identity,
        "wall_seconds_including_build": perf_counter() - start, "sites": results,
    }
    with args.output.open("x") as file:
        json.dump(result, file, indent=2, allow_nan=False)
        file.write("\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

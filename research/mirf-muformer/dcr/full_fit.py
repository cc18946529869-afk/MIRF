"""Fit all 263 sites with the Solver v2 protocol verified on five sites.

Each site uses one OpenMP thread and is fit twice from zero. A site that misses
the gradient test is kept for diagnosis and is not part of an official point
estimate. This script does not change the objective, tolerances, or iteration
limits.
"""

from __future__ import annotations

import json
import multiprocessing
import os
from pathlib import Path

import numpy as np

import potts_fit
import solver_v2

OUTPUT = Path(__file__).resolve().parent / "full_fit"
WORKERS = 40
SITE_COUNT = 263


def _passes(row: dict, finite_parameters: bool) -> bool:
    return bool(
        row["status"] == "converged"
        and row["converged"]
        and row["grad_inf"] <= potts_fit.OPT_TOL
        and np.isfinite(row["grad_inf"])
        and np.isfinite(row["objective"])
        and np.isfinite(row["objective_accumulated"])
        and row["unexpected_increases"] == 0
        and row["repeat_status"] == "converged"
        and row["repeat_max_abs_difference"] == 0.0
        and np.isfinite(row["repeat_grad_inf"])
        and row["repeat_grad_inf"] <= potts_fit.OPT_TOL
        and row["nonfinite_directions"] == 0
        and row["ascent_directions"] == 0
        and row["iterations"] <= potts_fit.MAX_ITER
        and row["evaluations"] <= potts_fit.MAX_FUN_EVALS
        and finite_parameters
    )


def main() -> None:
    alignment, _weights, _total = potts_fit.load_official_alignment()
    if alignment.shape[1] != SITE_COUNT:
        raise RuntimeError("the full fit requires the 263-column alignment")
    os.environ["POTTS_OUTPUT"] = str(OUTPUT)
    os.environ["POTTS_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    sites = list(range(SITE_COUNT))
    context = multiprocessing.get_context("spawn")
    rows = []
    progress = OUTPUT / "progress.jsonl"
    with context.Pool(WORKERS) as pool:
        for row in pool.imap_unordered(solver_v2._fit_one, sites):
            rows.append(row)
            with progress.open("a") as handle:
                handle.write(json.dumps(solver_v2._json_ready(row)) + "\n")
            print(
                f"site {row['site']:3d}  {row['status']}  grad {row['grad_inf']:.3e}  "
                f"iter {row['iterations']}  eval {row['evaluations']}  "
                f"repeat {row['repeat_max_abs_difference']:.3e}",
                flush=True,
            )
    rows.sort(key=lambda row: int(row["site"]))
    if [int(row["site"]) for row in rows] != sites:
        raise RuntimeError("the full fit did not return every site")
    for row in rows:
        saved = np.load(OUTPUT / f"site_{row['site']:03d}.npz")
        finite = bool(
            np.all(np.isfinite(saved["theta"]))
            and np.all(np.isfinite(saved["h"]))
            and np.all(np.isfinite(saved["couplings"]))
        )
        row["finite_parameters"] = finite
        row["passed"] = _passes(row, finite)
    accepted = all(row["passed"] for row in rows)
    report = {
        "accepted_point_estimate": accepted,
        "opt_tol": potts_fit.OPT_TOL,
        "max_iter": potts_fit.MAX_ITER,
        "max_fun_evals": potts_fit.MAX_FUN_EVALS,
        "threads": 1,
        "repeats": 2,
        "site_count": SITE_COUNT,
        "n_passed": sum(row["passed"] for row in rows),
        "failed_sites": [int(row["site"]) for row in rows if not row["passed"]],
        "sites": rows,
    }
    (OUTPUT / "summary.json").write_text(json.dumps(solver_v2._json_ready(report), indent=2) + "\n")
    print(
        f"passed {report['n_passed']}/{SITE_COUNT}  accepted_point_estimate {accepted}",
        flush=True,
    )


if __name__ == "__main__":
    main()

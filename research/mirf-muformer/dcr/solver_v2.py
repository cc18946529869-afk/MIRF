"""Run Solver v2 on the five named sites and check gauge-mapped components.

The 263-site fit is refused. Partner directions outside these five sites stay
the previously saved, unaccepted parameters. The component numbers are a
diagnostic, not an accepted point estimate.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import energy_components
import potts_fit

SITES = potts_fit.SOLVER_V2_SITES
OUTPUT = Path(__file__).resolve().parent / "solver_v2_delta"
PREVIOUS = Path(__file__).resolve().parent / "solver_v2"


def _load_saved_site(path: Path) -> tuple[np.ndarray, np.ndarray]:
    saved = np.load(path)
    return np.asarray(saved["h"], dtype=np.float64), np.asarray(saved["couplings"], dtype=np.float64)


def _background(alignment: np.ndarray, weights: np.ndarray, weight_total: float, q: int) -> np.ndarray:
    length = alignment.shape[1]
    frequency = np.zeros((length, q), dtype=np.float64)
    for column in range(length):
        codes = alignment[:, column]
        for state in range(q):
            frequency[column, state] = float(weights[codes == state].sum())
    frequency /= float(weight_total)
    pseudocount = 1.0 / (float(weight_total) + 1.0)
    return (float(weight_total) * pseudocount) * frequency + pseudocount / float(q)


def _lobe_groups(domain_map: Path, length: int) -> list[np.ndarray]:
    groups = {0: [], 1: []}
    with domain_map.open() as handle:
        header = handle.readline().rstrip("\n").split("\t")
        index = {name: position for position, name in enumerate(header)}
        for line in handle:
            parts = line.rstrip("\n").split("\t")
            if parts[index["mapped"]] != "1":
                continue
            column = int(parts[index["mature_pos"]]) - 1
            groups[int(parts[index["lobe_id"]])].append(column)
    covered = sorted(groups[0] + groups[1])
    if covered != list(range(length)):
        raise ValueError("the lobe map does not partition the mature sites")
    return [np.asarray(groups[0], dtype=int), np.asarray(groups[1], dtype=int)]


def _json_ready(value):
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _fit_one(site: int) -> dict:
    os.environ["POTTS_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    potts_fit.set_objective_threads(1)
    alignment, weights, total = potts_fit.load_official_alignment()
    started = time.perf_counter()
    first = potts_fit.fit_site(alignment, weights, site, total)
    print(
        f"site {site} first {first.status} grad {first.grad_inf:.3e} "
        f"iter {first.iterations} eval {first.evaluations} seconds {time.perf_counter() - started:.1f}",
        flush=True,
    )
    second = potts_fit.fit_site(alignment, weights, site, total)
    difference = float(np.max(np.abs(first.theta - second.theta)))
    output = OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    np.savez(
        output / f"site_{site:03d}.npz",
        h=first.h,
        couplings=first.couplings,
        theta=first.theta,
        status=np.array(first.status),
        objective=np.float64(first.objective),
        objective_accumulated=np.float64(first.objective_accumulated),
        nll=np.float64(first.nll),
        penalty=np.float64(first.penalty),
        grad_inf=np.float64(first.grad_inf),
        iterations=np.int32(first.iterations),
        evaluations=np.int32(first.evaluations),
        message=np.array(first.message),
        n_strong_wolfe=np.int32(first.n_strong_wolfe),
        n_armijo_fallback=np.int32(first.n_armijo_fallback),
        n_curvature_skipped=np.int32(first.n_curvature_skipped),
        curvature_sy=np.asarray(first.curvature_sy, dtype=np.float64),
        repeat_max_abs_difference=np.float64(difference),
        repeat_status=np.array(second.status),
        repeat_grad_inf=np.float64(second.grad_inf),
    )
    (output / f"site_{site:03d}_trace.json").write_text(json.dumps(_json_ready(first.trace)))
    accepted = [row for row in first.trace if row.get("event") == "accepted"]
    directions = [row for row in first.trace if row.get("event") == "direction"]
    increases = [row["delta"] for row in accepted if row["delta"] > 0.0]
    rhos = [row["rho"] for row in accepted if row.get("rho") is not None]
    cosines = [row["cos_phi"] for row in accepted if row.get("cos_phi") is not None]
    exits = [row.get("line_search_exit") for row in first.trace if row.get("line_search_exit")]
    return {
        "site": site,
        "status": first.status,
        "converged": first.converged,
        "grad_inf": first.grad_inf,
        "objective": first.objective,
        "objective_accumulated": first.objective_accumulated,
        "iterations": first.iterations,
        "evaluations": first.evaluations,
        "n_strong_wolfe": first.n_strong_wolfe,
        "n_armijo_fallback": first.n_armijo_fallback,
        "n_curvature_skipped": first.n_curvature_skipped,
        "curvature_sy_min": min(first.curvature_sy) if first.curvature_sy else None,
        "curvature_sy_max": max(first.curvature_sy) if first.curvature_sy else None,
        "rho_max": max(rhos) if rhos else None,
        "cos_phi_min": min(cosines) if cosines else None,
        "direction_l2_max": max(row["direction_l2"] for row in directions) if directions else None,
        "nonfinite_directions": sum(not row.get("direction_finite", False) for row in directions),
        "ascent_directions": sum(not row.get("descent", False) for row in directions),
        "line_search_exits": exits,
        "positive_curvature_steps": sum(value > 0.0 for value in first.curvature_sy),
        "nonpositive_curvature_steps": sum(value <= 0.0 for value in first.curvature_sy),
        "unexpected_increases": len(increases),
        "repeat_status": second.status,
        "repeat_grad_inf": second.grad_inf,
        "repeat_max_abs_difference": difference,
        "seconds": time.perf_counter() - started,
        "message": first.message,
    }


def _component_report(alignment: np.ndarray, weights: np.ndarray, weight_total: float) -> dict:
    length = alignment.shape[1]
    q = potts_fit.Q
    old_fields = []
    old_couplings = []
    new_fields = []
    new_couplings = []
    saved_dir = Path(__file__).resolve().parent / "point_estimate"
    for site in range(length):
        fields, couplings = _load_saved_site(saved_dir / f"site_{site:03d}.npz")
        old_fields.append(fields)
        old_couplings.append(couplings)
        new_fields.append(fields)
        new_couplings.append(couplings)
    for site in SITES:
        fields, couplings = _load_saved_site(OUTPUT / f"site_{site:03d}.npz")
        new_fields[site] = fields
        new_couplings[site] = couplings
    background = _background(alignment, weights, weight_total, q)
    groups = _lobe_groups(Path(__file__).resolve().parents[1] / "prep" / "domain_map.tsv", length)
    sequence = alignment[0]
    rows = []
    max_identity_error = 0.0
    for site in SITES:
        for new_state in range(q):
            if new_state == int(sequence[site]):
                continue
            old_parts = energy_components.components_from_directional(
                sequence, site, new_state, old_fields, old_couplings, background, groups
            )
            new_parts = energy_components.components_from_directional(
                sequence, site, new_state, new_fields, new_couplings, background, groups
            )
            old_sum = old_parts[0] + sum(old_parts[1])
            new_sum = new_parts[0] + sum(new_parts[1])
            max_identity_error = max(
                max_identity_error,
                abs(old_sum - old_parts[2]),
                abs(new_sum - new_parts[2]),
            )
            rows.append(
                {
                    "site": site,
                    "new_state": new_state,
                    "background_difference": new_parts[0] - old_parts[0],
                    "group_differences": [new_item - old_item for new_item, old_item in zip(new_parts[1], old_parts[1])],
                    "energy_difference": new_parts[2] - old_parts[2],
                }
            )
    return {
        "accepted_point_estimate": False,
        "sites_refit": list(SITES),
        "partner_directions_outside_these_sites": "unaccepted_saved_parameters",
        "max_component_sum_error": max_identity_error,
        "max_abs_background_difference": max(abs(row["background_difference"]) for row in rows),
        "max_abs_group_difference": max(abs(item) for row in rows for item in row["group_differences"]),
        "max_abs_energy_difference": max(abs(row["energy_difference"]) for row in rows),
        "mutations": rows,
    }


def main() -> None:
    if tuple(SITES) != (12, 95, 103, 208, 256) or len(SITES) == 263:
        raise RuntimeError("Solver v2 is not authorized to fit every site")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    import multiprocessing

    context = multiprocessing.get_context("spawn")
    with context.Pool(len(SITES)) as pool:
        rows = list(pool.map(_fit_one, SITES))
    alignment, weights, total = potts_fit.load_official_alignment()
    previous = {}
    previous_path = PREVIOUS / "summary.json"
    if previous_path.is_file():
        for row in json.loads(previous_path.read_text())["sites"]:
            previous[int(row["site"])] = row
    for row in rows:
        earlier = previous.get(int(row["site"]))
        if earlier is not None:
            row["previous_grad_inf"] = earlier["grad_inf"]
            row["previous_status"] = earlier["status"]
            row["previous_iterations"] = earlier["iterations"]
            row["grad_inf_difference"] = row["grad_inf"] - earlier["grad_inf"]
    report = {
        "opt_tol": potts_fit.OPT_TOL,
        "max_iter": potts_fit.MAX_ITER,
        "max_fun_evals": potts_fit.MAX_FUN_EVALS,
        "threads": 1,
        "sites": rows,
        "components": _component_report(alignment, weights, total),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(_json_ready(report), indent=2))
    for row in rows:
        print(
            f"site {row['site']:3d}  {row['status']}  grad {row['grad_inf']:.3e}  "
            f"wolfe {row['n_strong_wolfe']}  fallback {row['n_armijo_fallback']}  "
            f"repeat {row['repeat_max_abs_difference']:.3e}",
            flush=True,
        )


if __name__ == "__main__":
    main()

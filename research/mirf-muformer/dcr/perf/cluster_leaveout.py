"""One authorized observed-data cluster deletion using the frozen Potts objective."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path
from time import perf_counter

import numpy as np

from bootstrap_gauge import assemble_directional, gauge_parameters
import potts_fit
from .audit_saved_bootstrap_stability import apc, groups, load_arrays


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def save_json(path, record):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, allow_nan=False)
        handle.write("\n")


def run_site(site):
    start = perf_counter()
    initial = []
    if initial_parameters is None:
        fit = potts_fit.fit_site(rows, retained_weights, site, weight_total)
        theta = fit.theta
        extra = 2
    else:
        def objective(theta):
            nll, penalty, gradient = potts_fit.site_terms_c(theta, rows, retained_weights, site, weight_total)
            if not initial:
                initial.append(float(np.linalg.norm(gradient, ord=np.inf)))
            return nll + penalty, gradient

        direct_delta = potts_fit.make_direct_delta(rows, retained_weights, site, weight_total)
        fit = potts_fit.lbfgs(objective, initial_parameters[site], delta_fn=direct_delta)
        theta = fit.x
        extra = 1
    nll, penalty, gradient = potts_fit.site_terms_c(theta, rows, retained_weights, site, weight_total)
    finite = bool(np.isfinite(theta).all() and np.isfinite(gradient).all()
                  and np.isfinite([nll, penalty]).all())
    grad_max = float(np.linalg.norm(gradient, ord=np.inf)) if finite else None
    passed = fit.status == "converged" and finite and grad_max <= potts_fit.OPT_TOL
    destination = private_dir / f"site_{site:03d}.npz"
    temporary = destination.with_suffix(".tmp")
    with temporary.open("xb") as handle:
        np.savez(handle, theta=theta)
    temporary.rename(destination)
    return {
        "site": site,
        "status": fit.status,
        "message": fit.message,
        "terminal_gradient_inf": grad_max,
        "terminal_objective": nll + penalty if finite else None,
        "passed": passed,
        "iterations": fit.iterations,
        "function_evaluations_solver": fit.evaluations,
        "additional_terminal_evaluations": extra,
        "initial_gradient_inf": initial[0] if initial and np.isfinite(initial[0]) else None,
        "strong_wolfe": fit.n_strong_wolfe,
        "fallback": fit.n_armijo_fallback,
        "seconds": perf_counter() - start,
        "parameter_sha256": digest(destination),
    }


def relative_distance(candidate, reference):
    return float(np.linalg.norm((candidate - reference).ravel()) / np.linalg.norm(reference.ravel()))


def compare(source, certificate, output, manifest, sites):
    reference = source / "dcr/bootstrap/reference"
    h_ref, j_ref, hashes = load_arrays(reference)
    accepted = json.loads((reference / "manifest.json").read_text())
    cert = json.loads(certificate.read_text())
    if (not accepted["accepted_point_estimate"] or
        any(hashes[k] != accepted["gauge_sha256"][k] for k in hashes) or
        hashes["j_star.npy"] != cert["references_sha256"]["j_star.npy"] or
        digest(source / "dcr/contact_validation_summary.json") != cert["published_summary_sha256"]):
        raise ValueError("Full-data reference identity mismatch")
    fields, couplings = [], []
    for site in range(263):
        path = output / "private" / f"site_{site:03d}.npz"
        if digest(path) != sites[site]["parameter_sha256"]:
            raise ValueError(f"Incorrect terminal identity at site {site}")
        with np.load(path, allow_pickle=False) as values:
            field, coupling = potts_fit.unpack_parameters(values["theta"], 263)
            fields.append(field)
            couplings.append(coupling)
    h, j = gauge_parameters(*assemble_directional(np.array(fields), np.array(couplings)))
    candidate_score = apc(j)
    comparisons = groups(source, j_ref, cert)
    report_groups = {}
    for name, group in comparisons.items():
        group.add(candidate_score)
        report_groups[name] = {
            "reference_hits": group.reference_hits, "top_k": group.k, **group.rows[0],
            "leaveout_precision_at_l": group.rows[0]["contact_hits"] / group.k,
        }
    return {
        **manifest, "sites_planned": 263, "sites_passed": 263,
        "max_terminal_gradient_inf": max(s["terminal_gradient_inf"] for s in sites.values()),
        "summed_site_seconds": sum(s["seconds"] for s in sites.values()),
        "relative_h_displacement": relative_distance(h, h_ref),
        "relative_j_displacement": relative_distance(j, j_ref),
        "historical_r_j_estimable_from_one_deletion": False,
        "ranking_scope": "nominal binary64 ranking for deleted fit; full-data reference previously interval-certified",
        "ranking": report_groups, "reference_sha256": hashes,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--clusters", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--certificate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=40)
    parser.add_argument("--reference-start", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 40:
        parser.error("workers must be between 1 and 40")
    start = perf_counter()
    global rows, retained_weights, weight_total, private_dir, initial_parameters
    prep = args.source / "prep"
    matrix, weights, reference_total = potts_fit.load_official_alignment(prep)
    gate = json.loads(args.gate.read_text())
    entry = gate["granularities"]["211"]
    if any(digest(prep / name) != sha for name, sha in gate["sha256"].items()):
        raise ValueError("Alignment or weight table differs from the archived graph")
    if digest(args.clusters) != entry["cluster_assignments_sha256"]:
        raise ValueError("Wrong 211/263 component artifact")
    labels = np.load(args.clusters, allow_pickle=False)
    cluster_mass = np.bincount(labels, weights=weights)
    keep = labels != int(np.argmax(cluster_mass))
    removed = int(np.count_nonzero(~keep))
    removed_mass = float(weights[~keep].sum())
    if removed != entry["heaviest_component_rows"] or abs(removed_mass / reference_total - entry["largest_weight_share"]) > 1e-12:
        raise ValueError("Selected deletion does not match the audited heaviest component")
    rows = np.ascontiguousarray(matrix[keep])
    retained_weights = np.ascontiguousarray(weights[keep])
    weight_total = float(retained_weights.sum())
    initial_parameters = None
    if args.reference_start:
        manifest = json.loads((args.source / "dcr/bootstrap/reference/manifest.json").read_text())
        if not manifest["accepted_point_estimate"] or manifest["n_passed"] != 263:
            raise ValueError("Warm start requires the accepted full-data reference")
        initial_parameters = []
        for site in range(263):
            path = args.source / "dcr/full_fit" / f"site_{site:03d}.npz"
            if digest(path) != manifest["source_sha256"][path.name]:
                raise ValueError(f"Full-data parameter identity mismatch at site {site}")
            with np.load(path, allow_pickle=False) as values:
                theta = values["theta"]
            if theta.dtype != np.float64 or not np.isfinite(theta).all():
                raise ValueError(f"Invalid full-data start at site {site}")
            potts_fit.unpack_parameters(theta, 263)
            initial_parameters.append(theta)
    args.output.mkdir()
    private_dir = args.output / "private"
    private_dir.mkdir()
    potts_fit._compile_library()
    potts_fit.set_objective_threads(1)
    manifest = {
        "case": "A211-weight-max-deletion", "fit_kernel": "official A, one thread per worker",
        "workers": args.workers, "library_sha256": digest(potts_fit._LIBRARY),
        "no_resampling": True, "removed_rows": removed, "retained_rows": len(rows),
        "removed_reference_weight": removed_mass, "relative_removed_weight": removed_mass / reference_total,
        "weight_total": weight_total, "original_weight_total": reference_total,
        "initialization": "verified_full_data_parameter" if args.reference_start else "zero", "gradient_tolerance": potts_fit.OPT_TOL,
        "lambda_h": potts_fit.LAMBDA_H, "lambda_j_prime": potts_fit.LAMBDA_J_PRIME,
        "max_iter": potts_fit.MAX_ITER, "max_fun_evals": potts_fit.MAX_FUN_EVALS,
        "alignment_and_weights_sha256": gate["sha256"],
        "component_assignment_sha256": entry["cluster_assignments_sha256"],
    }
    save_json(args.output / "manifest.json", manifest)
    print(json.dumps(manifest), flush=True)
    sites = {}
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("fork")) as pool:
        tasks = {pool.submit(run_site, site): site for site in range(263)}
        with (args.output / "sites.jsonl").open("x", encoding="utf-8", buffering=1) as log:
            for task in as_completed(tasks):
                site = tasks[task]
                try:
                    row = task.result()
                except Exception as error:
                    row = {"site": site, "passed": False, "status": "worker_error",
                           "message": f"{type(error).__name__}: {error}"}
                sites[site] = row
                log.write(json.dumps(row, allow_nan=False) + "\n")
                print(f"{len(sites):03d}/263 site={site:03d} {row['status']} passed={row['passed']}", flush=True)
    failed = sorted(site for site, record in sites.items() if not record["passed"])
    result = ({**manifest, "sites_planned": 263, "failed_sites": failed, "sites_passed": 263 - len(failed)}
              if failed else compare(args.source, args.certificate, args.output, manifest, sites))
    result["wall_seconds"] = perf_counter() - start
    save_json(args.output / "result.json", result)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()

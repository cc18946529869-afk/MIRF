"""Certify the published ranking using full-site radii; exports counts, not rankings."""

import argparse
from bisect import bisect_left, bisect_right
from concurrent.futures import ProcessPoolExecutor
from decimal import Decimal
import json
import multiprocessing
from pathlib import Path
import resource
from time import perf_counter

import numpy as np

import contact_validation as contacts
import potts_fit
from .certify_saved_sites import Sources, same_hash, sha256
from .check_top263_synthetic import Intervals, ZERO, apc_enclosure, point, raw_enclosure

DIRECTIONAL = None
NUMERIC = None
RADII = None
N = 263


def raw_row(i):
    iv = Intervals()
    row = []
    xi = ZERO
    for j in range(i + 1, N):
        nominal = raw_enclosure(iv, DIRECTIONAL[i, j], DIRECTIONAL[j, i])
        xi = max(xi, iv.discrepancy(NUMERIC[i, j], nominal))
        if RADII is not None:
            radius = iv.hi.divide(iv.hi.add(RADII[i], RADII[j]), Decimal(2))
            row.append((max(ZERO, iv.lo.subtract(nominal[0], radius)), iv.hi.add(nominal[1], radius)))
    return i, row, xi


def assess_group(left, right, numeric, intervals, contact):
    count = len(left)
    if not count:
        raise ValueError("Empty published group")
    k = min(N, count)
    ordered = contacts.ranking_order(left, right, numeric[left, right])
    top = ordered[:k]
    gap = None if count == k else float(numeric[left[ordered[k-1]], right[ordered[k-1]]] -
                                        numeric[left[ordered[k]], right[ordered[k]]])
    hits = int(contact[left[top], right[top]].sum())
    record = {
        "candidate_pairs": count, "top_k": k, "numeric_hits": hits, "numeric_precision": hits / k,
        "numeric_boundary_gap": gap, "interval_boundary_slack_lower": None,
        "certified_numeric_top_set": False, "definite_in": None, "definite_out": None,
        "indeterminate_candidates": None, "uncertified_numeric_top_members": None,
    }
    if intervals is None:
        record["reason"] = "no_complete_finite_apc_enclosure"
        return record
    iv = Intervals()
    pair_intervals = [intervals[i][j] for i, j in zip(left, right)]
    slack = None if count == k else iv.lo.subtract(
        min(pair_intervals[i][0] for i in top),
        max(pair_intervals[i][1] for i in ordered[k:]),
    )
    if count == k or slack > ZERO:
        record.update(certified_numeric_top_set=True, definite_in=k, definite_out=count-k,
                      indeterminate_candidates=0, uncertified_numeric_top_members=0,
                      precision_hit_count_interval=[hits, hits], precision_interval=[hits / k, hits / k],
                      reason="strict_top_set_separation")
    else:
        lowers = sorted(x[0] for x in pair_intervals)
        uppers = sorted(x[1] for x in pair_intervals)
        # Upper >= own lower is potentially ahead; lower > own upper is certainly ahead.
        inside = [count - bisect_left(uppers, x[0]) - 1 < k for x in pair_intervals]
        outside = [count - bisect_right(lowers, x[1]) >= k for x in pair_intervals]
        labels = contact[left, right]
        confirmed_hits = sum(bool(c) and yes for c, yes in zip(labels, inside))
        possible_hits = sum(bool(c) and not out and not yes for c, out, yes in zip(labels, outside, inside))
        room = k - sum(inside)
        possible_nonhits = count - sum(outside) - sum(inside) - possible_hits
        hit_lo = confirmed_hits + max(0, room - possible_nonhits)
        hit_hi = confirmed_hits + min(room, possible_hits)
        if hit_lo > hit_hi:
            raise ArithmeticError("Inconsistent candidate count enclosure")
        record.update(
            definite_in=sum(inside), definite_out=sum(outside),
            indeterminate_candidates=count-sum(inside)-sum(outside),
            uncertified_numeric_top_members=k-sum(inside[i] for i in top),
            precision_hit_count_interval=[hit_lo, hit_hi],
            precision_interval=[hit_lo / k, hit_hi / k],
            reason="boundary_intervals_overlap_or_touch",
        )
    record["interval_boundary_slack_lower"] = None if slack is None else str(slack)
    return record


def main():
    global DIRECTIONAL, NUMERIC, RADII
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--site-bounds", required=True, type=Path, nargs="+")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to(args.source.resolve()):
        parser.error("Output must be new and outside the source")
    if not 1 <= args.workers <= N:
        parser.error("Invalid worker count")
    source = Sources(args.source.resolve())
    radii = {}
    bound_hashes = {}
    for path in args.site_bounds:
        saved = json.loads(path.read_text())
        bound_hashes[path.name] = sha256(path)
        if saved["frozen_identity"] != source.identity or saved["weights_total"] != source.total:
            raise ValueError("Site bounds refer to different frozen inputs")
        for row in saved["sites"]:
            site = row["site"]
            if site in radii or not 0 <= site < N:
                raise ValueError("Unexpected or duplicate site radius")
            radius = Decimal(row["parameter_radius_upper"])
            if not radius.is_finite() or radius < ZERO:
                raise ValueError("Invalid site radius")
            _, digest = source.theta(site)
            if row["source_parameter_sha256"] != digest:
                raise ValueError("Site bounds refer to different point-estimate parameters")
            radii[site] = radius
    if len(radii) == N:
        RADII = [radii[i] for i in range(N)]

    identities = {}
    for name in ("directional.npy", "j_star.npy", "fields.npy"):
        path = source.reference / name
        identities[name] = same_hash(path, source.manifest["gauge_sha256"][name])
    DIRECTIONAL = np.load(source.reference / "directional.npy", mmap_mode="r", allow_pickle=False)
    fields = np.load(source.reference / "fields.npy", mmap_mode="r", allow_pickle=False)
    jstar = np.load(source.reference / "j_star.npy", mmap_mode="r", allow_pickle=False)
    if DIRECTIONAL.shape != (N, N, 21, 21) or jstar.shape != DIRECTIONAL.shape:
        raise ValueError("Invalid saved artifact shape")
    for site in range(N):
        theta, _ = source.theta(site)
        h, j = potts_fit.unpack_parameters(theta, N)
        blocks = np.concatenate((DIRECTIONAL[site, :site], DIRECTIONAL[site, site+1:]))
        if not np.array_equal(fields[site], h) or not np.array_equal(blocks, j.transpose(1, 0, 2)):
            raise ValueError(f"Reference directional blocks differ from point estimate at site {site}")
    NUMERIC = contacts.amino_block_norm(jstar)
    numeric_apc = contacts.average_product_correction(NUMERIC)
    coords, mismatched = contacts.site_points(source.prep / "domain_map.tsv", source.prep / "1BTL.pdb")
    contact = contacts.contact_matrix(coords)
    left, right = contacts.pair_table(~mismatched)
    separation = right - left
    groups = {f"separation_at_least_{cut}": separation >= cut for cut in contacts.SEPARATION_CUTS}
    for low, high in contacts.SEPARATION_BINS:
        key = f"separation_{low}_and_above" if high is None else f"separation_{low}_to_{high}"
        groups[key] = separation >= low if high is None else (separation >= low) & (separation <= high)

    start = perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("fork")) as pool:
        rows = list(pool.map(raw_row, range(N)))
    raw_seconds = perf_counter() - start
    max_xi = max(row[2] for row in rows)
    start = perf_counter()
    corrected = None
    normalization = "unbounded_missing_site_radii"
    if RADII is not None:
        raw = [[point(0)] * N for _ in range(N)]
        for i, row, _ in rows:
            for j, bound in enumerate(row, start=i+1):
                raw[i][j] = raw[j][i] = bound
        corrected, normalization = apc_enclosure(Intervals(), raw)
        # Published nominal scores have positive background; this rare branch cannot
        # certify their ranking as the exact zero-background tie rule is still pending.
        if normalization == "exact-zero":
            corrected = None
    results = {key: assess_group(left[mask], right[mask], numeric_apc, corrected, contact) for key, mask in groups.items()}
    published_path = source.root / "dcr/contact_validation_summary.json"
    published = json.loads(published_path.read_text())["main"]
    for k, result in results.items():
        if (
            result["candidate_pairs"] != published[k]["pairs"]
            or result["numeric_precision"] != published[k]["precision_at_l"]
            or int(contact[left[groups[k]], right[groups[k]]].sum()) != published[k]["contacts"]
        ):
            raise ValueError(f"Published numeric ranking disagrees for {k}")
    report = {
        "scope": "Saved real point estimate; full global APC prior to published group mask; no optimizer or refit",
        "references_sha256": identities,
        "source_site_bounds_sha256": bound_hashes,
        "published_summary_sha256": sha256(published_path),
        "valid_parameter_radii": len(radii), "total_sites": N,
        "workers": args.workers,
        "max_xi_raw_upper": str(max_xi),
        "apc_enclosure_status": normalization,
        "certified_groups": sum(r["certified_numeric_top_set"] for r in results.values()),
        "groups_total": len(results), "groups": results,
        "seconds": {"raw_reference": raw_seconds, "apc_and_group_bounds": perf_counter()-start},
        "parent_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    with args.output.open("x") as file:
        json.dump(report, file, indent=2, allow_nan=False)
        file.write("\n")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

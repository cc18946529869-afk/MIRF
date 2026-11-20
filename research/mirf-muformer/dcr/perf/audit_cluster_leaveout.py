"""Validate the deletion fit from its archived terminals and frozen labels."""
from __future__ import annotations

import argparse
import json
import hashlib
from pathlib import Path

import numpy as np

import contact_validation as contacts


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def scores(couplings):
    # A direct contraction for the amino-acid scores, with full-matrix APC.
    amino = couplings[:, :, 1:, 1:]
    raw = np.sqrt(np.einsum("ijab,ijab->ij", amino, amino, optimize=False))
    row_sum = raw.sum(axis=1)
    if not row_sum.sum() > 0:
        raise ValueError("Expected a positive APC background for this diagnostic")
    correction = np.outer(row_sum, row_sum) * (len(row_sum) / ((len(row_sum) - 1) * row_sum.sum()))
    return raw - correction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fitdir", type=Path)
    parser.add_argument("source", type=Path)
    args = parser.parse_args()
    result = json.loads((args.fitdir / "result.json").read_text())
    log = [json.loads(line) for line in (args.fitdir / "sites.jsonl").read_text().splitlines()]
    sites = {x["site"]: x for x in log}
    if len(log) != 263 or sorted(sites) != list(range(263)):
        raise ValueError("Expected exactly one terminal ledger record per site")
    passed = {site for site, x in sites.items() if x["passed"]}
    assert len(passed) == result["sites_passed"]
    if len(passed) < 263:
        print(json.dumps({"sites_passed": len(passed), "failed_sites": sorted(set(sites) - passed),
                          "nominal_aggregate_undefined": True}, indent=2))
        return
    direction = np.zeros((263, 263, 21, 21))
    h = np.empty((263, 21))
    for i, entry in sorted(sites.items()):
        path = args.fitdir / "private" / f"site_{i:03d}.npz"
        assert digest(path) == entry["parameter_sha256"]
        assert entry["status"] == "converged" and entry["terminal_gradient_inf"] <= result["gradient_tolerance"]
        with np.load(path, allow_pickle=False) as values:
            theta = values["theta"]
        assert theta.shape == (21 + 21 * 262 * 21,) and np.isfinite(theta).all()
        direction[i, np.arange(263) != i] = theta[21:].reshape(21, 262, 21).swapaxes(0, 1)
        h[i] = theta[:21] + direction[i].mean(axis=2).sum(axis=0)
    center = np.eye(21) - np.ones((21, 21)) / 21
    standardized = (center @ direction) @ center
    j = 0.5 * (standardized + standardized.transpose(1, 0, 3, 2))

    ref_root = args.source / "dcr/bootstrap/reference"
    ref_h, ref_j = (np.load(ref_root / n, allow_pickle=False, mmap_mode="r") for n in ("h_hat.npy", "j_star.npy"))
    delta_h = float(np.linalg.norm(h - ref_h) / np.linalg.norm(ref_h))
    delta_j = float(np.linalg.norm(j - ref_j) / np.linalg.norm(ref_j))
    assert abs(delta_h - result["relative_h_displacement"]) < 1e-12
    assert abs(delta_j - result["relative_j_displacement"]) < 1e-12

    point, mismatch = contacts.site_points(args.source / "prep/domain_map.tsv", args.source / "prep/1BTL.pdb")
    labels = contacts.contact_matrix(point)
    before, after = scores(ref_j), scores(j)
    left, right = contacts.pair_table(~mismatch)
    distance = right - left
    tests = {
        "separation_at_least_6": distance >= 6,
        "separation_at_least_12": distance >= 12,
        "separation_6_to_11": (distance >= 6) & (distance <= 11),
        "separation_12_to_23": (distance >= 12) & (distance <= 23),
        "separation_24_and_above": distance >= 24,
    }
    output = {}
    for name, mask in tests.items():
        pairs = list(zip(left[mask], right[mask]))
        old = set(sorted(pairs, key=lambda p: (-before[p], p))[:263])
        new = set(sorted(pairs, key=lambda p: (-after[p], p))[:263])
        overlap = len(old & new)
        hits = sum(int(labels[p]) for p in new)
        assert overlap == result["ranking"][name]["overlap_with_reference"]
        assert hits == result["ranking"][name]["contact_hits"]
        output[name] = {"overlap_with_reference": overlap, "contact_hits": hits}
    index = np.arange(263)
    sep = np.abs(index[:, None] - index[None, :])
    squared = np.einsum("ijab,ijab->ij", j - ref_j, j - ref_j, optimize=False)
    ref_squared = np.einsum("ijab,ijab->ij", ref_j, ref_j, optimize=False)
    by_distance = {}
    for low, high in ((1, 5), (6, 11), (12, 23), (24, 262)):
        choose = (sep >= low) & (sep <= high)
        by_distance[f"{low}_{high}"] = float(np.sqrt(squared[choose].sum() / ref_squared[choose].sum()))
    print(json.dumps({"sites_passed": len(passed), "relative_h_displacement": delta_h,
                      "relative_j_displacement": delta_j, "relative_j_displacement_by_distance": by_distance,
                      "ranking": output}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

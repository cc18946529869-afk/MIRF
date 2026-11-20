"""Audit saved Bootstrap parameters and aggregate rankings without fitting or resampling."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

import contact_validation as contacts

N = 263
Q = 21
REPLICATES = 100
LOCK_FIELDS = ("formal_seed", "alignment_sha256", "neighbor_counts_sha256", "code_revision")
TOKEN_FIELDS = ("role", "replicate_id", "seed", *LOCK_FIELDS[1:], "indices_sha256", "draw_count", "weight_total")


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def validated_archive(root, replicate, lock):
    manifest = read_json(root / "manifest.json")
    if (
        any(manifest.get(key) != lock[key] for key in LOCK_FIELDS)
        or manifest.get("state") != "complete"
        or manifest.get("role") != "formal"
        or manifest.get("replicate_id") != replicate
        or manifest.get("formal_replicate_id") != replicate
        or manifest.get("seed") != lock["formal_seed"]
        or manifest.get("sites") != N
        or manifest.get("passed_sites") != N
        or manifest.get("failed_sites") != []
        or manifest.get("replicate_success") is not True
        or manifest.get("entered_formal_statistics") is not True
        or manifest.get("both_triangles") is not True
        or manifest.get("draw_count") != 47900
        or manifest.get("weight_total") != 47900.0
    ):
        raise ValueError(f"Invalid identity or completion state for replicate {replicate}")
    indices = np.load(root / "indices.npy", allow_pickle=False)
    if indices.dtype != np.int64 or indices.shape != (47900,):
        raise ValueError(f"Unexpected saved sampling-index array for replicate {replicate}")
    if hashlib.sha256(np.ascontiguousarray(indices).tobytes()).hexdigest() != manifest["indices_sha256"]:
        raise ValueError(f"Sampling-index hash differs for replicate {replicate}")
    raw = json.dumps({key: manifest[key] for key in TOKEN_FIELDS}, sort_keys=True, separators=(",", ":")).encode()
    if hashlib.sha256(raw).hexdigest() != manifest["identity_token"]:
        raise ValueError(f"Stored identity token differs for replicate {replicate}")
    return load_arrays(root)


def load_arrays(directory):
    files = {"h_hat.npy": (N, Q), "j_star.npy": (N, N, Q, Q)}
    result, hashes = [], {}
    for name, shape in files.items():
        path = directory / name
        hashes[name] = digest(path)
        data = np.load(path, mmap_mode="r", allow_pickle=False)
        if data.shape != shape or data.dtype != np.float64 or not np.isfinite(data).all():
            raise ValueError(f"Invalid exported parameter array: {path}")
        result.append(data)
    j = result[1]
    if not np.array_equal(j, j.transpose(1, 0, 3, 2)) or np.any(j[np.arange(N), np.arange(N)]):
        raise ValueError(f"Missing reciprocal blocks or nonzero diagonal: {directory}")
    return *result, hashes


class Moments:
    """Centered second moments via Welford, chunked by leading coordinate."""

    def __init__(self, shape):
        self.count = 0
        self.mean = np.zeros(shape)
        self.m2 = np.zeros(shape)

    def add(self, array):
        self.count += 1
        for i in range(0, len(array), 16):
            block = slice(i, i + 16)
            mean = self.mean[block]
            delta = array[block] - mean
            mean += delta / self.count
            self.m2[block] += delta * (array[block] - mean)

    def merge_balanced_halves(self, other):
        if self.count != other.count or not self.count:
            raise ValueError("Expected two equally sized nonempty halves")
        between = 0.0
        for i in range(0, len(self.mean), 16):
            block = slice(i, i + 16)
            delta = other.mean[block] - self.mean[block]
            term = delta * delta * (self.count / 2.0)
            between += float(term.sum())
            self.m2[block] += other.m2[block] + term
        self.count += other.count
        return between


def estimate(first, second, reference, record, key):
    norm_squared = float(np.einsum("i,i->", reference.reshape(-1), reference.reshape(-1)))
    if norm_squared == 0:
        raise ValueError("Zero reference Frobenius norm")
    r_first = float(np.sqrt(first.m2.sum() / (first.count - 1) / norm_squared))
    r_second = float(np.sqrt(second.m2.sum() / (second.count - 1) / norm_squared))
    between = first.merge_balanced_halves(second)
    total = float(first.m2.sum())
    r = float(np.sqrt(total / (first.count - 1) / norm_squared))
    d = abs(r - r_first) / r if r else 0.0
    for name, measured in ((f"r_{key}", r), (f"d_{key}", d)):
        if abs(measured - record[name]) > 1e-12:
            raise ValueError(f"{name} does not reproduce the archived statistic")
    return {
        "r_all": r, "r_first_50": r_first, "r_last_50": r_second, "d_full_vs_first_50": d,
        "between_half_means_share_of_total_squared_deviation": between / total if total else 0,
        "reference_norm_squared": norm_squared,
        "sample_deviation_norm_squared": total / (first.count - 1),
    }


def parts(coupling_m2, ref, n):
    """Disjoint slices; denominator is the historical both-triangles layout."""
    ref_sq = ref * ref
    total_noise = float(coupling_m2.sum())
    total_reference = float(ref_sq.sum())
    noise_pairs = coupling_m2.sum(axis=(2, 3))
    ref_pairs = ref_sq.sum(axis=(2, 3))
    gap_noise = float(coupling_m2[:, :, 0, :].sum() + coupling_m2[:, :, 1:, 0].sum())
    gap_ref = float(ref_sq[:, :, 0, :].sum() + ref_sq[:, :, 1:, 0].sum())

    def one(noise, reference):
        return {
            "share_of_centered_squared_deviation": noise / total_noise,
            "share_of_reference_squared_norm": reference / total_reference,
            "relative_frobenius": float(np.sqrt(noise / (n - 1) / reference)) if reference else None,
        }

    state = {
        "gap_in_either_axis": one(gap_noise, gap_ref),
        "both_amino_acids": one(total_noise - gap_noise, total_reference - gap_ref),
    }
    idx = np.arange(N)
    distance = np.abs(idx[:, None] - idx[None, :])
    groups = {}
    for name, lower, upper in (("1_to_5", 1, 5), ("6_to_11", 6, 11), ("12_to_23", 12, 23), ("24_and_above", 24, N)):
        selection = (distance >= lower) & (distance <= upper)
        groups[name] = one(float(noise_pairs[selection].sum()), float(ref_pairs[selection].sum()))
    for ratios in (state, groups):
        for key in ("share_of_centered_squared_deviation", "share_of_reference_squared_norm"):
            if not np.isclose(sum(x[key] for x in ratios.values()), 1.0, rtol=0, atol=1e-12):
                raise ArithmeticError(f"Partition failed to account for {key}")
    return {"nonoverlapping_sequence_distance_bins": groups, "state_blocks": state}


class Group:
    def __init__(self, left, right, labels, reference, top):
        self.left, self.right = left, right
        self.k = min(top, len(left))
        self.labels = labels[left, right]
        selected = self.selection(reference)
        self.reference_included = np.zeros(len(left), dtype=bool)
        self.reference_included[selected] = True
        self.always = self.reference_included.copy()
        self.reference_hits = int(self.labels[selected].sum())
        self.rows = []

    def selection(self, score):
        ordered = contacts.ranking_order(self.left, self.right, score[self.left, self.right])
        return ordered[: self.k]

    def add(self, score):
        selected = self.selection(score)
        self.rows.append({
            "overlap_with_reference": int(self.reference_included[selected].sum()),
            "contact_hits": int(self.labels[selected].sum()),
        })
        included = np.zeros_like(self.always)
        included[selected] = True
        self.always &= included

    def summarize(self):
        def distribution(key):
            values = [x[key] for x in self.rows]
            return dict(min=min(values), median=float(np.median(values)), max=max(values),
                        histogram=dict(sorted(Counter(values).items())))
        return {
            "candidates": len(self.left), "top_k": self.k, "reference_hits": self.reference_hits,
            "exact_reference_set_replicates": sum(x["overlap_with_reference"] == self.k for x in self.rows),
            "reference_pairs_in_every_replicate": int(self.always.sum()),
            "overlap_counts": distribution("overlap_with_reference"),
            "contact_hit_counts": distribution("contact_hits"),
            "per_replicate": self.rows,
        }


def groups(root, jstar, certification):
    labels, mismatch = contacts.site_points(root / "prep/domain_map.tsv", root / "prep/1BTL.pdb")
    labels = contacts.contact_matrix(labels)
    score = apc(jstar)
    left, right = contacts.pair_table(~mismatch)
    distance = right - left
    select = {f"separation_at_least_{cut}": distance >= cut for cut in contacts.SEPARATION_CUTS}
    for low, high in contacts.SEPARATION_BINS:
        key = f"separation_{low}_and_above" if high is None else f"separation_{low}_to_{high}"
        select[key] = distance >= low if high is None else (distance >= low) & (distance <= high)
    published = read_json(root / "dcr/contact_validation_summary.json")["main"]
    result = {name: Group(left[mask], right[mask], labels, score, N) for name, mask in select.items()}
    for name, g in result.items():
        if (g.reference_hits / g.k != published[name]["precision_at_l"]
            or len(g.left) != published[name]["pairs"]
            or certification["groups"][name]["numeric_hits"] != g.reference_hits
            or not certification["groups"][name]["certified_numeric_top_set"]):
            raise ValueError(f"Frozen published ranking differs for {name}")
    return result


def apc(params):
    return contacts.average_product_correction(contacts.amino_block_norm(params))


def run(source, certificate_path):
    begin = perf_counter()
    reference = source / "dcr/bootstrap/reference"
    formal = source / "dcr/bootstrap/formal"
    lock = read_json(formal / "seed.json")
    archived_path = formal / "stage_100.json"
    archive = read_json(archived_path)
    if any(archive[key] != lock[key] for key in LOCK_FIELDS) or archive["success_count"] != REPLICATES:
        raise ValueError("Unexpected archive or seed lock")
    cert = read_json(certificate_path)
    rh, rj, hashes = load_arrays(reference)
    ref_manifest = read_json(reference / "manifest.json")
    if (
        not ref_manifest["accepted_point_estimate"]
        or any(hashes[k] != ref_manifest["gauge_sha256"][k] for k in hashes)
        or hashes["j_star.npy"] != cert["references_sha256"]["j_star.npy"]
        or digest(source / "dcr/contact_validation_summary.json") != cert["published_summary_sha256"]
        or cert["certified_groups"] != 5
    ):
        raise ValueError("Saved reference disagrees with the accepted certification")
    g = groups(source, rj, cert)
    h = [Moments(rh.shape), Moments(rh.shape)]
    j = [Moments(rj.shape), Moments(rj.shape)]
    file_hashes = []
    # A direct centered-variance check of independent coordinates, using the
    # same underlying bytes but neither the Welford nor the merge formula.
    sample_coordinates = np.linspace(0, rj.size - 1, 257, dtype=int)
    sample_columns = []

    for rid in range(1, REPLICATES + 1):
        field, coupling, saved_hash = validated_archive(formal / f"replicate_{rid:03d}", rid, lock)
        half = int(rid > 50)
        h[half].add(field)
        j[half].add(coupling)
        sample_columns.append(coupling.reshape(-1)[sample_coordinates])
        score = apc(coupling)
        for group in g.values():
            group.add(score)
        file_hashes.append({"replicate": rid, **saved_hash})
        if rid % 10 == 0:
            print(f"{rid}/{REPLICATES} archived replicates checked", flush=True)

    h_metrics = estimate(*h, rh, archive, "h")
    j_metrics = estimate(*j, rj, archive, "j")
    np.testing.assert_allclose(j[0].m2.reshape(-1)[sample_coordinates] / 99,
                               np.var(np.asarray(sample_columns), axis=0, ddof=1), rtol=1e-12, atol=1e-16)
    return {
        "scope": "Historical archive read only; no fitting, resampling, tree inference or saved-array writes.",
        "reference": {
            "parameter_sha256": hashes, "reference_manifest_sha256": digest(reference / "manifest.json"),
            "certificate_sha256": digest(certificate_path), "published_contact_summary_sha256": cert["published_summary_sha256"],
        },
        "historical": {
            "stage_summary_sha256": digest(archived_path), "archived_decision": archive["action"],
            "replicates": REPLICATES, "parameter_threshold": 0.10, "artifact_sha256": file_hashes,
        },
        "parameter": {"h": h_metrics, "j": j_metrics, "coupling_components": parts(j[0].m2, rj, REPLICATES)},
        "ranking": {
            "scope": "Nominal binary64 Bootstrap scores compared with the certified saved full-data reference. Replicate scores are not interval-certified.",
            "per_replicate_order": "001 through 100",
            "groups": {name: group.summarize() for name, group in g.items()},
        },
        "seconds": perf_counter() - begin,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--certificate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to(args.source.resolve()):
        parser.error("Output must be a new file outside the original source workspace")
    report = run(args.source.resolve(), args.certificate.resolve())
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(f"Wrote {args.output}; wall seconds: {report['seconds']:.2f}")


if __name__ == "__main__":
    main()

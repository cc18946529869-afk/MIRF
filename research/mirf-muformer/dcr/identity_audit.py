"""Check saved Potts parameters against the two energy identities.

This does not refit, resample, or write mutation features. Each replicate uses
its own q.npy. Delta E is taken from potts_energy.mutation_energy.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import contact_validation
import potts_energy

ACCEPTANCE = 1e-12
LENGTH = 263
STATES = 21


def normalized_residual(difference: float, magnitudes: tuple[float, ...]) -> float:
    """Scale a raw difference by one or the sum of absolute magnitudes."""
    return abs(float(difference)) / max(1.0, float(sum(abs(value) for value in magnitudes)))


def geometry(domain_map: Path, pdb_path: Path) -> dict[str, np.ndarray]:
    """Return the sealed lobe and coordinate masks on the mature sites."""
    points, mismatched = contact_validation.site_points(domain_map, pdb_path)
    difference = points[:, None, :] - points[None, :, :]
    distance = np.sqrt(np.sum(difference * difference, axis=2))
    in_coord = ~mismatched
    both = in_coord[:, None] & in_coord[None, :]
    weight = np.zeros((LENGTH, LENGTH), dtype=np.float64)
    weight[both] = 1.0 / (1.0 + np.exp((distance[both] - 8.0) / 1.5))
    lobe = np.full(LENGTH, -1, dtype=int)
    for line in domain_map.read_text().splitlines()[1:]:
        if not line:
            continue
        parts = line.split("\t")
        if parts[6] != "1":
            continue
        lobe[int(parts[1]) - 1] = int(parts[4])
    if np.any(lobe < 0):
        raise ValueError("a mature site has no lobe")
    same = lobe[:, None] == lobe[None, :]
    eye = np.eye(LENGTH, dtype=bool)
    other = ~same & ~eye
    return {
        "source": same & ~eye,
        "other": other,
        "other_known": other & both,
        "other_unknown": other & ~both,
        "weight": weight,
    }


def audit_parameters(
    sequence: np.ndarray,
    fields: np.ndarray,
    couplings: np.ndarray,
    background: np.ndarray,
    masks: dict[str, np.ndarray],
) -> dict[str, float]:
    """Return the largest raw and normalized residuals over all target states."""
    sequence = np.asarray(sequence, dtype=np.int64)
    fields = np.asarray(fields, dtype=np.float64)
    couplings = np.asarray(couplings, dtype=np.float64)
    background = np.asarray(background, dtype=np.float64)
    if background.shape != (sequence.shape[0], STATES):
        raise ValueError("background q must have one row per site and 21 states")
    if not np.isfinite(background).all() or np.any(background < 0):
        raise ValueError("background q must be finite and nonnegative")
    index = np.arange(sequence.shape[0])
    old = couplings[index, :, sequence, :]
    largest = {
        "raw_energy": 0.0,
        "raw_geometry": 0.0,
        "normalized_energy": 0.0,
        "normalized_geometry": 0.0,
    }
    for state in range(STATES):
        delta = couplings[index, :, state, :] - old
        mean = np.einsum("ijb,jb->ij", delta, background)
        context = delta[index[:, None], index[None, :], sequence[None, :]] - mean
        np.fill_diagonal(mean, 0.0)
        np.fill_diagonal(context, 0.0)
        delta_h = fields[:, state] - fields[index, sequence]
        r_mean = -(delta_h + mean.sum(axis=1))
        r_source = -(context * masks["source"]).sum(axis=1)
        r_other = -(context * masks["other"]).sum(axis=1)
        r_contact = -(context * masks["weight"] * masks["other_known"]).sum(axis=1)
        r_remote = -(context * (1.0 - masks["weight"]) * masks["other_known"]).sum(axis=1)
        r_unknown = -(context * masks["other_unknown"]).sum(axis=1)
        geometry_sum = r_contact + r_remote + r_unknown
        for site in range(sequence.shape[0]):
            delta_e = potts_energy.mutation_energy(sequence, site, state, fields, couplings)
            raw_energy = abs(delta_e - (r_mean[site] + r_source[site] + r_other[site]))
            raw_geometry = abs(r_other[site] - geometry_sum[site])
            normalized_energy = normalized_residual(
                raw_energy,
                (delta_e, r_mean[site], r_source[site], r_other[site]),
            )
            normalized_geometry = normalized_residual(
                raw_geometry,
                (r_other[site], r_contact[site], r_remote[site], r_unknown[site]),
            )
            largest["raw_energy"] = max(largest["raw_energy"], raw_energy)
            largest["raw_geometry"] = max(largest["raw_geometry"], raw_geometry)
            largest["normalized_energy"] = max(largest["normalized_energy"], normalized_energy)
            largest["normalized_geometry"] = max(largest["normalized_geometry"], normalized_geometry)
    largest["q_abs"] = float(np.max(np.abs(background.sum(axis=1) - 1.0)))
    return largest


def _require_replicate(directory: Path, replicate_id: int, lock: dict) -> None:
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("replicate_success") is not True or manifest.get("passed_sites") != LENGTH:
        raise ValueError(f"replicate {replicate_id} is not a successful formal fit")
    if int(manifest.get("formal_replicate_id")) != replicate_id:
        raise ValueError(f"replicate {replicate_id} has a different manifest id")
    for key in ("code_revision", "alignment_sha256", "neighbor_counts_sha256"):
        if manifest.get(key) != lock[key]:
            raise ValueError(f"replicate {replicate_id} does not match seed.json {key}")


def run(root: Path) -> dict:
    """Audit formal replicates 1 through 100 and return the maxima."""
    dcr = root / "dcr"
    prep = root / "prep"
    lock = json.loads((dcr / "bootstrap" / "formal" / "seed.json").read_text())
    sequence = np.load(prep / "mature_codes.npy")[0]
    masks = geometry(prep / "domain_map.tsv", prep / "1BTL.pdb")
    rows = []
    for replicate_id in range(1, 101):
        directory = dcr / "bootstrap" / "formal" / f"replicate_{replicate_id:03d}"
        _require_replicate(directory, replicate_id, lock)
        summary = audit_parameters(
            sequence,
            np.load(directory / "h_hat.npy"),
            np.load(directory / "j_star.npy"),
            np.load(directory / "q.npy"),
            masks,
        )
        summary["replicate"] = replicate_id
        rows.append(summary)
        print(
            f"{replicate_id} {summary['normalized_energy']:.3e} {summary['normalized_geometry']:.3e}",
            flush=True,
        )
    keys = ("raw_energy", "raw_geometry", "normalized_energy", "normalized_geometry", "q_abs")
    maxima = {key: max(row[key] for row in rows) for key in keys}
    passed = (
        maxima["normalized_energy"] <= ACCEPTANCE
        and maxima["normalized_geometry"] <= ACCEPTANCE
        and maxima["q_abs"] <= ACCEPTANCE
    )
    return {
        "acceptance": ACCEPTANCE,
        "replicates": 100,
        "sites": LENGTH,
        "states": STATES,
        "comparisons": 100 * LENGTH * STATES,
        "passed": passed,
        "maxima": maxima,
        "rows": rows,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    summary = run(root)
    destination = Path(__file__).resolve().parent / "bootstrap_identity_audit.json"
    destination.write_text(json.dumps(summary, indent=2) + "\n")
    print(destination)
    print(json.dumps({"passed": summary["passed"], "maxima": summary["maxima"]}))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""Rank gauged couplings against 1BTL contacts.

The score is not a DCR feature. It asks whether large 20-state blocks of the
full-fit J* lie near one another in the crystal. Gap state 0 is left out of
the block norm. Sequence-separation cuts are fixed before the scores are read.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

AMINO_COUNT = 20
LENGTH = 263
CONTACT_ANGSTROM = 8.0
SEPARATION_CUTS = (6, 12)
SEPARATION_BINS = ((6, 11), (12, 23), (24, None))


def amino_block_norm(couplings: np.ndarray) -> np.ndarray:
    """Return the Frobenius norm of each 20 by 20 amino-acid block."""
    block = np.asarray(couplings, dtype=np.float64)[:, :, 1:, 1:]
    if block.shape[-2:] != (AMINO_COUNT, AMINO_COUNT):
        raise ValueError("the amino-acid block must exclude one gap state")
    squared = np.sum(block * block, axis=(2, 3))
    return np.sqrt(squared)


def average_product_correction(scores: np.ndarray) -> np.ndarray:
    """Correct a symmetric pair-score matrix with a zero diagonal."""
    values = np.asarray(scores, dtype=np.float64).copy()
    if values.shape[0] != values.shape[1]:
        raise ValueError("pair scores must be square")
    np.fill_diagonal(values, 0.0)
    count = values.shape[0] - 1
    row_mean = values.sum(axis=1) / float(count)
    center = float(row_mean.mean())
    if center == 0.0:
        raise ValueError("average-product correction needs a nonzero background")
    corrected = values - np.outer(row_mean, row_mean) / center
    np.fill_diagonal(corrected, 0.0)
    return corrected


def ranking_metrics(contact: np.ndarray, score: np.ndarray, length: int) -> dict:
    """Score pairs already restricted to one comparison set."""
    order = np.argsort(score, kind="mergesort")[::-1]
    ranked = np.asarray(contact, dtype=bool)[order]
    total = int(ranked.size)
    hits = int(ranked.sum())
    base_rate = float(hits / total) if total else None

    def precision(count: int) -> float | None:
        if total == 0 or int(count) <= 0:
            return None
        used = min(int(count), total)
        return float(ranked[:used].mean())

    if hits == 0:
        average_precision = 0.0
    else:
        cumulative = np.cumsum(ranked)
        found = np.flatnonzero(ranked)
        average_precision = float(np.mean(cumulative[found] / (found + 1)))
    return {
        "pairs": total,
        "contacts": hits,
        "contact_rate": base_rate,
        "precision_at_l": precision(length),
        "precision_at_l_half": precision(length // 2),
        "precision_at_l_fifth": precision(length // 5),
        "average_precision": average_precision,
    }


def pair_table(included: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return upper-triangle index pairs that pass a symmetric site mask."""
    left, right = np.triu_indices(included.shape[0], k=1)
    keep = included[left] & included[right]
    return left[keep], right[keep]


def evaluate_sets(
    contact: np.ndarray,
    score: np.ndarray,
    included: np.ndarray,
    length: int,
) -> dict:
    """Evaluate the fixed separation sets without choosing among them."""
    left, right = pair_table(included)
    separation = right - left
    outcome = {}
    for cutoff in SEPARATION_CUTS:
        chosen = separation >= cutoff
        outcome[f"separation_at_least_{cutoff}"] = ranking_metrics(
            contact[left[chosen], right[chosen]],
            score[left[chosen], right[chosen]],
            length,
        )
    for low, high in SEPARATION_BINS:
        chosen = separation >= low if high is None else (separation >= low) & (separation <= high)
        label = f"separation_{low}_and_above" if high is None else f"separation_{low}_to_{high}"
        outcome[label] = ranking_metrics(
            contact[left[chosen], right[chosen]],
            score[left[chosen], right[chosen]],
            length,
        )
    return outcome


def coordinates_from_pdb(path: Path) -> dict[int, dict[str, np.ndarray]]:
    """Read the first Chain A heavy-atom coordinate for CA and CB."""
    found: dict[int, dict[str, np.ndarray]] = {}
    for line in path.read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        if line[21] != "A":
            continue
        altloc = line[16]
        if altloc not in (" ", "A"):
            continue
        atom = line[12:16].strip()
        if atom not in ("CA", "CB"):
            continue
        resid = int(line[22:26])
        point = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])], dtype=np.float64)
        found.setdefault(resid, {})
        found[resid].setdefault(atom, point)
    return found


def site_points(domain_map: Path, pdb_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return usable coordinates, the two mismatched sites, and whether each site is glycine."""
    atoms = coordinates_from_pdb(pdb_path)
    points = np.full((LENGTH, 3), np.nan, dtype=np.float64)
    mismatched = np.zeros(LENGTH, dtype=bool)
    glycine = np.zeros(LENGTH, dtype=bool)
    for line in domain_map.read_text().splitlines()[1:]:
        ref_pos, mature_pos, _chain, auth_resid, _lobe_id, _lobe_name, mapped, wt_aa, pdb_aa, has_ca, aa_match = (
            line.split("\t")
        )
        if mapped != "1":
            continue
        index = int(mature_pos) - 1
        glycine[index] = wt_aa == "G"
        mismatched[index] = aa_match == "0"
        resid = int(auth_resid)
        atom_name = "CA" if wt_aa == "G" else "CB"
        if resid not in atoms or atom_name not in atoms[resid]:
            raise ValueError(f"reference position {ref_pos} has no {atom_name}")
        if has_ca != "1":
            raise ValueError(f"reference position {ref_pos} was expected to have CA")
        points[index] = atoms[resid][atom_name]
    if int(mismatched.sum()) != 2:
        raise ValueError("the domain map should mark exactly two amino-acid mismatches")
    return points, mismatched, glycine


def contact_matrix(points: np.ndarray) -> np.ndarray:
    """Mark pairs whose selected atoms are strictly closer than 8 angstroms."""
    difference = points[:, None, :] - points[None, :, :]
    distance = np.sqrt(np.sum(difference * difference, axis=2))
    contact = distance < CONTACT_ANGSTROM
    np.fill_diagonal(contact, False)
    return contact


def run(root: Path) -> dict:
    """Compare the saved full-fit J* with 1BTL and return the fixed summaries."""
    couplings = np.load(root / "dcr" / "bootstrap" / "reference" / "j_star.npy")
    raw_scores = amino_block_norm(couplings)
    corrected = average_product_correction(raw_scores)
    points, mismatched, _glycine = site_points(root / "prep" / "domain_map.tsv", root / "prep" / "1BTL.pdb")
    contact = contact_matrix(points)
    included = ~mismatched
    return {
        "length": LENGTH,
        "contact_rule": "selected atoms closer than 8 angstroms",
        "atom_rule": "glycine CA, every other residue CB",
        "score": "average-product corrected Frobenius norm of the 20 amino-acid states",
        "excluded_reference_positions": [82, 182],
        "main": evaluate_sets(contact, corrected, included, LENGTH),
        "sensitivity_with_mismatched_coordinates": evaluate_sets(
            contact,
            corrected,
            np.ones(LENGTH, dtype=bool),
            LENGTH,
        ),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    summary = run(root)
    destination = Path(__file__).resolve().parent / "contact_validation_summary.json"
    destination.write_text(json.dumps(summary, indent=2) + "\n")
    print(destination)


if __name__ == "__main__":
    main()

"""Rank gauged couplings against 1BTL contacts.

The score is not a DCR feature. It asks whether large 20-state blocks of the
full-fit J* lie near one another in the crystal. Gap state 0 is left out of
the block norm. Sequence-separation cuts are fixed before the scores are read.

Average-product correction uses all 263 mature sites. Reference positions 82
and 182 stay in that background. They leave only the main contact ranking.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

AMINO_COUNT = 20
LENGTH = 263
CONTACT_ANGSTROM = 8.0
SEPARATION_CUTS = (6, 12)
SEPARATION_BINS = ((6, 11), (12, 23), (24, None))
MISMATCH_REFERENCE_POSITIONS = (82, 182)
EXPECTED_MISMATCHES = (
    {"ref_pos": 82, "mature_pos": 59, "pdb_auth_resid": 84, "wt_aa": "V", "pdb_aa": "I"},
    {"ref_pos": 182, "mature_pos": 159, "pdb_auth_resid": 184, "wt_aa": "A", "pdb_aa": "V"},
)
TIE_BREAK = "higher score, then smaller left index, then smaller right index"
APC_SCOPE = (
    "row means use all 263 mature sites, including reference positions 82 and 182; "
    "those two positions are removed only from the main contact ranking"
)


def amino_block_norm(couplings: np.ndarray) -> np.ndarray:
    """Return the Frobenius norm of each 20 by 20 amino-acid block."""
    block = np.asarray(couplings, dtype=np.float64)[:, :, 1:, 1:]
    if block.shape[-2:] != (AMINO_COUNT, AMINO_COUNT):
        raise ValueError("the amino-acid block must exclude one gap state")
    squared = np.sum(block * block, axis=(2, 3))
    return np.sqrt(squared)


def average_product_correction(scores: np.ndarray) -> np.ndarray:
    """Correct a symmetric pair-score matrix with a zero diagonal.

    Every site in the matrix enters the row means. Callers that later drop
    sites from a contact ranking do not remove those sites from this step.
    """
    values = np.asarray(scores, dtype=np.float64).copy()
    if values.shape[0] != values.shape[1]:
        raise ValueError("pair scores must be square")
    if not np.isfinite(values).all():
        raise ValueError("pair scores must be finite")
    np.fill_diagonal(values, 0.0)
    count = values.shape[0] - 1
    if count <= 0:
        raise ValueError("average-product correction needs at least two sites")
    row_mean = values.sum(axis=1) / float(count)
    center = float(row_mean.mean())
    if center == 0.0:
        raise ValueError("average-product correction needs a nonzero background")
    corrected = values - np.outer(row_mean, row_mean) / center
    np.fill_diagonal(corrected, 0.0)
    return corrected


def ranking_order(left: np.ndarray, right: np.ndarray, score: np.ndarray) -> np.ndarray:
    """Order pairs by descending score, then by ascending site indices."""
    left = np.asarray(left)
    right = np.asarray(right)
    score = np.asarray(score, dtype=np.float64)
    if not np.isfinite(score).all():
        raise ValueError("ranking scores must be finite")
    return np.lexsort((right, left, -score))


def _prefix_precision(contact: np.ndarray, chosen: np.ndarray) -> float | None:
    if chosen.size == 0:
        return None
    return float(np.mean(contact[chosen]))


def _boundary(contact: np.ndarray, score: np.ndarray, order: np.ndarray, count: int) -> dict:
    total = int(order.size)
    requested = int(count)
    if total == 0 or requested <= 0:
        return {"requested": requested, "used": 0, "boundary_tie": False, "gap": None}
    used = min(requested, total)
    precision = _prefix_precision(contact, order[:used])
    if used >= total:
        return {
            "requested": requested,
            "used": used,
            "boundary_tie": False,
            "gap": None,
            "precision": precision,
            "precision_if_tied_group_excluded": None,
            "precision_if_tied_group_included": None,
        }
    last = float(score[order[used - 1]])
    nxt = float(score[order[used]])
    tied = last == nxt
    report = {
        "requested": requested,
        "used": used,
        "boundary_tie": tied,
        "gap": last - nxt,
        "precision": precision,
        "precision_if_tied_group_excluded": None,
        "precision_if_tied_group_included": None,
    }
    if tied:
        above = np.flatnonzero(score > last)
        at_or_above = np.flatnonzero(score >= last)
        report["precision_if_tied_group_excluded"] = _prefix_precision(contact, above)
        report["precision_if_tied_group_included"] = _prefix_precision(contact, at_or_above)
        report["tied_group_size"] = int(np.count_nonzero(score == last))
    return report


def ranking_metrics(
    left: np.ndarray,
    right: np.ndarray,
    contact: np.ndarray,
    score: np.ndarray,
    length: int,
) -> dict:
    """Score one comparison set with an explicit tie break."""
    contact = np.asarray(contact, dtype=bool)
    score = np.asarray(score, dtype=np.float64)
    order = ranking_order(left, right, score)
    ranked = contact[order]
    total = int(ranked.size)
    hits = int(ranked.sum())
    base_rate = float(hits / total) if total else None
    unique, counts = np.unique(score, return_counts=True) if total else (np.array([]), np.array([]))
    del unique
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
        "precision_at_l": _prefix_precision(contact, order[: min(length, total)]) if total else None,
        "precision_at_l_half": _prefix_precision(contact, order[: min(length // 2, total)]) if total else None,
        "precision_at_l_fifth": _prefix_precision(contact, order[: min(length // 5, total)]) if total else None,
        "average_precision": average_precision,
        "tie_break": TIE_BREAK,
        "duplicate_score_values": int(np.count_nonzero(counts > 1)),
        "pairs_sharing_a_score": int(counts[counts > 1].sum()) if counts.size else 0,
        "boundaries": {
            "l": _boundary(contact, score, order, length),
            "l_half": _boundary(contact, score, order, length // 2),
            "l_fifth": _boundary(contact, score, order, length // 5),
        },
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
    """Evaluate the fixed separation sets without choosing among them.

    ``score`` is used as given. This function does not recompute the
    average-product correction, so a site can remain in that correction and
    still be absent from the ranking.
    """
    left, right = pair_table(included)
    separation = right - left
    outcome = {}
    for cutoff in SEPARATION_CUTS:
        chosen = separation >= cutoff
        outcome[f"separation_at_least_{cutoff}"] = ranking_metrics(
            left[chosen],
            right[chosen],
            contact[left[chosen], right[chosen]],
            score[left[chosen], right[chosen]],
            length,
        )
    for low, high in SEPARATION_BINS:
        chosen = separation >= low if high is None else (separation >= low) & (separation <= high)
        label = f"separation_{low}_and_above" if high is None else f"separation_{low}_to_{high}"
        outcome[label] = ranking_metrics(
            left[chosen],
            right[chosen],
            contact[left[chosen], right[chosen]],
            score[left[chosen], right[chosen]],
            length,
        )
    return outcome


def coordinates_from_pdb(path: Path) -> dict[int, dict[str, np.ndarray]]:
    """Read Chain A CA and CB coordinates. A nonblank insertion code is refused."""
    found: dict[int, dict[str, np.ndarray]] = {}
    for line in path.read_text().splitlines():
        if not line.startswith("ATOM") or len(line) < 54:
            continue
        if line[21] != "A":
            continue
        altloc = line[16]
        if altloc not in (" ", "A"):
            continue
        atom = line[12:16].strip()
        if atom not in ("CA", "CB"):
            continue
        if line[26] != " ":
            raise ValueError(f"PDB residue {line[22:26]} has an insertion code")
        resid = int(line[22:26])
        point = np.array(
            [float(line[30:38]), float(line[38:46]), float(line[46:54])],
            dtype=np.float64,
        )
        if not np.isfinite(point).all():
            raise ValueError(f"PDB residue {resid} has a non-finite {atom} coordinate")
        found.setdefault(resid, {})
        found[resid].setdefault(atom, point)
    return found


def read_mapped_sites(domain_map: Path) -> list[dict]:
    """Return the mapped domain-map rows."""
    rows = []
    for line in domain_map.read_text().splitlines()[1:]:
        if not line:
            continue
        ref_pos, mature_pos, chain, auth_resid, _lobe_id, _lobe_name, mapped, wt_aa, pdb_aa, has_ca, aa_match = (
            line.split("\t")
        )
        if mapped != "1":
            continue
        rows.append(
            {
                "ref_pos": int(ref_pos),
                "mature_pos": int(mature_pos),
                "chain": chain,
                "pdb_auth_resid": int(auth_resid),
                "wt_aa": wt_aa,
                "pdb_aa": pdb_aa,
                "has_ca": has_ca,
                "aa_match": aa_match,
            }
        )
    return rows


def require_site_mapping(rows: list[dict], length: int, expected_mismatches: tuple[dict, ...]) -> None:
    """Require one mature site per index, finite identity fields, and the known mismatches."""
    if len(rows) != length:
        raise ValueError(f"expected {length} mapped sites, found {len(rows)}")
    mature = sorted(row["mature_pos"] for row in rows)
    if mature != list(range(1, length + 1)):
        raise ValueError("mapped mature positions are not exactly 1 through L")
    if any(row["chain"] != "A" or row["has_ca"] != "1" for row in rows):
        raise ValueError("every mapped site must be Chain A and have CA")
    auth = [row["pdb_auth_resid"] for row in rows]
    if len(set(auth)) != len(auth):
        raise ValueError("mapped PDB residue numbers are not unique")
    found = [
        {
            "ref_pos": row["ref_pos"],
            "mature_pos": row["mature_pos"],
            "pdb_auth_resid": row["pdb_auth_resid"],
            "wt_aa": row["wt_aa"],
            "pdb_aa": row["pdb_aa"],
        }
        for row in rows
        if row["aa_match"] != "1"
    ]
    if found != [dict(item) for item in expected_mismatches]:
        raise ValueError("amino-acid mismatches are not the two known reference positions")


def site_points(
    domain_map: Path,
    pdb_path: Path,
    length: int = LENGTH,
    expected_mismatches: tuple[dict, ...] = EXPECTED_MISMATCHES,
) -> tuple[np.ndarray, np.ndarray]:
    """Return one finite coordinate per mature site and the mismatch mask.

    Glycine uses CA. Every other residue uses CB. A missing atom raises.
    """
    rows = read_mapped_sites(domain_map)
    require_site_mapping(rows, length, expected_mismatches)
    atoms = coordinates_from_pdb(pdb_path)
    points = np.full((length, 3), np.nan, dtype=np.float64)
    mismatched = np.zeros(length, dtype=bool)
    for row in rows:
        index = row["mature_pos"] - 1
        atom_name = "CA" if row["wt_aa"] == "G" else "CB"
        residue = atoms.get(row["pdb_auth_resid"])
        if residue is None or atom_name not in residue:
            raise ValueError(f"reference position {row['ref_pos']} has no {atom_name}")
        points[index] = residue[atom_name]
        mismatched[index] = row["aa_match"] != "1"
    if not np.isfinite(points).all():
        raise ValueError("a mapped site has no finite coordinate")
    return points, mismatched


def contact_matrix(points: np.ndarray) -> np.ndarray:
    """Mark pairs whose selected atoms are strictly closer than 8 angstroms."""
    coordinates = np.asarray(points, dtype=np.float64)
    if not np.isfinite(coordinates).all():
        raise ValueError("contact distances require finite coordinates")
    difference = coordinates[:, None, :] - coordinates[None, :, :]
    distance = np.sqrt(np.sum(difference * difference, axis=2))
    contact = distance < CONTACT_ANGSTROM
    np.fill_diagonal(contact, False)
    return contact


def _published_metrics(summary: dict) -> dict:
    kept = {}
    for group in ("main", "sensitivity_with_mismatched_coordinates"):
        kept[group] = {}
        for name, metrics in summary[group].items():
            kept[group][name] = {
                key: metrics[key]
                for key in (
                    "pairs",
                    "contacts",
                    "contact_rate",
                    "precision_at_l",
                    "precision_at_l_half",
                    "precision_at_l_fifth",
                    "average_precision",
                )
            }
    return kept


def run(root: Path) -> dict:
    """Compare the saved full-fit J* with 1BTL and return the audit."""
    points, mismatched = site_points(root / "prep" / "domain_map.tsv", root / "prep" / "1BTL.pdb")
    couplings = np.load(root / "dcr" / "bootstrap" / "reference" / "j_star.npy")
    corrected = average_product_correction(amino_block_norm(couplings))
    contact = contact_matrix(points)
    published_path = Path(__file__).resolve().parent / "contact_validation_summary.json"
    published = json.loads(published_path.read_text())
    current = {
        "main": evaluate_sets(contact, corrected, ~mismatched, LENGTH),
        "sensitivity_with_mismatched_coordinates": evaluate_sets(
            contact,
            corrected,
            np.ones(LENGTH, dtype=bool),
            LENGTH,
        ),
    }
    digest = hashlib.sha256(published_path.read_bytes()).hexdigest()
    return {
        "length": LENGTH,
        "contact_rule": "selected atoms strictly closer than 8 angstroms",
        "atom_rule": "glycine CA, every other residue CB",
        "score": "average-product corrected Frobenius norm of the 20 amino-acid states",
        "apc_scope": APC_SCOPE,
        "tie_break": TIE_BREAK,
        "mapping": {
            "mapped_sites": LENGTH,
            "finite_coordinates": int(np.isfinite(points).all(axis=1).sum()),
            "amino_acid_mismatches": [dict(item) for item in EXPECTED_MISMATCHES],
        },
        "published_summary_sha256": digest,
        "published_precision_matches_tie_break": _published_metrics(published) == _published_metrics(current),
        "main": current["main"],
        "sensitivity_with_mismatched_coordinates": current["sensitivity_with_mismatched_coordinates"],
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    summary = run(root)
    destination = Path(__file__).resolve().parent / "contact_validation_audit.json"
    destination.write_text(json.dumps(summary, indent=2) + "\n")
    print(destination)


if __name__ == "__main__":
    main()

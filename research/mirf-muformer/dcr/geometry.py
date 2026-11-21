"""Soft-weight geometry for the residual, with positions 82 and 182 kept.

The contact ranking still uses ``contact_validation`` and its mismatch mask.
This module does not call that ranking and does not set
``in_coord = ~mismatched``. A site with a finite 1BTL atom stays in the
coordinate set, including the two sequence mismatches. Both of those residues
use the Cβ already stored for chain A.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import contact_validation
import lobes

LENGTH = contact_validation.LENGTH


def _pairwise_masks(points: np.ndarray, coordinate: np.ndarray, lobe: np.ndarray) -> dict[str, np.ndarray]:
    difference = points[:, None, :] - points[None, :, :]
    distance = np.sqrt(np.sum(difference * difference, axis=2))
    both = coordinate[:, None] & coordinate[None, :]
    weight = np.zeros((LENGTH, LENGTH), dtype=np.float64)
    weight[both] = 1.0 / (1.0 + np.exp((distance[both] - 8.0) / 1.5))
    same = lobe[:, None] == lobe[None, :]
    eye = np.eye(LENGTH, dtype=bool)
    other = ~same & ~eye
    return {
        "source": same & ~eye,
        "other": other,
        "other_known": other & both,
        "other_unknown": other & ~both,
        "weight": weight,
        "coordinate": coordinate.copy(),
        "distance": distance,
    }


def residual_geometry(domain_map: Path, pdb_path: Path) -> dict:
    """Return component masks whose coordinate set keeps 82 and 182.

    The returned masks use the same keys as ``identity_audit.geometry`` so
    ``component_state`` can read them. ``record`` states the wild-type and
    PDB amino acids at the two mismatched reference positions.
    """
    points, mismatched = contact_validation.site_points(domain_map, pdb_path)
    coordinate = np.isfinite(points).all(axis=1)
    if coordinate.shape != (LENGTH,) or not bool(coordinate.all()):
        raise ValueError("a mature site has no finite 1BTL coordinate")
    if np.array_equal(coordinate, ~mismatched):
        raise ValueError("the soft-weight coordinate set collapsed to the mismatch complement")
    lobe = lobes.lobe_ids(domain_map)
    masks = _pairwise_masks(points, coordinate, lobe)
    expected = []
    for item in contact_validation.EXPECTED_MISMATCHES:
        index = int(item["mature_pos"]) - 1
        if not bool(mismatched[index]) or not bool(coordinate[index]):
            raise ValueError(f"reference position {item['ref_pos']} is not both mismatched and coordinated")
        if item["wt_aa"] == "G":
            raise ValueError("the sealed mismatches are not glycine, so the atom is Cβ")
        expected.append(
            {
                "ref_pos": int(item["ref_pos"]),
                "mature_pos": int(item["mature_pos"]),
                "wt_aa": item["wt_aa"],
                "pdb_aa": item["pdb_aa"],
                "atom": "CB",
                "in_soft_weight_coordinate_set": True,
            }
        )
    masks["record"] = {
        "mismatches": expected,
        "used_in_coord_equals_not_mismatched": False,
        "contact_ranking_unchanged": True,
    }
    return masks

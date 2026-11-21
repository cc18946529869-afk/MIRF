"""Read the sealed TEM-1 lobe table.

Signal-peptide positions stay out of Potts. Token index equals the 1-based
reference position because ``<s>`` occupies index 0.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

MATURE_START = 24
MATURE_END = 286
HELICAL_LOBE = 1
OTHER_LOBE = 0


def read_rows(domain_map: Path) -> list[dict[str, str]]:
    lines = Path(domain_map).read_text().splitlines()
    header = lines[0].split("\t")
    rows = []
    for line in lines[1:]:
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) != len(header):
            raise ValueError("domain_map.tsv has a ragged row")
        rows.append(dict(zip(header, parts)))
    return rows


def mature_rows(domain_map: Path) -> list[dict[str, str]]:
    """Mapped mature sites, in mature-position order."""
    selected = [row for row in read_rows(domain_map) if row["mapped"] == "1"]
    selected.sort(key=lambda row: int(row["mature_pos"]))
    if len(selected) != 263:
        raise ValueError("the mature map does not contain 263 sites")
    positions = [int(row["ref_pos"]) for row in selected]
    if positions[0] != MATURE_START or positions[-1] != MATURE_END:
        raise ValueError("mature reference positions do not run from 24 to 286")
    if positions != list(range(MATURE_START, MATURE_END + 1)):
        raise ValueError("mature reference positions are not contiguous")
    return selected


def lobe_ids(domain_map: Path) -> np.ndarray:
    """Lobe id of each mature site. Helical is 1, the other lobe is 0."""
    rows = mature_rows(domain_map)
    lobes = np.array([int(row["lobe_id"]) for row in rows], dtype=int)
    if set(lobes.tolist()) != {OTHER_LOBE, HELICAL_LOBE}:
        raise ValueError("lobe ids are not the two sealed values")
    return lobes


def token_indices_for_lobe(domain_map: Path, lobe_id: int) -> np.ndarray:
    """1-based reference positions of one lobe. These are also token indices."""
    rows = mature_rows(domain_map)
    indices = [int(row["ref_pos"]) for row in rows if int(row["lobe_id"]) == lobe_id]
    if not indices:
        raise ValueError(f"lobe {lobe_id} is empty")
    return np.array(indices, dtype=np.int64)


def source_and_other_tokens(domain_map: Path, ref_pos: int) -> tuple[np.ndarray, np.ndarray]:
    """Token indices of the mutation's lobe and of the other lobe.

    The mutation site stays in the source average. Signal peptide, ``<s>`` and
    the final end token are absent from both lists.
    """
    rows = mature_rows(domain_map)
    matched = [row for row in rows if int(row["ref_pos"]) == int(ref_pos)]
    if len(matched) != 1:
        raise ValueError(f"reference position {ref_pos} is not one mature site")
    source_id = int(matched[0]["lobe_id"])
    other_id = HELICAL_LOBE if source_id == OTHER_LOBE else OTHER_LOBE
    source = token_indices_for_lobe(domain_map, source_id)
    other = token_indices_for_lobe(domain_map, other_id)
    if int(ref_pos) not in set(source.tolist()):
        raise ValueError("the mutated site is missing from its lobe average")
    if int(ref_pos) in set(other.tolist()):
        raise ValueError("the mutated site appears in the other lobe")
    if source.min() < MATURE_START or other.min() < MATURE_START:
        raise ValueError("a lobe average includes the signal peptide or <s>")
    return source, other

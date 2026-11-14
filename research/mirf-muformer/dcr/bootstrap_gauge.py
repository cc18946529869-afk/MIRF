"""Assemble directional Potts parameters and apply the frozen gauge map.

The field recovery uses couplings before the gauge map. The averaged couplings
are not substituted back into that recovery.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import gauge


def partner_index(site: int, partner: int) -> int:
    """Return the packed partner slot for one ordered site pair.

    Packed couplings store the other columns in increasing column index.
    """
    if partner == site:
        raise ValueError("a site has no self coupling")
    if partner < site:
        return partner
    return partner - 1


def assemble_directional(
    site_fields: list[np.ndarray],
    site_couplings: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Stack per-site parameters into fields and a full directional tensor.

    Each field is ``(q,)``. Each coupling block is ``(q, length - 1, q)`` in
    predicted-state, packed-partner, partner-state order. The returned
    directional tensor is ``(length, length, q, q)`` with zero diagonal blocks.
    """
    length = len(site_fields)
    if len(site_couplings) != length or length == 0:
        raise ValueError("every site needs one field and one coupling block")
    q = int(site_fields[0].shape[0])
    fields = np.zeros((length, q), dtype=np.float64)
    directional = np.zeros((length, length, q, q), dtype=np.float64)
    for site, (field, couplings) in enumerate(zip(site_fields, site_couplings)):
        field = np.asarray(field, dtype=np.float64)
        couplings = np.asarray(couplings, dtype=np.float64)
        if field.shape != (q,) or couplings.shape != (q, length - 1, q):
            raise ValueError(f"site {site} has an unexpected parameter shape")
        fields[site] = field
        for partner in range(length):
            if partner == site:
                continue
            directional[site, partner] = couplings[:, partner_index(site, partner), :]
    return fields, directional


def gauge_parameters(fields: np.ndarray, directional: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Recover the field, then average the Ising-transformed couplings."""
    recovered = gauge.recover_fields(fields, directional)
    averaged = gauge.average_couplings(directional)
    return recovered, averaged


def load_directional(directory: Path, site_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Load ``site_XXX.npz`` files and assemble their directional parameters."""
    fields = []
    couplings = []
    for site in range(site_count):
        saved = np.load(directory / f"site_{site:03d}.npz")
        fields.append(saved["h"])
        couplings.append(saved["couplings"])
    return assemble_directional(fields, couplings)

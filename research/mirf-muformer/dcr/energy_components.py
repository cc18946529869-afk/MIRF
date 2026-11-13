"""Energy-component check for an already gauge-mapped Potts pair.

The split is the frozen one: a mutation's ΔE equals the background term plus
the sum of the group terms. This module does not change the gauge map.
"""

from __future__ import annotations

import numpy as np

import gauge


def packed_block(couplings: np.ndarray, site: int, partner: int) -> np.ndarray:
    """Return one site's (predicted state, partner state) block."""
    if partner == site:
        raise ValueError("a site has no coupling block with itself")
    index = partner if partner < site else partner - 1
    return np.asarray(couplings[:, index, :], dtype=np.float64)


def paired_ising_block(block_from_site: np.ndarray, block_from_partner: np.ndarray) -> np.ndarray:
    """Average the two Ising images of one unordered pair.

    ``block_from_partner`` is indexed by the partner's own state first.
    """
    from_site = gauge.ising_couplings(block_from_site)
    from_partner = gauge.ising_couplings(block_from_partner).T
    return 0.5 * (from_site + from_partner)


def recovered_site_field(fields: np.ndarray, couplings: np.ndarray) -> np.ndarray:
    """Recover one site's field from its own pre-transform couplings."""
    directional_mean = np.asarray(couplings, dtype=np.float64).mean(axis=2).sum(axis=1)
    return np.asarray(fields, dtype=np.float64) + directional_mean


def mutation_components(
    sequence: np.ndarray,
    site: int,
    new_state: int,
    fields: np.ndarray,
    couplings: np.ndarray,
    background: np.ndarray,
    groups: list[np.ndarray],
) -> tuple[float, tuple[float, ...]]:
    """Return the background term and one term per group.

    ``fields`` is (L, q) and ``couplings`` is (L, L, q, q), both after the
    five-step map. The groups partition the sites that enter the sum.
    """
    sequence = np.asarray(sequence)
    site = int(site)
    old_state = int(sequence[site])
    new_state = int(new_state)
    delta_h = float(fields[site, new_state] - fields[site, old_state])
    length = int(sequence.shape[0])
    directed = np.zeros(length, dtype=np.float64)
    mean_sum = 0.0
    for partner in range(length):
        if partner == site:
            continue
        delta_j = couplings[site, partner, new_state, :] - couplings[site, partner, old_state, :]
        mean = float(np.dot(background[partner], delta_j))
        mean_sum += mean
        directed[partner] = float(delta_j[int(sequence[partner])] - mean)
    background_term = -(delta_h + mean_sum)
    group_terms = []
    for group in groups:
        total = 0.0
        for partner in np.asarray(group, dtype=int):
            if int(partner) == site:
                continue
            total -= float(directed[int(partner)])
        group_terms.append(total)
    return background_term, tuple(group_terms)


def components_from_directional(
    sequence: np.ndarray,
    site: int,
    new_state: int,
    directional_fields: list[np.ndarray],
    directional_couplings: list[np.ndarray],
    background: np.ndarray,
    groups: list[np.ndarray],
) -> tuple[float, tuple[float, ...], float]:
    """Apply one site's field recovery and each pair's Ising average, then split ΔE.

    The third return is the mutation energy of those same blocks. Pairs are
    formed one at a time; this does not allocate an L by L coupling tensor.
    """
    site = int(site)
    new_state = int(new_state)
    length = len(directional_fields)
    recovered = recovered_site_field(directional_fields[site], directional_couplings[site])
    old_state = int(sequence[site])
    delta_h = float(recovered[new_state] - recovered[old_state])
    directed = np.zeros(length, dtype=np.float64)
    mean_sum = 0.0
    energy = -delta_h
    for partner in range(length):
        if partner == site:
            continue
        block = paired_ising_block(
            packed_block(directional_couplings[site], site, partner),
            packed_block(directional_couplings[partner], partner, site),
        )
        delta_j = block[new_state, :] - block[old_state, :]
        mean = float(np.dot(background[partner], delta_j))
        mean_sum += mean
        partner_state = int(sequence[partner])
        directed[partner] = float(delta_j[partner_state] - mean)
        energy -= float(block[new_state, partner_state] - block[old_state, partner_state])
    background_term = -(delta_h + mean_sum)
    group_terms = []
    for group in groups:
        total = 0.0
        for partner in np.asarray(group, dtype=int):
            if int(partner) == site:
                continue
            total -= float(directed[int(partner)])
        group_terms.append(total)
    return background_term, tuple(group_terms), energy

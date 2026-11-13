"""Potts energy of one fixed gauge pair.

The conditional model uses exp(h + J) as its unnormalized probability.
The energy whose decrease raises that probability is the negative of those
parameters. Pairs are counted once, with i < j.
"""

from __future__ import annotations

import numpy as np


def sequence_energy(sequence: np.ndarray, fields: np.ndarray, couplings: np.ndarray) -> float:
    """Return E(sequence) for fields (L, q) and couplings (L, L, q, q)."""
    sequence = np.asarray(sequence)
    fields = np.asarray(fields, dtype=np.float64)
    couplings = np.asarray(couplings, dtype=np.float64)
    length = sequence.shape[0]
    energy = -float(fields[np.arange(length), sequence].sum())
    left, right = np.triu_indices(length, 1)
    energy -= float(couplings[left, right, sequence[left], sequence[right]].sum())
    return energy


def mutation_energy(
    sequence: np.ndarray,
    site: int,
    new_state: int,
    fields: np.ndarray,
    couplings: np.ndarray,
) -> float:
    """Return E(mutant) - E(sequence) for one substitution."""
    sequence = np.asarray(sequence)
    fields = np.asarray(fields, dtype=np.float64)
    couplings = np.asarray(couplings, dtype=np.float64)
    old_state = int(sequence[site])
    delta = -(float(fields[site, new_state]) - float(fields[site, old_state]))
    others = np.arange(sequence.shape[0]) != site
    partner_states = sequence[others]
    delta -= float(
        couplings[site, others, new_state, partner_states].sum()
        - couplings[site, others, old_state, partner_states].sum()
    )
    return delta

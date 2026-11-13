"""Field recovery and the Ising average for one asymmetric Potts fit.

The field uses the directional couplings before the gauge map. The averaged
Ising couplings are a different object and are not fed back into that sum.
"""

from __future__ import annotations

import numpy as np


def ising_couplings(couplings: np.ndarray) -> np.ndarray:
    """Map one q by q block to zero row and column sums.

    The averages are uniform over the q states.
    """
    block = np.asarray(couplings, dtype=np.float64)
    return (
        block
        - block.mean(axis=0, keepdims=True)
        - block.mean(axis=1, keepdims=True)
        + block.mean()
    )


def recover_fields(fields: np.ndarray, directional: np.ndarray) -> np.ndarray:
    """Recover fields from pre-transform directional couplings.

    ``directional[i, j, a, b]`` is the coupling at target ``i``, partner ``j``,
    target state ``a`` and partner state ``b``. Diagonal blocks must be zero.
    The partner-state average is uniform, including the gap state.
    """
    fields = np.asarray(fields, dtype=np.float64)
    directional = np.asarray(directional, dtype=np.float64)
    partner_mean = directional.mean(axis=3)
    return fields + partner_mean.sum(axis=1)


def average_couplings(directional: np.ndarray) -> np.ndarray:
    """Ising-transform each direction, then average on aligned state axes.

    The returned array satisfies ``J[j, i, b, a] == J[i, j, a, b]``.
    """
    directional = np.asarray(directional, dtype=np.float64)
    length, _, q, _ = directional.shape
    averaged = np.zeros((length, length, q, q), dtype=np.float64)
    for i in range(length):
        for j in range(i + 1, length):
            from_i = ising_couplings(directional[i, j])
            from_j = ising_couplings(directional[j, i]).T
            block = 0.5 * (from_i + from_j)
            averaged[i, j] = block
            averaged[j, i] = block.T
    return averaged

"""Exact neighbor weights on the 263-column integer alignment.

The neighbor test is the integer count c_ab >= 211. Rows are not deduplicated.
The blocked counter is the routine to call. It accumulates one count per row and
does not store sequence pairs.
"""

from __future__ import annotations

import numpy as np

N_COLUMNS = 263
MIN_MATCHES = 211


def neighbor_counts(msa: np.ndarray, block_rows: int = 64) -> np.ndarray:
    """Return m_b, including each row itself, for every row of ``msa``."""
    alignment = _as_code_matrix(msa)
    if block_rows < 1:
        raise ValueError("block_rows must be positive")
    n_rows = alignment.shape[0]
    counts = np.ones(n_rows, dtype=np.int64)
    for start in range(0, n_rows, block_rows):
        stop = min(start + block_rows, n_rows)
        left = alignment[start:stop]
        for other in range(start, n_rows, block_rows):
            other_stop = min(other + block_rows, n_rows)
            right = alignment[other:other_stop]
            matched = np.count_nonzero(
                left[:, None, :] == right[None, :, :],
                axis=2,
            )
            if start == other:
                left_index, right_index = np.triu_indices(stop - start, k=1)
                keep = matched[left_index, right_index] >= MIN_MATCHES
                chosen_left = left_index[keep]
                chosen_right = right_index[keep]
            else:
                chosen_left, chosen_right = np.nonzero(matched >= MIN_MATCHES)
            np.add.at(counts, start + chosen_left, 1)
            np.add.at(counts, other + chosen_right, 1)
    return counts


def sequence_weights(
    msa: np.ndarray,
    block_rows: int = 64,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return neighbor counts, weights 1/m_b, and M_eff."""
    counts = neighbor_counts(msa, block_rows=block_rows)
    if np.any(counts < 1):
        raise RuntimeError("every row must count itself")
    weights = np.reciprocal(counts.astype(np.float64))
    effective = float(np.sum(weights, dtype=np.float64))
    return counts, weights, effective


def _as_code_matrix(msa: np.ndarray) -> np.ndarray:
    alignment = np.asarray(msa)
    if alignment.ndim != 2 or alignment.shape[1] != N_COLUMNS:
        raise ValueError(
            f"expected shape (B, {N_COLUMNS}), got {alignment.shape}"
        )
    if not np.issubdtype(alignment.dtype, np.integer):
        raise ValueError("alignment codes must be integers")
    return np.ascontiguousarray(alignment)

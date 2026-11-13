"""Count each 53-block candidate once, then stop a distance scan at 53 mismatches.

The neighbor test is unchanged: at least 211 identical columns. ``block_index``
still contains the older scan that compares a pair again for every block.
This module is the one that merges those blocks before the comparison.
"""

from __future__ import annotations

import ctypes
import subprocess
from pathlib import Path

import numpy as np

import block_index
import sequence_weights

N_COLUMNS = sequence_weights.N_COLUMNS
MISMATCH_LIMIT = N_COLUMNS - sequence_weights.MIN_MATCHES + 1
HERE = Path(__file__).resolve().parent
LIBRARY = HERE / "candidate_scan_native.so"


def discrimination(alignment: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return D_k and a column order from high D_k to low, ties by column index."""
    coded = sequence_weights._as_code_matrix(alignment)
    scores = np.empty(N_COLUMNS, dtype=np.float64)
    scale = float(coded.shape[0])
    for column in range(N_COLUMNS):
        counts = np.bincount(coded[:, column], minlength=block_index.ALPHABET)
        probability = counts.astype(np.float64) / scale
        scores[column] = 1.0 - float(np.dot(probability, probability))
    order = np.lexsort((np.arange(N_COLUMNS), -scores)).astype(np.int32)
    return scores, order


def column_block_ids(blocks: list[tuple[int, ...]]) -> np.ndarray:
    """Map each mature column to the one block that contains it."""
    block_index.validate_partition(blocks)
    owner = np.empty(N_COLUMNS, dtype=np.int32)
    for block_id, columns in enumerate(blocks):
        owner[list(columns)] = block_id
    return owner


def build_block_index(
    alignment: np.ndarray,
    blocks: list[tuple[int, ...]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return codes, per-row bucket bounds, and the row ids stored in each bucket."""
    coded = np.ascontiguousarray(sequence_weights._as_code_matrix(alignment), dtype=np.uint8)
    block_index.validate_partition(blocks)
    n_rows = coded.shape[0]
    n_blocks = len(blocks)
    starts = np.empty((n_blocks, n_rows), dtype=np.int32)
    ends = np.empty((n_blocks, n_rows), dtype=np.int32)
    members = np.empty((n_blocks, n_rows), dtype=np.int32)
    for block_id, columns in enumerate(blocks):
        keys = block_index.pack_block(coded, columns)
        order = np.argsort(keys, kind="mergesort").astype(np.int32)
        ordered = keys[order]
        change = np.empty(n_rows, dtype=bool)
        change[0] = True
        change[1:] = ordered[1:] != ordered[:-1]
        group_id = np.cumsum(change, dtype=np.int32) - 1
        group_starts = np.flatnonzero(change).astype(np.int32)
        group_ends = np.empty(group_starts.size, dtype=np.int32)
        group_ends[:-1] = group_starts[1:]
        group_ends[-1] = n_rows
        sorted_start = group_starts[group_id]
        sorted_end = group_ends[group_id]
        starts[block_id, order] = sorted_start
        ends[block_id, order] = sorted_end
        members[block_id] = order
    return coded, starts, ends, members


def unique_degree_python(
    starts: np.ndarray,
    ends: np.ndarray,
    members: np.ndarray,
) -> tuple[int, np.ndarray]:
    """Reference union. Each unordered pair is kept once, from the smaller index."""
    n_blocks, n_rows = starts.shape
    visited = np.zeros(n_rows, dtype=np.int32)
    degree = np.zeros(n_rows, dtype=np.int64)
    pairs = 0
    for row in range(n_rows):
        stamp = row + 1
        for block_id in range(n_blocks):
            group = members[block_id, starts[block_id, row] : ends[block_id, row]]
            for other in group:
                other = int(other)
                if other <= row or visited[other] == stamp:
                    continue
                visited[other] = stamp
                pairs += 1
                degree[row] += 1
                degree[other] += 1
    return pairs, degree


def early_compare(
    left: np.ndarray,
    right: np.ndarray,
    column_order: np.ndarray,
    source_block: int,
    column_block: np.ndarray,
    mismatch_limit: int = MISMATCH_LIMIT,
) -> tuple[bool, int]:
    """Return whether the pair is a neighbor, and how many columns were read."""
    mismatches = 0
    compared = 0
    for column in column_order:
        if int(column_block[column]) == source_block:
            continue
        compared += 1
        if int(left[column]) != int(right[column]):
            mismatches += 1
            if mismatches == mismatch_limit:
                return False, compared
    return mismatches < mismatch_limit, compared


def pack_bitplanes(alignment: np.ndarray, column_order: np.ndarray) -> np.ndarray:
    """Pack codes into 5 bit planes of 5 uint64 words. Unused high bits stay zero."""
    ordered = np.ascontiguousarray(alignment[:, column_order], dtype=np.uint8)
    if ordered.shape[1] != N_COLUMNS:
        raise ValueError(f"expected {N_COLUMNS} columns, found {ordered.shape[1]}")
    planes = np.zeros((ordered.shape[0], 5, 5), dtype=np.uint64)
    for bit in range(5):
        flags = ((ordered >> bit) & 1).astype(np.uint64)
        for word in range(5):
            piece = flags[:, word * 64 : (word + 1) * 64]
            packed = np.zeros(ordered.shape[0], dtype=np.uint64)
            for offset in range(piece.shape[1]):
                packed |= piece[:, offset] << np.uint64(offset)
            planes[:, bit, word] = packed
    return np.ascontiguousarray(planes)


def compile_library() -> ctypes.CDLL:
    """Compile the scan if the shared library is older than its source."""
    source = HERE / "candidate_scan.c"
    if not LIBRARY.exists() or LIBRARY.stat().st_mtime < source.stat().st_mtime:
        subprocess.run(
            [
                "gcc",
                "-O3",
                "-fPIC",
                "-shared",
                "-Wall",
                "-Wextra",
                str(source),
                "-o",
                str(LIBRARY),
            ],
            check=True,
        )
    library = ctypes.CDLL(str(LIBRARY))
    library.unique_candidate_degree.argtypes = [
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
    ]
    library.early_abort_sample.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int64),
    ]
    library.hamming_bitplane_pairs.argtypes = [
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.POINTER(ctypes.c_uint16),
    ]
    return library


def unique_candidate_degree(
    starts: np.ndarray,
    ends: np.ndarray,
    members: np.ndarray,
) -> tuple[int, np.ndarray]:
    """Union size C and the symmetric per-row candidate count."""
    library = compile_library()
    n_blocks, n_rows = starts.shape
    bound_starts = np.ascontiguousarray(starts, dtype=np.int32)
    bound_ends = np.ascontiguousarray(ends, dtype=np.int32)
    bound_members = np.ascontiguousarray(members, dtype=np.int32)
    degree = np.zeros(n_rows, dtype=np.int64)
    pairs = ctypes.c_int64()
    library.unique_candidate_degree(
        n_rows,
        n_blocks,
        bound_starts.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        bound_ends.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        bound_members.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        degree.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.byref(pairs),
    )
    if pairs.value < 0:
        raise MemoryError("candidate scan could not allocate the row stamps")
    return int(pairs.value), degree


def early_abort_stats(
    alignment: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    members: np.ndarray,
    column_block: np.ndarray,
    column_order: np.ndarray,
    sample_rows: np.ndarray,
    higher_only: bool,
    neighbor_counts: np.ndarray | None = None,
) -> dict[str, object]:
    """Measure columns read on the sampled rows' deduplicated candidates."""
    library = compile_library()
    coded = np.ascontiguousarray(alignment, dtype=np.uint8)
    sample = np.ascontiguousarray(sample_rows, dtype=np.int32)
    bound_starts = np.ascontiguousarray(starts, dtype=np.int32)
    bound_ends = np.ascontiguousarray(ends, dtype=np.int32)
    bound_members = np.ascontiguousarray(members, dtype=np.int32)
    owners = np.ascontiguousarray(column_block, dtype=np.int32)
    order = np.ascontiguousarray(column_order, dtype=np.int32)
    reject = np.zeros(N_COLUMNS + 1, dtype=np.int64)
    accept = np.zeros(N_COLUMNS + 1, dtype=np.int64)
    n_reject = ctypes.c_int64()
    n_accept = ctypes.c_int64()
    compared = ctypes.c_int64()
    counts_pointer = None
    if neighbor_counts is not None:
        if neighbor_counts.dtype != np.int64 or not neighbor_counts.flags.c_contiguous:
            raise ValueError("neighbor_counts must be a contiguous int64 array")
        if neighbor_counts.shape != (coded.shape[0],):
            raise ValueError("neighbor_counts must have one entry per row")
        counts_pointer = neighbor_counts.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
    library.early_abort_sample(
        coded.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        coded.shape[0],
        coded.shape[1],
        starts.shape[0],
        bound_starts.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        bound_ends.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        bound_members.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        owners.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        order.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        sample.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        sample.size,
        1 if higher_only else 0,
        MISMATCH_LIMIT,
        reject.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        accept.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
        ctypes.byref(n_reject),
        ctypes.byref(n_accept),
        ctypes.byref(compared),
        counts_pointer,
    )
    if n_reject.value < 0:
        raise MemoryError("early-abort scan could not allocate the row stamps")
    total = int(n_reject.value + n_accept.value)
    return {
        "n_reject": int(n_reject.value),
        "n_accept": int(n_accept.value),
        "columns_compared": int(compared.value),
        "mean_columns": float(compared.value) / total if total else 0.0,
        "reject_hist": reject,
        "accept_hist": accept,
    }


def hamming_bitplane(
    planes: np.ndarray,
    left_rows: np.ndarray,
    right_rows: np.ndarray,
    early_limit: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the bit-plane distance and how many 64-column words were read."""
    library = compile_library()
    packed = np.ascontiguousarray(planes, dtype=np.uint64)
    left = np.ascontiguousarray(left_rows, dtype=np.int32)
    right = np.ascontiguousarray(right_rows, dtype=np.int32)
    distances = np.empty(left.size, dtype=np.uint16)
    words = np.empty(left.size, dtype=np.uint16)
    library.hamming_bitplane_pairs(
        packed.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
        packed.shape[0],
        left.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        right.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        left.size,
        early_limit,
        distances.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
        words.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
    )
    return distances, words



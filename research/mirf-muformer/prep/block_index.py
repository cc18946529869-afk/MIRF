"""Exact candidate generation for the 263-column neighbor test.

A neighbor is still a pair with at least 211 identical columns. The 53 blocks
only propose pairs. Every proposed pair is checked on all 263 columns, and a
pair that collides in several blocks is counted once. Rows are not merged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import sequence_weights

N_COLUMNS = sequence_weights.N_COLUMNS
MIN_MATCHES = sequence_weights.MIN_MATCHES
N_BLOCKS = 53
N_LONG_BLOCKS = 51
LONG_WIDTH = 5
SHORT_WIDTH = 4
ALPHABET = 21
LOW_COVERAGE_REFERENCE = (
    tuple(range(24, 40))
    + tuple(range(83, 86))
    + (142,)
    + tuple(range(166, 172))
    + (255,)
    + tuple(range(266, 287))
)


@dataclass(frozen=True)
class BlockStats:
    n_buckets: int
    max_bucket: int
    mean_bucket: float
    n_colliding_buckets: int
    raw_pairs: int


def low_coverage_columns() -> tuple[int, ...]:
    """0-based mature-column indices of the 48 low-coverage positions."""
    columns = tuple(position - 24 for position in LOW_COVERAGE_REFERENCE)
    if len(columns) != 48 or len(set(columns)) != 48:
        raise RuntimeError("low-coverage positions are not 48 distinct columns")
    return columns


def validate_partition(blocks: list[tuple[int, ...]]) -> None:
    """Require 53 non-overlapping blocks covering every mature column."""
    if len(blocks) != N_BLOCKS:
        raise ValueError(f"expected {N_BLOCKS} blocks, found {len(blocks)}")
    widths = [len(block) for block in blocks]
    if widths.count(LONG_WIDTH) != N_LONG_BLOCKS or widths.count(SHORT_WIDTH) != 2:
        raise ValueError("expected 51 blocks of 5 columns and 2 blocks of 4")
    flat = [column for block in blocks for column in block]
    if len(flat) != N_COLUMNS or set(flat) != set(range(N_COLUMNS)):
        raise ValueError("blocks must cover each of the 263 columns once")
    for block in blocks:
        if len(set(block)) != len(block):
            raise ValueError("a block repeats a column")


def column_entropy(alignment: np.ndarray) -> np.ndarray:
    """Shannon entropy of each column, in nats, over the 21 codes."""
    coded = sequence_weights._as_code_matrix(alignment)
    entropy = np.empty(N_COLUMNS, dtype=np.float64)
    for column in range(N_COLUMNS):
        counts = np.bincount(coded[:, column], minlength=ALPHABET)
        probability = counts[counts > 0] / coded.shape[0]
        entropy[column] = -np.sum(probability * np.log(probability))
    return entropy


def pairwise_mutual_information(alignment: np.ndarray) -> np.ndarray:
    """Symmetric mutual information between mature columns, in nats."""
    coded = sequence_weights._as_code_matrix(alignment)
    n_rows = coded.shape[0]
    entropy = column_entropy(coded)
    information = np.zeros((N_COLUMNS, N_COLUMNS), dtype=np.float64)
    for left in range(N_COLUMNS):
        left_code = coded[:, left].astype(np.int32)
        for right in range(left + 1, N_COLUMNS):
            joint = np.bincount(
                left_code * ALPHABET + coded[:, right],
                minlength=ALPHABET * ALPHABET,
            )
            probability = joint[joint > 0] / n_rows
            joint_entropy = -np.sum(probability * np.log(probability))
            value = entropy[left] + entropy[right] - joint_entropy
            information[left, right] = value
            information[right, left] = value
    return information


def contiguous_partition() -> list[tuple[int, ...]]:
    """51 adjacent 5-column blocks, then two adjacent 4-column blocks."""
    blocks = [tuple(range(start, start + LONG_WIDTH)) for start in range(0, 255, LONG_WIDTH)]
    blocks.append(tuple(range(255, 259)))
    blocks.append(tuple(range(259, 263)))
    validate_partition(blocks)
    return blocks


def coverage_sequential_partition() -> list[tuple[int, ...]]:
    """One low-coverage column with the next four high-coverage columns."""
    return _one_low_partition(
        high_order=tuple(
            column
            for column in range(N_COLUMNS)
            if column not in set(low_coverage_columns())
        )
    )


def coverage_entropy_spread_partition(entropy: np.ndarray) -> list[tuple[int, ...]]:
    """Give every low-coverage block a mix of high-entropy columns."""
    highs = _high_columns()
    order = tuple(sorted(highs, key=lambda column: (-entropy[column], column)))
    return _one_low_partition(high_order=order, spread=True)


def coverage_entropy_front_partition(entropy: np.ndarray) -> list[tuple[int, ...]]:
    """Give the earliest low-coverage blocks the highest-entropy columns."""
    highs = _high_columns()
    order = tuple(sorted(highs, key=lambda column: (-entropy[column], column)))
    return _one_low_partition(high_order=order, spread=False)


def coverage_entropy_balanced_partition(entropy: np.ndarray) -> list[tuple[int, ...]]:
    """Deal high-entropy columns across all 53 blocks, including the tail."""
    order = sorted(_high_columns(), key=lambda column: (-entropy[column], column))
    capacities = [4] * 48 + [5] * 3 + [4] * 2
    assigned: list[list[int]] = [[] for _ in capacities]
    remaining = capacities[:]
    cursor = 0
    for column in order:
        for step in range(N_BLOCKS):
            position = (cursor + step) % N_BLOCKS
            if remaining[position] == 0:
                continue
            assigned[position].append(column)
            remaining[position] -= 1
            cursor = (position + 1) % N_BLOCKS
            break
    if any(remaining):
        raise RuntimeError("balanced partition left a block short")
    blocks = [
        tuple(sorted((low, *assigned[index])))
        for index, low in enumerate(low_coverage_columns())
    ]
    blocks.extend(tuple(sorted(group)) for group in assigned[48:])
    validate_partition(blocks)
    return blocks


def coverage_information_balanced_partition(
    entropy: np.ndarray,
    information: np.ndarray,
) -> list[tuple[int, ...]]:
    """Spread columns by entropy, then prefer blocks they are less dependent on."""
    capacities = [4] * 48 + [5] * 3 + [4] * 2
    assigned: list[list[int]] = [[] for _ in capacities]
    for index, low in enumerate(low_coverage_columns()):
        assigned[index].append(low)
    remaining = capacities[:]
    order = sorted(_high_columns(), key=lambda column: (-float(entropy[column]), column))
    for column in order:
        choice = None
        choice_key = None
        for position, room in enumerate(remaining):
            if room == 0:
                continue
            redundancy = sum(float(information[column, member]) for member in assigned[position])
            key = (redundancy, -room, position)
            if choice_key is None or key < choice_key:
                choice_key = key
                choice = position
        if choice is None:
            raise RuntimeError("no block had room for a high-coverage column")
        assigned[choice].append(column)
        remaining[choice] -= 1
    blocks = [tuple(sorted(group)) for group in assigned]
    validate_partition(blocks)
    return blocks


def coverage_information_partition(
    entropy: np.ndarray,
    information: np.ndarray,
) -> list[tuple[int, ...]]:
    """Anchor each block on one low-coverage column and add diverse columns."""
    remaining = set(_high_columns())
    lows = sorted(low_coverage_columns(), key=lambda column: (entropy[column], column))
    blocks: list[tuple[int, ...]] = []
    for low in lows:
        chosen = [low]
        for _ in range(4):
            best = min(
                remaining,
                key=lambda column: (
                    -(entropy[column] - sum(information[column, member] for member in chosen)),
                    column,
                ),
            )
            chosen.append(best)
            remaining.remove(best)
        blocks.append(tuple(sorted(chosen)))
    rest = tuple(sorted(remaining, key=lambda column: (-entropy[column], column)))
    blocks.extend(_tail_blocks(rest))
    validate_partition(blocks)
    return blocks


def random_partition(seed: int, one_low: bool) -> list[tuple[int, ...]]:
    """A reproducible random cover. ``one_low`` keeps the 48 anchors apart."""
    generator = np.random.default_rng(seed)
    if not one_low:
        columns = generator.permutation(N_COLUMNS)
        blocks = [
            tuple(sorted(int(column) for column in columns[start : start + LONG_WIDTH]))
            for start in range(0, 255, LONG_WIDTH)
        ]
        tail = [int(column) for column in columns[255:]]
        blocks.append(tuple(sorted(tail[:4])))
        blocks.append(tuple(sorted(tail[4:])))
        validate_partition(blocks)
        return blocks
    lows = list(low_coverage_columns())
    highs = list(_high_columns())
    generator.shuffle(lows)
    generator.shuffle(highs)
    return _one_low_partition(high_order=tuple(highs), low_order=tuple(lows))


def pack_block(alignment: np.ndarray, columns: tuple[int, ...]) -> np.ndarray:
    """Pack one block into a base-21 integer. Codes must be 0 through 20."""
    packed = np.zeros(alignment.shape[0], dtype=np.int32)
    place = 1
    for column in columns:
        packed += alignment[:, column].astype(np.int32) * place
        place *= ALPHABET
    return packed


def block_bucket_stats(alignment: np.ndarray, columns: tuple[int, ...]) -> BlockStats:
    """Occupied-bucket sizes and the raw unordered collisions inside them."""
    _keys, counts = np.unique(pack_block(alignment, columns), return_counts=True)
    widths = counts.astype(np.int64)
    return BlockStats(
        n_buckets=int(widths.size),
        max_bucket=int(widths.max()) if widths.size else 0,
        mean_bucket=float(widths.mean()) if widths.size else 0.0,
        n_colliding_buckets=int(np.count_nonzero(widths >= 2)),
        raw_pairs=int(np.sum(widths * (widths - 1) // 2, dtype=np.int64)),
    )


def partition_bucket_stats(
    alignment: np.ndarray,
    blocks: list[tuple[int, ...]],
) -> tuple[BlockStats, list[BlockStats]]:
    """Sum raw collisions over blocks. The same pair can be counted more than once."""
    coded = sequence_weights._as_code_matrix(alignment)
    validate_partition(blocks)
    per_block = [block_bucket_stats(coded, block) for block in blocks]
    total = BlockStats(
        n_buckets=sum(item.n_buckets for item in per_block),
        max_bucket=max(item.max_bucket for item in per_block),
        mean_bucket=float(np.mean([item.mean_bucket for item in per_block])),
        n_colliding_buckets=sum(item.n_colliding_buckets for item in per_block),
        raw_pairs=sum(item.raw_pairs for item in per_block),
    )
    return total, per_block


def neighbor_counts_indexed(
    msa: np.ndarray,
    blocks: list[tuple[int, ...]],
    max_raw_pairs: int = 2_000_000,
) -> np.ndarray:
    """Return m_b from verified block collisions.

    ``max_raw_pairs`` stops the routine before a partition that still collides
    in nearly every row is treated as an exact full-scale run.
    """
    alignment = sequence_weights._as_code_matrix(msa)
    total, _per_block = partition_bucket_stats(alignment, blocks)
    if total.raw_pairs > max_raw_pairs:
        raise ValueError(
            f"raw colliding pairs are {total.raw_pairs}, above {max_raw_pairs}"
        )
    counts = np.ones(alignment.shape[0], dtype=np.int64)
    seen: set[tuple[int, int]] = set()
    for block in blocks:
        keys = pack_block(alignment, block)
        order = np.argsort(keys, kind="mergesort")
        ordered = keys[order]
        cuts = np.flatnonzero(ordered[1:] != ordered[:-1]) + 1
        starts = np.concatenate((np.asarray([0]), cuts))
        stops = np.concatenate((cuts, np.asarray([alignment.shape[0]])))
        for start, stop in zip(starts, stops):
            if stop - start < 2:
                continue
            _add_verified_pairs(counts, seen, alignment, order[start:stop])
    return counts


def _add_verified_pairs(
    counts: np.ndarray,
    seen: set[tuple[int, int]],
    alignment: np.ndarray,
    rows: np.ndarray,
) -> None:
    """Verify one hash bucket on all 263 columns and count each pair once."""
    group_size = int(rows.size)
    block_rows = 32
    for start in range(0, group_size, block_rows):
        stop = min(start + block_rows, group_size)
        left = alignment[rows[start:stop]]
        for other in range(start, group_size, block_rows):
            other_stop = min(other + block_rows, group_size)
            right = alignment[rows[other:other_stop]]
            matched = np.count_nonzero(
                left[:, None, :] == right[None, :, :],
                axis=2,
            )
            if start == other:
                left_index, right_index = np.triu_indices(stop - start, k=1)
                keep = matched[left_index, right_index] >= MIN_MATCHES
                pairs = zip(left_index[keep], right_index[keep])
                right_base = start
            else:
                left_index, right_index = np.nonzero(matched >= MIN_MATCHES)
                pairs = zip(left_index, right_index)
                right_base = other
            for local_left, local_right in pairs:
                left_row = int(rows[start + int(local_left)])
                right_row = int(rows[right_base + int(local_right)])
                if left_row > right_row:
                    left_row, right_row = right_row, left_row
                pair = (left_row, right_row)
                if pair in seen:
                    continue
                seen.add(pair)
                counts[left_row] += 1
                counts[right_row] += 1


def _high_columns() -> tuple[int, ...]:
    low = set(low_coverage_columns())
    return tuple(column for column in range(N_COLUMNS) if column not in low)


def _tail_blocks(columns: tuple[int, ...]) -> list[tuple[int, ...]]:
    if len(columns) != 23:
        raise ValueError("the high-coverage remainder must have 23 columns")
    cuts = (0, 5, 10, 15, 19, 23)
    return [tuple(columns[start:stop]) for start, stop in zip(cuts, cuts[1:])]


def _one_low_partition(
    high_order: tuple[int, ...],
    spread: bool = False,
    low_order: tuple[int, ...] | None = None,
) -> list[tuple[int, ...]]:
    lows = low_coverage_columns() if low_order is None else low_order
    if len(high_order) != 215 or len(lows) != 48:
        raise ValueError("one-low partitions need 48 low and 215 high columns")
    if spread:
        assigned = [[] for _ in lows]
        for offset, column in enumerate(high_order[:192]):
            assigned[offset % 48].append(column)
        blocks = [
            tuple(sorted((low, *chosen)))
            for low, chosen in zip(lows, assigned)
        ]
        blocks.extend(_tail_blocks(tuple(high_order[192:])))
    else:
        blocks = [
            tuple(sorted((low, *high_order[4 * index : 4 * index + 4])))
            for index, low in enumerate(lows)
        ]
        blocks.extend(_tail_blocks(tuple(high_order[192:])))
    validate_partition(blocks)
    return blocks

"""Checks for one-time candidate verification and early Hamming rejection."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import block_index
import candidate_scan
import sequence_weights


def brute_unique(alignment: np.ndarray, blocks: list[tuple[int, ...]]) -> tuple[int, np.ndarray]:
    n_rows = alignment.shape[0]
    degree = np.zeros(n_rows, dtype=np.int64)
    pairs = 0
    for left in range(n_rows):
        for right in range(left + 1, n_rows):
            shared = any(
                np.array_equal(alignment[left, list(block)], alignment[right, list(block)])
                for block in blocks
            )
            if not shared:
                continue
            pairs += 1
            degree[left] += 1
            degree[right] += 1
    return pairs, degree


def deduped_neighbor_counts(alignment: np.ndarray, blocks: list[tuple[int, ...]]) -> tuple[np.ndarray, int]:
    _coded, starts, ends, members = candidate_scan.build_block_index(alignment, blocks)
    owners = candidate_scan.column_block_ids(blocks)
    _scores, order = candidate_scan.discrimination(alignment)
    n_rows = alignment.shape[0]
    visited = np.zeros(n_rows, dtype=np.int32)
    counts = np.ones(n_rows, dtype=np.int64)
    comparisons = 0
    for row in range(n_rows):
        stamp = row + 1
        for block_id in range(len(blocks)):
            group = members[block_id, starts[block_id, row] : ends[block_id, row]]
            for other in group:
                other = int(other)
                if other <= row or visited[other] == stamp:
                    continue
                visited[other] = stamp
                comparisons += 1
                is_neighbor, _read = candidate_scan.early_compare(
                    alignment[row],
                    alignment[other],
                    order,
                    block_id,
                    owners,
                )
                if is_neighbor:
                    counts[row] += 1
                    counts[other] += 1
    return counts, comparisons


class UnionTest(unittest.TestCase):
    def test_compiled_degree_matches_the_pair_union(self) -> None:
        generator = np.random.default_rng(211)
        alignment = generator.integers(0, 21, size=(24, sequence_weights.N_COLUMNS), dtype=np.uint8)
        alignment[3] = alignment[1]
        blocks = block_index.contiguous_partition()
        _coded, starts, ends, members = candidate_scan.build_block_index(alignment, blocks)
        python_pairs, python_degree = candidate_scan.unique_degree_python(starts, ends, members)
        compiled_pairs, compiled_degree = candidate_scan.unique_candidate_degree(starts, ends, members)
        brute_pairs, brute_degree = brute_unique(alignment, blocks)
        self.assertEqual(compiled_pairs, brute_pairs)
        self.assertEqual(python_pairs, brute_pairs)
        np.testing.assert_array_equal(compiled_degree, brute_degree)
        np.testing.assert_array_equal(python_degree, brute_degree)
        self.assertEqual(int(compiled_degree.sum()), 2 * compiled_pairs)

    def test_each_unique_candidate_is_compared_once(self) -> None:
        generator = np.random.default_rng(7)
        alignment = generator.integers(0, 21, size=(18, sequence_weights.N_COLUMNS), dtype=np.uint8)
        blocks = block_index.coverage_sequential_partition()
        counts, comparisons = deduped_neighbor_counts(alignment, blocks)
        _coded, starts, ends, members = candidate_scan.build_block_index(alignment, blocks)
        pairs, _degree = candidate_scan.unique_degree_python(starts, ends, members)
        self.assertEqual(comparisons, pairs)
        np.testing.assert_array_equal(counts, sequence_weights.neighbor_counts(alignment))

    def test_fifty_two_mismatches_survive_early_stop(self) -> None:
        blocks = block_index.contiguous_partition()
        alignment = np.ones((2, sequence_weights.N_COLUMNS), dtype=np.uint8)
        for block in blocks[:-1]:
            alignment[1, block[0]] = 2
        self.assertEqual(int(np.count_nonzero(alignment[0] == alignment[1])), 211)
        owners = candidate_scan.column_block_ids(blocks)
        order = np.arange(sequence_weights.N_COLUMNS, dtype=np.int32)
        is_neighbor, compared = candidate_scan.early_compare(
            alignment[0], alignment[1], order, len(blocks) - 1, owners
        )
        self.assertTrue(is_neighbor)
        self.assertEqual(compared, sequence_weights.N_COLUMNS - len(blocks[-1]))


class NeighborCountTest(unittest.TestCase):
    def test_accepted_pairs_update_both_endpoints(self) -> None:
        generator = np.random.default_rng(11)
        alignment = generator.integers(
            0, 21, size=(16, sequence_weights.N_COLUMNS), dtype=np.uint8
        )
        alignment[2] = alignment[0]
        alignment[4] = 0
        blocks = block_index.contiguous_partition()
        coded, starts, ends, members = candidate_scan.build_block_index(alignment, blocks)
        owners = candidate_scan.column_block_ids(blocks)
        _scores, order = candidate_scan.discrimination(coded)
        counts = np.ones(coded.shape[0], dtype=np.int64)
        stats = candidate_scan.early_abort_stats(
            coded,
            starts,
            ends,
            members,
            owners,
            order,
            np.arange(coded.shape[0], dtype=np.int32),
            higher_only=True,
            neighbor_counts=counts,
        )
        reference = sequence_weights.neighbor_counts(alignment)
        np.testing.assert_array_equal(counts, reference)
        self.assertEqual(int(np.sum(counts - 1)), 2 * stats["n_accept"])
        self.assertGreaterEqual(int(counts.min()), 1)
        self.assertGreaterEqual(int(counts[0]), 2)
        self.assertEqual(int(counts[0]), int(counts[2]))


class EarlyAbortTest(unittest.TestCase):
    def test_compiled_histogram_matches_python(self) -> None:
        generator = np.random.default_rng(19)
        alignment = generator.integers(0, 21, size=(12, sequence_weights.N_COLUMNS), dtype=np.uint8)
        blocks = block_index.random_partition(19, one_low=True)
        coded, starts, ends, members = candidate_scan.build_block_index(alignment, blocks)
        owners = candidate_scan.column_block_ids(blocks)
        _scores, order = candidate_scan.discrimination(coded)
        sample = np.arange(coded.shape[0], dtype=np.int32)
        stats = candidate_scan.early_abort_stats(
            coded, starts, ends, members, owners, order, sample, higher_only=True
        )
        reject = np.zeros(sequence_weights.N_COLUMNS + 1, dtype=np.int64)
        accept = np.zeros_like(reject)
        for row in range(coded.shape[0]):
            seen: set[int] = set()
            for block_id, block in enumerate(blocks):
                group = members[block_id, starts[block_id, row] : ends[block_id, row]]
                for other in group:
                    other = int(other)
                    if other <= row or other in seen:
                        continue
                    seen.add(other)
                    is_neighbor, compared = candidate_scan.early_compare(
                        coded[row], coded[other], order, block_id, owners
                    )
                    if is_neighbor:
                        accept[compared] += 1
                    else:
                        reject[compared] += 1
        np.testing.assert_array_equal(stats["reject_hist"], reject)
        np.testing.assert_array_equal(stats["accept_hist"], accept)


class BitPlaneTest(unittest.TestCase):
    def test_full_distance_matches_column_mismatches(self) -> None:
        generator = np.random.default_rng(5)
        alignment = generator.integers(0, 21, size=(30, sequence_weights.N_COLUMNS), dtype=np.uint8)
        alignment[4, 0] = 20
        alignment[4, 262] = 16
        order = np.arange(sequence_weights.N_COLUMNS, dtype=np.int32)
        planes = candidate_scan.pack_bitplanes(alignment, order)
        left = np.repeat(np.arange(30, dtype=np.int32), 30)
        right = np.tile(np.arange(30, dtype=np.int32), 30)
        distances, words = candidate_scan.hamming_bitplane(planes, left, right, early_limit=0)
        expected = np.count_nonzero(alignment[left] != alignment[right], axis=1)
        np.testing.assert_array_equal(distances, expected)
        self.assertTrue(np.all(words == 5))

    def test_early_word_stop_does_not_call_a_far_pair_a_neighbor(self) -> None:
        alignment = np.zeros((2, sequence_weights.N_COLUMNS), dtype=np.uint8)
        alignment[1, :80] = 1
        order = np.arange(sequence_weights.N_COLUMNS, dtype=np.int32)
        planes = candidate_scan.pack_bitplanes(alignment, order)
        distances, words = candidate_scan.hamming_bitplane(
            planes,
            np.asarray([0], dtype=np.int32),
            np.asarray([1], dtype=np.int32),
            early_limit=53,
        )
        self.assertGreaterEqual(int(distances[0]), 53)
        self.assertLess(int(words[0]), 5)
        self.assertGreater(int(np.count_nonzero(alignment[0] != alignment[1])), 52)


if __name__ == "__main__":
    unittest.main()

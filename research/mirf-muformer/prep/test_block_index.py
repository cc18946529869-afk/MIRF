"""Checks that 53-block candidates match the exact neighbor counts."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import block_index
import sequence_weights


class PartitionTest(unittest.TestCase):
    def test_low_coverage_positions_are_48_mature_columns(self) -> None:
        columns = block_index.low_coverage_columns()
        self.assertEqual(len(columns), 48)
        self.assertEqual(columns[0], 0)
        self.assertEqual(columns[-1], 262)

    def test_named_partitions_cover_every_column_once(self) -> None:
        entropy = np.linspace(0.1, 2.0, sequence_weights.N_COLUMNS)
        information = np.zeros((sequence_weights.N_COLUMNS, sequence_weights.N_COLUMNS))
        one_low = [
            block_index.coverage_sequential_partition(),
            block_index.coverage_entropy_spread_partition(entropy),
            block_index.coverage_entropy_front_partition(entropy),
            block_index.coverage_entropy_balanced_partition(entropy),
            block_index.coverage_information_partition(entropy, information),
            block_index.coverage_information_balanced_partition(entropy, information),
            block_index.random_partition(211, one_low=True),
        ]
        low = set(block_index.low_coverage_columns())
        for blocks in [block_index.contiguous_partition(), block_index.random_partition(211, one_low=False), *one_low]:
            block_index.validate_partition(blocks)
        for blocks in one_low:
            for block in blocks[:48]:
                self.assertEqual(len(set(block) & low), 1)

    def test_balanced_entropy_spreads_informative_columns_into_the_tail(self) -> None:
        entropy = np.arange(sequence_weights.N_COLUMNS, dtype=np.float64)
        blocks = block_index.coverage_entropy_balanced_partition(entropy)
        low = set(block_index.low_coverage_columns())
        pure = [block for block in blocks if low.isdisjoint(block)]
        pure_columns = {column for block in pure for column in block}
        weakest = set(sorted(set(range(sequence_weights.N_COLUMNS)) - low, key=lambda column: entropy[column])[:23])
        self.assertTrue(pure_columns - weakest)


class IndexedCountTest(unittest.TestCase):
    def test_52_mismatches_are_kept_and_53_are_not(self) -> None:
        blocks = block_index.contiguous_partition()
        msa = np.ones((3, sequence_weights.N_COLUMNS), dtype=np.int8)
        for block in blocks[:-1]:
            msa[1, block[0]] = 2
        for block in blocks:
            msa[2, block[0]] = 2
        self.assertEqual(int(np.count_nonzero(msa[0] == msa[1])), 211)
        self.assertEqual(int(np.count_nonzero(msa[0] == msa[2])), 210)
        counts = block_index.neighbor_counts_indexed(msa, blocks)
        np.testing.assert_array_equal(counts, sequence_weights.neighbor_counts(msa))
        self.assertEqual(int(counts[0]), 2)
        self.assertEqual(int(counts[2]), 2)

    def test_shared_block_with_210_matches_is_rejected(self) -> None:
        blocks = block_index.contiguous_partition()
        msa = np.ones((2, sequence_weights.N_COLUMNS), dtype=np.int8)
        msa[1, blocks[0][0]] = 2
        msa[1, blocks[0][1]] = 2
        for block in blocks[1:-1]:
            msa[1, block[0]] = 2
        self.assertEqual(int(np.count_nonzero(msa[0] == msa[1])), 210)
        self.assertTrue(np.array_equal(msa[0, list(blocks[-1])], msa[1, list(blocks[-1])]))
        counts = block_index.neighbor_counts_indexed(msa, blocks)
        np.testing.assert_array_equal(counts, [1, 1])

    def test_duplicate_and_all_gap_rows_stay(self) -> None:
        blocks = block_index.coverage_sequential_partition()
        duplicate = np.zeros((4, sequence_weights.N_COLUMNS), dtype=np.int8)
        duplicate[2:] = 3
        gaps = np.zeros((3, sequence_weights.N_COLUMNS), dtype=np.int8)
        for msa in (duplicate, gaps):
            counts = block_index.neighbor_counts_indexed(msa, blocks)
            np.testing.assert_array_equal(counts, sequence_weights.neighbor_counts(msa))

    def test_random_matrix_matches_reference_for_every_partition(self) -> None:
        generator = np.random.default_rng(211)
        msa = generator.integers(0, 21, size=(18, sequence_weights.N_COLUMNS), dtype=np.int8)
        entropy = block_index.column_entropy(msa)
        information = np.eye(sequence_weights.N_COLUMNS)
        partitions = [
            block_index.contiguous_partition(),
            block_index.coverage_sequential_partition(),
            block_index.coverage_entropy_spread_partition(entropy),
            block_index.coverage_information_partition(entropy, information),
            block_index.random_partition(7, one_low=False),
            block_index.random_partition(7, one_low=True),
        ]
        reference = sequence_weights.neighbor_counts(msa)
        for blocks in partitions:
            counts = block_index.neighbor_counts_indexed(msa, blocks)
            np.testing.assert_array_equal(counts, reference)


class RealSubsetTest(unittest.TestCase):
    def test_real_rows_match_blocked_reference(self) -> None:
        path = Path(__file__).resolve().parent / "mature_codes.npy"
        if not path.is_file():
            self.skipTest("mature matrix is not in prep/")
        msa = np.load(path, mmap_mode="r")
        matches = []
        query = np.asarray(msa[0])
        for start in range(1, msa.shape[0], 4096):
            chunk = np.asarray(msa[start : start + 4096])
            counts = np.count_nonzero(chunk == query, axis=1)
            chosen = np.flatnonzero(counts >= 211)
            matches.extend(start + int(offset) for offset in chosen[: 20 - len(matches)])
            if len(matches) >= 20:
                break
        rows = [0, *matches[:20], *range(1, 21)]
        sample = np.asarray(msa[rows])
        reference = sequence_weights.neighbor_counts(sample)
        entropy = block_index.column_entropy(sample)
        for blocks in (
            block_index.contiguous_partition(),
            block_index.coverage_sequential_partition(),
            block_index.coverage_entropy_spread_partition(entropy),
            block_index.random_partition(211, one_low=True),
        ):
            counts = block_index.neighbor_counts_indexed(sample, blocks)
            np.testing.assert_array_equal(counts, reference)


if __name__ == "__main__":
    unittest.main()

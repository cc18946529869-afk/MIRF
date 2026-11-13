"""Checks for the integer neighbor counts and the A2M column rules."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import msa_prepare
import sequence_weights


def reference_counts(msa: np.ndarray) -> np.ndarray:
    """Independent double loop. It is only for small tests."""
    n_rows = msa.shape[0]
    counts = np.ones(n_rows, dtype=np.int64)
    for left in range(n_rows):
        for right in range(left + 1, n_rows):
            matched = np.count_nonzero(msa[left] == msa[right])
            if matched >= sequence_weights.MIN_MATCHES:
                counts[left] += 1
                counts[right] += 1
    return counts


def row_with_matches(matched: int, other: int = 1) -> np.ndarray:
    row = np.zeros(sequence_weights.N_COLUMNS, dtype=np.int8)
    row[matched:] = other
    return row


class NeighborCountTest(unittest.TestCase):
    def test_211_matches_are_neighbors(self) -> None:
        msa = np.vstack(
            [
                np.zeros(sequence_weights.N_COLUMNS, dtype=np.int8),
                row_with_matches(211),
            ]
        )
        counts, _, _ = sequence_weights.sequence_weights(msa, block_rows=1)
        self.assertEqual(int(np.count_nonzero(msa[0] == msa[1])), 211)
        np.testing.assert_array_equal(counts, [2, 2])

    def test_210_matches_are_not_neighbors(self) -> None:
        msa = np.vstack(
            [
                np.zeros(sequence_weights.N_COLUMNS, dtype=np.int8),
                row_with_matches(210),
            ]
        )
        counts, _, _ = sequence_weights.sequence_weights(msa, block_rows=1)
        self.assertEqual(int(np.count_nonzero(msa[0] == msa[1])), 210)
        np.testing.assert_array_equal(counts, [1, 1])

    def test_self_is_counted(self) -> None:
        msa = np.zeros((1, sequence_weights.N_COLUMNS), dtype=np.int8)
        counts, weights, effective = sequence_weights.sequence_weights(msa)
        np.testing.assert_array_equal(counts, [1])
        self.assertEqual(weights.tolist(), [1.0])
        self.assertEqual(effective, 1.0)

    def test_duplicate_rows_stay_separate(self) -> None:
        row = row_with_matches(0)
        msa = np.vstack([row, row.copy()])
        counts, weights, effective = sequence_weights.sequence_weights(msa)
        np.testing.assert_array_equal(counts, [2, 2])
        np.testing.assert_allclose(weights, [0.5, 0.5])
        self.assertEqual(effective, 1.0)

    def test_all_gap_rows_keep_positive_weight(self) -> None:
        msa = np.zeros((3, sequence_weights.N_COLUMNS), dtype=np.int8)
        counts, weights, effective = sequence_weights.sequence_weights(msa)
        np.testing.assert_array_equal(counts, [3, 3, 3])
        np.testing.assert_allclose(weights, [1 / 3, 1 / 3, 1 / 3])
        self.assertAlmostEqual(effective, 1.0)

    def test_each_pair_is_added_once(self) -> None:
        msa = np.zeros((4, sequence_weights.N_COLUMNS), dtype=np.int8)
        counts = sequence_weights.neighbor_counts(msa, block_rows=3)
        np.testing.assert_array_equal(counts, reference_counts(msa))
        self.assertTrue(np.all(counts == 4))

    def test_blocked_counts_match_reference(self) -> None:
        rng = np.random.default_rng(211)
        msa = rng.integers(0, 21, size=(17, sequence_weights.N_COLUMNS), dtype=np.int8)
        expected = reference_counts(msa)
        for block_rows in (1, 2, 5, 16, 64):
            got = sequence_weights.neighbor_counts(msa, block_rows=block_rows)
            np.testing.assert_array_equal(got, expected)

    def test_neighbor_relation_is_symmetric_and_counts_differ(self) -> None:
        base = np.zeros(sequence_weights.N_COLUMNS, dtype=np.int8)
        close = row_with_matches(211)
        far = row_with_matches(0, other=2)
        msa = np.vstack([base, close, far])
        counts = sequence_weights.neighbor_counts(msa, block_rows=2)
        np.testing.assert_array_equal(counts, [2, 2, 1])

    def test_float_codes_are_rejected(self) -> None:
        msa = np.zeros((2, sequence_weights.N_COLUMNS), dtype=np.float64)
        with self.assertRaises(ValueError):
            sequence_weights.neighbor_counts(msa)

    def test_codes_outside_0_through_20_are_rejected(self) -> None:
        for bad in (-1, 21):
            msa = np.zeros((2, sequence_weights.N_COLUMNS), dtype=np.int8)
            msa[1, 0] = bad
            with self.assertRaises(ValueError):
                sequence_weights.neighbor_counts(msa)


class A2mColumnTest(unittest.TestCase):
    def test_lowercase_standard_residue_is_kept(self) -> None:
        text = ">q\nAxC\n>h\na.c\n"
        records = msa_prepare.read_a2m(text)
        encoded, nonstandard = msa_prepare.encode_fixed_columns(records)
        self.assertEqual(encoded.shape, (2, 3))
        np.testing.assert_array_equal(
            encoded[0],
            [
                msa_prepare.CODE["A"],
                msa_prepare.GAP_CODE,
                msa_prepare.CODE["C"],
            ],
        )
        np.testing.assert_array_equal(
            encoded[1],
            [
                msa_prepare.CODE["A"],
                msa_prepare.GAP_CODE,
                msa_prepare.CODE["C"],
            ],
        )
        self.assertEqual(nonstandard, {"x": 1})

    def test_dot_and_dash_are_gaps(self) -> None:
        text = ">q\nA.C\n>h\nA-C\n"
        records = msa_prepare.read_a2m(text)
        encoded, nonstandard = msa_prepare.encode_fixed_columns(records)
        np.testing.assert_array_equal(
            encoded,
            [
                [
                    msa_prepare.CODE["A"],
                    msa_prepare.GAP_CODE,
                    msa_prepare.CODE["C"],
                ],
                [
                    msa_prepare.CODE["A"],
                    msa_prepare.GAP_CODE,
                    msa_prepare.CODE["C"],
                ],
            ],
        )
        self.assertEqual(nonstandard, {})

    def test_uppercase_x_is_counted_and_column_kept(self) -> None:
        text = ">q\nA-C\n>h\nAXC\n"
        records = msa_prepare.read_a2m(text)
        encoded, nonstandard = msa_prepare.encode_fixed_columns(records)
        np.testing.assert_array_equal(
            encoded[1],
            [
                msa_prepare.CODE["A"],
                msa_prepare.GAP_CODE,
                msa_prepare.CODE["C"],
            ],
        )
        self.assertEqual(nonstandard, {"X": 1})

    def test_mature_crop_keeps_lowercase_target_positions(self) -> None:
        target = msa_prepare.TARGET_SEQUENCE
        query = target[:23].lower() + target[23:39].lower() + target[39:]
        homolog = "." * 23 + target[23:]
        text = (
            f">{msa_prepare.OFFICIAL_HEADER}\n{query}\n"
            f">homolog\n{homolog}\n"
        )
        mature, audit = msa_prepare.mature_alignment(text)
        self.assertEqual(mature.shape, (2, 263))
        self.assertEqual(audit.n_alignment_columns, 286)
        self.assertEqual(audit.n_lowercase_columns, 23 + 16)
        self.assertEqual(len(audit.mature_lowercase_positions), 16)
        self.assertEqual(audit.mature_lowercase_positions[0], 24)
        self.assertEqual(audit.mature_lowercase_positions[-1], 39)
        decoded = "".join(
            msa_prepare.AMINO_ACIDS[code - 1] for code in mature[0]
        )
        self.assertEqual(decoded, target[23:])
        self.assertEqual(decoded[0], "H")
        np.testing.assert_array_equal(audit.mature_gap_fraction, np.zeros(263))

    def test_query_must_match_target_after_case_folding(self) -> None:
        target = msa_prepare.TARGET_SEQUENCE
        query = "A" + target[1:]
        text = f">{msa_prepare.OFFICIAL_HEADER}\n{query}\n"
        with self.assertRaises(msa_prepare.MsaFormatError):
            msa_prepare.mature_alignment(text)

    def test_unequal_lengths_stop(self) -> None:
        with self.assertRaises(msa_prepare.MsaFormatError):
            msa_prepare.read_a2m(">q\nAC\n>h\nA.C\n")


class RealMsaTest(unittest.TestCase):
    def test_official_a2m_keeps_48_mature_low_coverage_sites(self) -> None:
        path = Path(__file__).resolve().parent / "BLAT_ECOLX_full_11-26-2021_b02.a2m"
        if not path.is_file():
            self.skipTest("official A2M is not in prep/")
        mature, audit = msa_prepare.mature_alignment(path.read_text())
        self.assertEqual(audit.n_sequences, 209644)
        self.assertEqual(audit.n_alignment_columns, 286)
        self.assertEqual(audit.n_uppercase_columns, 215)
        self.assertEqual(audit.n_lowercase_columns, 71)
        self.assertEqual(len(audit.mature_lowercase_positions), 48)
        self.assertEqual(mature.shape, (209644, 263))
        self.assertGreaterEqual(int(mature.min()), 0)
        self.assertLessEqual(int(mature.max()), 20)
        decoded = "".join(
            msa_prepare.AMINO_ACIDS[code - 1] for code in mature[0]
        )
        self.assertEqual(decoded, msa_prepare.TARGET_SEQUENCE[23:])
        self.assertEqual(audit.mature_gap_fraction.shape, (263,))
        self.assertEqual(
            audit.nonstandard_mapped_to_gap,
            {"X": 1676, "x": 209, "B": 2, "b": 1},
        )


if __name__ == "__main__":
    unittest.main()

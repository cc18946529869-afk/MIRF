"""Sequence identity for the saved soft-weight table and the residual batch."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

import feature_align
import residual


class FeatureAlignTests(unittest.TestCase):
    def test_membership_finds_both_training_files_in_the_train_split(self) -> None:
        index = feature_align.FeatureIndex(feature_align.table_path())
        n1000 = feature_align.membership(index, feature_align.N1000_TRAIN, "train")
        full = feature_align.membership(index, feature_align.FULL_TRAIN, "train")
        self.assertEqual(n1000["rows"], 1000)
        self.assertEqual(n1000["found_in_split"], 1000)
        self.assertEqual(full["rows"], 3497)
        self.assertEqual(full["found_in_split"], 3497)
        self.assertEqual(len(index.rows), 4996)
        report = feature_align.alignment_report(index)
        self.assertEqual(report["n1000_train"]["found_in_split"], 1000)
        self.assertEqual(report["full_train"]["found_in_split"], 3497)
        self.assertEqual(report["valid"]["found_in_split"], 500)
        self.assertEqual(report["test"]["found_in_split"], 999)
        self.assertEqual(report["split_rows"], {"train": 3497, "valid": 500, "test": 999})

    def test_shuffle_subset_and_repeated_queries_keep_each_row(self) -> None:
        index = feature_align.FeatureIndex(feature_align.table_path())
        sequences, scores = feature_align.read_scored_tsv(feature_align.N1000_TRAIN)
        generator = np.random.default_rng(0)
        order = generator.permutation(len(sequences))
        shuffled_sequences = [sequences[int(item)] for item in order]
        shuffled_scores = scores[order]
        aligned = feature_align.align_scored_sequences(index, shuffled_sequences, shuffled_scores)
        for row, sequence in enumerate(shuffled_sequences):
            values, mask = index.lookup([sequence])
            self.assertTrue(np.array_equal(aligned["values"][row], values[0]))
            self.assertTrue(np.array_equal(aligned["mask"][row], mask[0]))
            self.assertEqual(aligned["score"][row], shuffled_scores[row])
            self.assertEqual(aligned["sequence"][row], sequence)
        subset = shuffled_sequences[:17]
        subset_values, _subset_mask = index.lookup(subset)
        self.assertTrue(np.array_equal(subset_values, aligned["values"][:17]))
        repeated, repeated_mask = index.lookup([sequences[3], sequences[3]])
        self.assertTrue(np.array_equal(repeated[0], repeated[1]))
        self.assertTrue(np.array_equal(repeated_mask[0], repeated_mask[1]))
        original = feature_align.align_scored_sequences(index, sequences, scores)
        self.assertTrue(np.array_equal(original["score"], scores))
        full_sequences, full_scores = feature_align.read_scored_tsv(feature_align.FULL_TRAIN)
        full_order = generator.permutation(len(full_sequences))[:25]
        full_subset = [full_sequences[int(item)] for item in full_order]
        full_aligned = feature_align.align_scored_sequences(index, full_subset, full_scores[full_order])
        for row, sequence in enumerate(full_subset):
            values, mask = index.lookup([sequence])
            self.assertTrue(np.array_equal(full_aligned["values"][row], values[0]))
            self.assertTrue(np.array_equal(full_aligned["mask"][row], mask[0]))
            self.assertEqual(full_aligned["score"][row], full_scores[full_order][row])

    def test_a_missing_sequence_is_refused(self) -> None:
        index = feature_align.FeatureIndex(feature_align.table_path())
        with self.assertRaises(KeyError):
            index.lookup(["NOT-A-BLAT-SEQUENCE"])

    def test_a_false_mask_is_preserved(self) -> None:
        header = ["sequence", "split", "ref_pos", *feature_align.FEATURE_NAMES, *feature_align.MASK_NAMES]
        row = {"sequence": "ACDE", "split": "train", "ref_pos": "24"}
        for name in feature_align.FEATURE_NAMES:
            row[name] = "0.5"
        for name in feature_align.MASK_NAMES:
            row[name] = "1"
        row["mask_02"] = "0"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.tsv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=header, delimiter="\t")
                writer.writeheader()
                writer.writerow(row)
            index = feature_align.FeatureIndex(path)
            values, mask = index.lookup(["ACDE", "ACDE"])
        self.assertEqual(mask[0, 1], 0.0)
        self.assertEqual(mask[1, 1], 0.0)
        self.assertTrue(np.all(values[0, 1:] == 0.5))

    def test_real_values_reach_fit_predictor(self) -> None:
        index = feature_align.FeatureIndex(feature_align.table_path())
        sequences, scores = feature_align.read_scored_tsv(feature_align.N1000_TRAIN)
        aligned = feature_align.align_scored_sequences(index, sequences[:40], scores[:40])
        values = torch.as_tensor(aligned["values"], dtype=torch.float32)
        mask = torch.as_tensor(aligned["mask"], dtype=torch.float32)
        batch = {
            "site": torch.zeros(40, 4),
            "source": torch.zeros(40, 4),
            "other": torch.zeros(40, 4),
            "values": values,
            "mask": mask,
        }
        seen = []
        original = residual._epoch

        def spy(model, optimizer, received, rows, target, dropout_generator, mask_log):
            seen.append((
                rows.detach().cpu().clone(),
                received["values"].detach().cpu().clone(),
                received["mask"].detach().cpu().clone(),
            ))
            return original(model, optimizer, received, rows, target, dropout_generator, mask_log)

        residual._epoch = spy
        try:
            residual.fit_predictor(
                batch,
                torch.as_tensor(aligned["score"], dtype=torch.float32),
                np.arange(36),
                np.arange(36, 40),
                embed_dim=4,
                outer_seed=0,
            )
        finally:
            residual._epoch = original
        self.assertGreater(len(seen), 0)
        rows, received_values, received_mask = seen[0]
        self.assertTrue(torch.equal(received_values, values))
        self.assertTrue(torch.equal(received_mask, mask))
        self.assertTrue(torch.equal(received_values[rows], values[rows]))
        self.assertTrue(torch.equal(received_mask[rows], mask[rows]))

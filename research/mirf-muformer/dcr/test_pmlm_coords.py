"""Dictionary audit of every saved BLAT row. No hidden states are stored."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch

import feature_align
import frozen_pmlm
import pmlm_coords

CHECKPOINT = Path(__file__).resolve().parents[3] / "Mu-Protein" / "pretrained" / "pmlm_650m.pt"


class PmlmCoordinateTests(unittest.TestCase):
    def test_every_saved_row_has_its_own_site_token(self) -> None:
        index = feature_align.FeatureIndex(feature_align.table_path())
        with tempfile.TemporaryDirectory() as directory:
            payload = pmlm_coords.audit_table(index, Path(directory) / "pmlm_coords.json")
        self.assertTrue(payload["passed"])
        self.assertEqual(payload["checked_rows"], 4996)
        self.assertEqual(payload["source_lengths"], [112, 151])
        self.assertEqual(payload["distinct_source_index_sets"], 2)
        self.assertTrue(payload["mature_pos_equals_ref_pos_minus_23"])
        self.assertFalse(payload["source_and_other_shared_across_rows"])
        self.assertFalse(payload["embeddings_stored"])
        self.assertEqual(payload["failures"], [])

    def test_two_real_sequences_use_their_own_token_indices(self) -> None:
        index = feature_align.FeatureIndex(feature_align.table_path())
        chosen = []
        seen = set()
        for sequence, row in index.rows.items():
            if row["split"] != "train":
                continue
            source, _other = pmlm_coords.per_row_lobe_indices(pmlm_coords.DOMAIN_MAP, row["ref_pos"])
            length = int(source.shape[0])
            if length in seen:
                continue
            chosen.append(sequence)
            seen.add(length)
            if len(chosen) == 2:
                break
        self.assertEqual(len(chosen), 2)
        self.assertTrue(CHECKPOINT.is_file())
        model = frozen_pmlm.load_independent_pretrained(str(CHECKPOINT))
        self.assertFalse(any(bool(parameter.requires_grad) for parameter in model.parameters()))
        self.assertFalse(model.training)
        dictionary = pmlm_coords.load_dictionary()
        tokens = torch.stack([pmlm_coords.encode_sequence(dictionary, sequence) for sequence in chosen])
        site_index = []
        source_index = []
        other_index = []
        for sequence in chosen:
            ref_pos = index.row(sequence)["ref_pos"]
            site_index.append(pmlm_coords.token_index_for_reference(dictionary, sequence, ref_pos))
            source, other = pmlm_coords.per_row_lobe_indices(pmlm_coords.DOMAIN_MAP, ref_pos)
            source_index.append(torch.as_tensor(source, dtype=torch.long))
            other_index.append(torch.as_tensor(other, dtype=torch.long))
        self.assertNotEqual(int(source_index[0].numel()), int(source_index[1].numel()))
        site, source, other = model.representations(
            tokens,
            torch.tensor(site_index),
            source_index,
            other_index,
        )
        hidden = model(tokens)
        self.assertEqual(tuple(hidden.shape), (int(tokens.shape[1]), 2, 1280))
        self.assertEqual(tuple(site.shape), (2, 1280))
        self.assertTrue(torch.equal(site[0], hidden[site_index[0], 0]))
        self.assertTrue(torch.equal(site[1], hidden[site_index[1], 1]))
        self.assertTrue(torch.allclose(source[0], hidden[source_index[0], 0].mean(dim=0)))
        self.assertTrue(torch.allclose(source[1], hidden[source_index[1], 1].mean(dim=0)))
        self.assertTrue(torch.allclose(other[0], hidden[other_index[0], 0].mean(dim=0)))
        self.assertTrue(torch.allclose(other[1], hidden[other_index[1], 1].mean(dim=0)))
        self.assertFalse(hasattr(model, "fc_reduce"))

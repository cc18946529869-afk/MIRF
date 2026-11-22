"""The synthetic rehearsal stays off the formal ten-seed run."""

from __future__ import annotations

import unittest
from pathlib import Path

import rehearsal


class RehearsalTests(unittest.TestCase):
    def test_synthetic_rehearsal_refuses_a_formal_mode(self) -> None:
        precheck = Path(__file__).resolve().parent / "precheck"
        with self.assertRaises(RuntimeError):
            rehearsal.rehearse(precheck, mode="formal")
        result = rehearsal.rehearse(precheck, mode="synthetic")
        self.assertFalse(result["formal_training_started"])
        self.assertFalse(result["formal_metrics_computed"])
        self.assertTrue(result["ten_seed_entry_refused"])
        self.assertEqual(result["fold_calls"], 5)
        self.assertEqual(result["k_for_999"], 100)
        self.assertEqual(result["formal_bootstrap_default"], 10000)
        self.assertTrue(result["shared_initial_state"])
        self.assertTrue(result["external_sum"])
        self.assertTrue(result["synthetic_spearman_passes_on_constructed_arrays"])
        self.assertEqual(result["sealed_residual_calls"], 3)
        self.assertEqual(result["sealed_patience"], 30)
        self.assertEqual(result["sealed_batch_size"], 64)
        self.assertEqual(result["sealed_max_epochs"], 300)
        self.assertEqual(result["muformer_batch_size"], 16)
        self.assertEqual(result["residual_batch_size"], 64)
        self.assertNotEqual(result["muformer_batch_size"], result["residual_batch_size"])
        self.assertEqual(result["muformer_hidden_size"], 256)
        self.assertEqual(result["muformer_dropout"], 0.2)
        self.assertEqual(result["muformer_decoder_name"], "siamese")
        self.assertEqual(result["muformer_encoder_lr"], 1e-5)
        self.assertEqual(result["muformer_decoder_lr"], 1e-4)
        self.assertEqual(result["muformer_weight_decay"], 0)
        self.assertEqual(result["muformer_warmup_epochs"], 10)
        self.assertNotEqual(result["e_s"], 229)


if __name__ == "__main__":
    unittest.main()

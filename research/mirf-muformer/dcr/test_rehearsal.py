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


if __name__ == "__main__":
    unittest.main()

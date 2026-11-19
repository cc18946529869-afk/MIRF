"""Checks for the contact-ranking arithmetic, not for the 1BTL result."""

from __future__ import annotations

import unittest

import numpy as np

import contact_validation


class ContactValidationTests(unittest.TestCase):
    def test_amino_block_norm_drops_the_gap_state(self) -> None:
        couplings = np.zeros((2, 2, 21, 21), dtype=np.float64)
        couplings[0, 1, 0, 0] = 10.0
        couplings[0, 1, 1, 2] = 3.0
        couplings[0, 1, 2, 1] = 4.0
        norms = contact_validation.amino_block_norm(couplings)
        self.assertAlmostEqual(norms[0, 1], 5.0)

    def test_average_product_correction_removes_a_constant_background(self) -> None:
        scores = np.full((4, 4), 2.0)
        np.fill_diagonal(scores, 0.0)
        corrected = contact_validation.average_product_correction(scores)
        self.assertTrue(np.allclose(corrected, 0.0, atol=1e-8))

    def test_precision_reads_the_ranked_prefix(self) -> None:
        contact = np.array([1, 0, 1, 0], dtype=bool)
        score = np.array([4.0, 3.0, 2.0, 1.0])
        metrics = contact_validation.ranking_metrics(contact, score, length=4)
        self.assertEqual(metrics["precision_at_l"], 0.5)
        self.assertEqual(metrics["precision_at_l_half"], 0.5)
        self.assertAlmostEqual(metrics["average_precision"], (1.0 + 2.0 / 3.0) / 2.0)


if __name__ == "__main__":
    unittest.main()

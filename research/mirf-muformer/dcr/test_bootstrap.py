"""Checks for bootstrap sampling, gauge assembly, and the R/D decision."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bootstrap_fit
import bootstrap_gauge
import bootstrap_stats
import gauge
import potts_fit


class GaugeAssemblyTests(unittest.TestCase):
    def test_packed_partners_round_trip_in_column_order(self):
        length, q = 4, 3
        generator = np.random.default_rng(3)
        directional = generator.normal(size=(length, length, q, q))
        for site in range(length):
            directional[site, site] = 0.0
        fields = generator.normal(size=(length, q))
        packed_fields = []
        packed_couplings = []
        for site in range(length):
            partners = [partner for partner in range(length) if partner != site]
            block = np.zeros((q, length - 1, q), dtype=np.float64)
            for slot, partner in enumerate(partners):
                self.assertEqual(bootstrap_gauge.partner_index(site, partner), slot)
                block[:, slot, :] = directional[site, partner]
            packed_fields.append(fields[site])
            packed_couplings.append(block)
        restored_fields, restored = bootstrap_gauge.assemble_directional(packed_fields, packed_couplings)
        np.testing.assert_allclose(restored_fields, fields)
        np.testing.assert_allclose(restored, directional)

    def test_recovered_field_uses_the_pretransform_couplings(self):
        directional = np.zeros((3, 3, 2, 2), dtype=np.float64)
        directional[0, 1, 0, 0] = 4.0
        fields = np.zeros((3, 2), dtype=np.float64)
        recovered, averaged = bootstrap_gauge.gauge_parameters(fields, directional)
        direct = gauge.recover_fields(fields, directional)
        np.testing.assert_allclose(recovered, direct)
        self.assertNotAlmostEqual(float(recovered[0, 0]), 0.0)
        fed_back = gauge.recover_fields(fields, averaged)
        self.assertGreater(abs(float(recovered[0, 0] - fed_back[0, 0])), 0.1)

    def test_averaged_couplings_keep_both_triangles(self):
        generator = np.random.default_rng(5)
        directional = generator.normal(size=(5, 5, 2, 2))
        for site in range(5):
            directional[site, site] = 0.0
        averaged = gauge.average_couplings(directional)
        transposed = np.transpose(averaged, (1, 0, 3, 2))
        np.testing.assert_allclose(averaged, transposed)
        upper = np.sum(averaged[np.triu_indices(5, k=1)] ** 2)
        self.assertAlmostEqual(float(np.sum(averaged ** 2)), float(2.0 * upper))


class StatisticTests(unittest.TestCase):
    def test_zero_deviation_and_zero_reference_are_distinct(self):
        samples = np.ones((6, 2), dtype=np.float64)
        self.assertEqual(bootstrap_stats.relative_frobenius(samples, np.array([2.0, 0.0])), 0.0)
        self.assertIsNone(bootstrap_stats.relative_frobenius(samples, np.zeros(2)))
        self.assertEqual(bootstrap_stats.relative_change(0.0, 0.0), 0.0)

    def test_half_range_is_the_preassigned_ids(self):
        success = [False] * 10 + [True] * 90
        fields = {identifier: np.array([1.0, 2.0]) for identifier in range(1, 101) if success[identifier - 1]}
        couplings = {identifier: np.zeros((2, 2, 1, 1)) for identifier in fields}
        couplings_reference = np.zeros((2, 2, 1, 1))
        couplings_reference[0, 1, 0, 0] = 1.0
        couplings_reference[1, 0, 0, 0] = 1.0
        outcome = bootstrap_stats.evaluate_stage(
            success,
            fields,
            couplings,
            np.array([1.0, 2.0]),
            couplings_reference,
            100,
        )
        self.assertEqual(outcome["success_count"], 90)
        self.assertEqual(outcome["action"], "pass")
        self.assertEqual(outcome["d_h"], 0.0)

    def test_too_few_successes_do_not_compute_d(self):
        success = [True] * 89 + [False] * 11
        outcome = bootstrap_stats.evaluate_stage(success, {}, {}, np.ones(2), np.ones((1, 1)), 100)
        self.assertEqual(outcome["action"], "extend")
        self.assertIsNone(outcome["r_h"])
        self.assertIsNone(outcome["d_h"])

    def test_stable_deviation_with_large_parameter_spread_does_not_add_replicates(self):
        success = [True] * 100
        fields = {}
        for identifier in range(1, 101):
            fields[identifier] = np.array([0.0, 0.0] if identifier % 2 else [10.0, 0.0])
        couplings = {identifier: np.ones((1, 1)) for identifier in fields}
        outcome = bootstrap_stats.evaluate_stage(success, fields, couplings, np.array([1.0, 0.0]), np.ones((1, 1)), 100)
        self.assertLessEqual(outcome["d_h"], 0.10)
        self.assertGreater(outcome["r_h"], 0.10)
        self.assertEqual(outcome["action"], "parameter_unstable")
        self.assertFalse(outcome["passed"])

    def test_two_hundred_without_enough_successes_has_no_parameter_verdict(self):
        success = [True] * 179 + [False] * 21
        outcome = bootstrap_stats.evaluate_stage(success, {}, {}, np.ones(2), np.ones((1, 1)), 200)
        self.assertEqual(outcome["action"], "no_conclusion")
        self.assertIsNone(outcome["parameter_stable"])


class SamplingTests(unittest.TestCase):
    def test_duplicate_draws_remain_separate_rows(self):
        alignment = np.arange(12, dtype=np.int8).reshape(4, 3)
        sample = bootstrap_fit.replicate_sample(alignment, np.array([0, 0, 2]))
        self.assertEqual(sample.shape, (3, 3))
        np.testing.assert_array_equal(sample[0], sample[1])
        self.assertFalse(np.array_equal(sample[1], sample[2]))

    def test_official_probabilities_match_the_weight_total(self):
        _alignment, weights, total = potts_fit.load_official_alignment()
        self.assertEqual(total, potts_fit.OFFICIAL_M_EFF)
        generator = np.random.default_rng(11)
        indices = bootstrap_fit.draw_indices(generator, weights, total, count=1000)
        self.assertEqual(indices.shape, (1000,))
        self.assertTrue(np.all((indices >= 0) & (indices < weights.shape[0])))

    def test_formal_replicate_streams_are_independent_of_order(self):
        first = bootstrap_fit.formal_seed_sequence(19, 1).integers(0, 1000, size=4)
        again = bootstrap_fit.formal_seed_sequence(19, 1).integers(0, 1000, size=4)
        later = bootstrap_fit.formal_seed_sequence(19, 101).integers(0, 1000, size=4)
        np.testing.assert_array_equal(first, again)
        self.assertFalse(np.array_equal(first, later))

    def test_timing_seed_is_rejected_for_verification(self):
        with self.assertRaises(ValueError):
            bootstrap_fit.run_verification(20261114)


if __name__ == "__main__":
    unittest.main()

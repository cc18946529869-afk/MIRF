"""Regression tests for the continuation safeguards and discrepancy selection."""
from decimal import Decimal, localcontext
import math
import unittest

import numpy as np

from .check_f81_edge_conditioning import max_gradient_disagreement
from .check_f81_full_tree_iteration import both_gates, plateau, recurrent_period, transfer_summary


class FullTreeContinuationTests(unittest.TestCase):
    def test_plateau_has_window_and_minimum_progress(self):
        self.assertFalse(plateau([100.]*8))
        self.assertTrue(plateau([100.]*8+[99.5]))
        self.assertFalse(plateau([100.]*8+[98.5]))

    def test_cycle_requires_three_ordered_repetitions(self):
        self.assertEqual(recurrent_period([42, 2, 7, 2, 7, 2, 7]), [2, 7])
        self.assertEqual(recurrent_period([5, 5, 5]), [5])
        self.assertIsNone(recurrent_period([2, 7, 2, 7, 7, 2]))

    def test_both_projection_paths_and_absolute_agreement(self):
        fields = {"projected_gradient_inf": 1e-5, "rerooted_projected_gradient_inf": 1e-5,
                  "cross_path_gradient_passed": True}
        self.assertTrue(both_gates(fields))
        self.assertFalse(both_gates({**fields, "rerooted_projected_gradient_inf": 1.1e-5}))
        self.assertFalse(both_gates({**fields, "cross_path_gradient_passed": False}))

    def test_disagreement_location_raw_values_and_ulp(self):
        a, b = 5e7, math.nextafter(5e7, math.inf)
        with localcontext() as ctx:
            ctx.prec = 115
            row = max_gradient_disagreement(np.array([0., a]), np.array([0., b]))
        self.assertEqual(row["edge"], 1)
        self.assertEqual((row["ordinary"], row["rerooted_log"]), (a, b))
        self.assertEqual(row["absolute_difference"], b-a)
        self.assertGreater(Decimal(row["symmetric_relative_difference"]), 0)
        self.assertEqual(row["ulp_distance"], 1)

    def test_rejected_trial_records_residual_transferred_to_held_edge(self):
        point, trial = np.array([0., 1., 1.]), np.array([.1, 1., 1.])
        before = {"vectors": [None, np.array([-3., -2., .5])]}
        after = {"vectors": [None, np.array([0., -4., .5])]}
        result = transfer_summary(point, trial, 0, before, after)
        self.assertEqual(result["chosen_abs_before"], 3.)
        self.assertEqual(result["chosen_abs_after"], 0.)
        self.assertEqual(result["other_edges_increased"], 1)
        self.assertEqual(result["largest_increases"], [{"edge": 1, "before": 2., "after": 4., "increase": 2.}])


if __name__ == "__main__":
    unittest.main()

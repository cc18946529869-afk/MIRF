"""Regression tests for charged expansion and history recovery decisions."""
from types import SimpleNamespace
import unittest

import numpy as np

from .f81_optimize import (
    expansion_steps,
    normalized_negative_gradient,
    optimize_projected,
    stale_direction,
)
from .f81_unrooted import UnrootedPruning


class DisconnectedAcceptanceFixture(UnrootedPruning):
    """Scalar oracle with an Armijo pass on either side of a rejection."""

    def __init__(self):
        self.topology = SimpleNamespace(child_to_edge=np.array([0]))
        self.pruning = SimpleNamespace(states=[0])
        self.probes = []

    def evaluate(self, lengths, **kwargs):
        self.probes.append(float(lengths[0]))
        alpha = round((float(lengths[0]) - 1.) / 1e-6)
        values = {0: 10., 1: 9.99, 4: 11., 128: 9.}
        return SimpleNamespace(log_likelihood=-values[alpha],
                               length_gradient=np.array([1e-6]), log_domain_sites=0)


class OptimizerAlternativesTests(unittest.TestCase):
    def test_expansion_respects_conflict_boundary(self):
        self.assertEqual(list(expansion_steps(1., 10.)), [4., 9.5])
        self.assertEqual(list(expansion_steps(1., 1.)), [])
        self.assertEqual(list(expansion_steps(1., np.inf)), [4., 128., 8192., 1048576., 67108864.])

    def test_counts_rejection_and_keeps_expanded_incumbent_at_budget(self):
        for budget, winning_call, expected_value in ((3, 2, 9.99), (4, 4, 9.)):
            with self.subTest(budget=budget):
                engine = DisconnectedAcceptanceFixture()
                events = []
                point, fit = optimize_projected(engine, [1.], emit=events.append, maxeval=budget,
                                                gtol=1e-10, expand_steps=True, search_diagnostics=True)
                self.assertEqual(fit["solver_likelihood_gradient_evaluations"], budget)
                self.assertEqual(fit["expansion_evaluations"], budget - 2)
                self.assertEqual(fit["expansion_selections"], int(budget == 4))
                self.assertEqual(fit["status"], "hard_limit_on_likelihood_gradient_evaluations")
                self.assertEqual(fit["terminal_log_likelihood"], -expected_value)
                self.assertEqual([e["calls"] for e in events if e["event"] == "accepted"], [winning_call])
                self.assertEqual(len(engine.probes), budget + 1)
                self.assertEqual(point[0], engine.probes[winning_call - 1])
                self.assertIn("armijo_rejected", [e.get("outcome") for e in events])

    def test_history_recovery_requires_all_three_observed_conditions(self):
        self.assertTrue(stale_direction(8, 1e-42, 2e-14))
        self.assertFalse(stale_direction(7, 1e-42, 2e-14))
        self.assertFalse(stale_direction(8, 1e-9, 2e-14))
        self.assertFalse(stale_direction(8, 1e-42, .2))
        self.assertFalse(stale_direction(8, None, 2e-14))
        np.testing.assert_array_equal(normalized_negative_gradient(np.array([1e30, 0., -.5, -2.])),
                                      [-1., 0., .5, 1.])


if __name__ == "__main__":
    unittest.main()

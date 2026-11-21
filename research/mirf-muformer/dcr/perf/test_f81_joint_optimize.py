"""Focused tests for relative trust, mixed coordinates, and joint acceptance."""
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from .f81_edge_conditioning import condition
from .f81_joint_optimize import chart, curvature_pair, direction, solve


class JointTests(unittest.TestCase):
    def test_tiny_log_gradient_is_not_suppressed_by_absolute_floor(self):
        with localcontext() as ctx:
            ctx.prec = 80
            _, step = condition(Decimal("1.147e-24"), Decimal("-251520"), Decimal("1.664e13"),
                                "hybrid_log_diag")
        self.assertGreater(step, Decimal(".99"))
        self.assertLessEqual(step, Decimal(4))

    def test_mixed_diagonal_matches_centered_derivative(self):
        point = np.array([.13, 0., 0.])
        g = np.array([point[0]-1, -.3, 2.])
        positive, active, _, diag = chart(point, g, np.ones(3))
        def log_score(z):
            s = np.exp(z)
            return s*(s-1)
        z = np.log(point[0])
        h = 1e-6
        self.assertAlmostEqual(diag[0], (log_score(z+h)-log_score(z-h))/(2*h), places=9)
        np.testing.assert_array_equal(active, [False, False, True])
        np.testing.assert_array_equal(positive, [True, False, False])
        np.testing.assert_array_equal(diag[1:], [1., 1.])

    def test_secants_are_used_on_stable_coordinates(self):
        gradient, diagonal = np.array([2., 1.]), np.array([1., 1.])
        pair = curvature_pair(np.array([1., .25]), np.array([2., 1.]))
        self.assertIsNotNone(pair)
        self.assertFalse(np.array_equal(direction(gradient, diagonal, 4., [pair]),
                                        direction(gradient, diagonal, 4., [])))

    def test_joint_solver_preserves_zero_and_charges_trials(self):
        class Quadratic:
            def evaluate(self, s, **kwargs):
                displacement = s-np.array([.2, -.4, .7])
                loss = .5 * float(displacement @ displacement)
                return SimpleNamespace(log_likelihood=-loss, site_log_likelihood=np.array([-loss]),
                                       length_gradient=-displacement, negative_log_hessian_diagonal=np.ones(3))
        q = Quadratic()
        guard = SimpleNamespace(feasible=lambda p: bool(np.all(p >= 0)))
        events = []
        p, result = solve(q, q, guard, np.array([0., .03, .3]), events.append, maxeval=300)
        self.assertTrue(result["both_gates_passed"])
        self.assertEqual(p[1], 0.)
        self.assertLessEqual(max(abs(p[0]-.2), abs(p[2]-.7)), 1e-5)
        trials = [e for e in events if e["event"] == "trial"]
        self.assertEqual(len(trials), result["counts"]["line_search_trials"])
        accepted = [e for e in trials if e["status"] == "accepted"]
        self.assertTrue(all(e["objective_change"] <= 1e-4*e["slope"] < 0 for e in accepted))
        self.assertTrue(any(e["changed_edges"] > 1 for e in events if e["event"] == "iterate"))

    def test_archive_keeps_gradient_state_apart_from_objective_state(self):
        class Tilted:
            def evaluate(self, s, **kwargs):
                target = np.array([.2, -.4, .7])
                loss = .5 * float((s-target) @ (s-target)) + .05 * float(s[0])
                gradient = (s-target) + np.array([.05, 0., 0.])
                return SimpleNamespace(log_likelihood=-loss, site_log_likelihood=np.array([-loss]),
                                       length_gradient=-gradient, negative_log_hessian_diagonal=np.ones(3))
        q = Tilted()
        guard = SimpleNamespace(feasible=lambda p: bool(np.all(p >= 0)))
        with tempfile.TemporaryDirectory() as raw:
            directory = Path(raw)
            _, result = solve(q, q, guard, np.array([0., .03, .3]), lambda _row: None,
                              maxeval=80, archive=directory)
            names = ("initial", "best_projection", "best_objective", "final")
            self.assertEqual(set(result["checkpoints"]), set(names))
            for name in names:
                blob = directory / f"{name}.npz"
                meta = json.loads((directory / f"{name}.json").read_text())
                self.assertEqual(meta["sha256"], hashlib.sha256(blob.read_bytes()).hexdigest())
                saved = np.load(blob)
                self.assertEqual(saved["lengths"].shape, (3,))
                self.assertEqual(meta["lengths_sha256"], hashlib.sha256(np.ascontiguousarray(saved["lengths"]).tobytes()).hexdigest())
            best = np.load(directory / "best_projection.npz")
            final = np.load(directory / "final.npz")
            self.assertLessEqual(float(best["projection"]), float(final["projection"]))

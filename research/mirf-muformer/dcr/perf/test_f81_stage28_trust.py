"""Quadratic checks for the residual-tail trust step.

The coupled pair uses the same 2×2 scale as the Stage 2.7 terminal block:
off-diagonal curvature as large as the diagonal, so the undamped Newton step
is the infeasible near-null direction. The negative-curvature edge checks that
the step stays inside the length cap.
"""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .f81_stage28_trust import (
    free_edges, hessian_product, incident_edges, solve_subspace_trust,
)


class Quadratic:
    def __init__(self, hessian, origin, gradient, objective):
        self.hessian = np.array(hessian, dtype=float)
        self.origin = np.array(origin, dtype=float)
        self.gradient = np.array(gradient, dtype=float)
        self.objective = float(objective)

    def at(self, lengths):
        displacement = np.asarray(lengths, dtype=float) - self.origin
        value = self.objective + float(self.gradient @ displacement)
        value += 0.5 * float(displacement @ (self.hessian @ displacement))
        slope = self.gradient + self.hessian @ displacement
        return value, slope

    def evaluate(self, lengths, *, threads=1, gradient=True, diagonal=False, log_domain=False):
        del threads, log_domain
        value, slope = self.at(lengths)
        return SimpleNamespace(
            log_likelihood=-value,
            site_log_likelihood=np.array([-value]),
            length_gradient=None if not gradient else -slope,
            negative_log_hessian_diagonal=None if not diagonal else np.diag(self.hessian).copy(),
        )


class Guard:
    def feasible(self, lengths):
        return bool(np.all(np.asarray(lengths) >= 0))


class TrustTests(unittest.TestCase):
    def test_product_matches_the_quadratic_hessian(self):
        hessian = np.array([[3., 2.5], [2.5, 3.]])
        origin = np.array([0.03, 0.06])
        engine = Quadratic(hessian, origin, [0.01, 0.01], 1.)

        def gradient_at(lengths):
            _value, slope = engine.at(lengths)
            return slope

        direction = np.array([0.2, -0.1])
        product = hessian_product(origin, direction, gradient_at)
        np.testing.assert_allclose(product, hessian @ direction, rtol=1e-8, atol=1e-10)

    def test_coupled_pair_reduces_the_infinity_norm_without_moving_the_bound(self):
        hessian = np.diag([3.00876, 3.0224, 1e-5, 1.])
        hessian[0, 1] = hessian[1, 0] = 3.014421
        origin = np.array([0.033411, 0.067652, 1., 0.])
        # Edge 2 is adjacent and has a tiny diagonal, so its raw diagonal step
        # is far outside its length cap. That edge must not shrink the pair.
        gradient = np.array([0.013947, 0.011585, 5e-4, 0.04])
        engine = Quadratic(hessian, origin, gradient, 10.)
        pairs = np.array([[1, 0], [2, 0], [1, 6], [5, 1]])
        adjacency = incident_edges(pairs)
        index, _threshold, seeds = free_edges(origin, gradient, adjacency)
        np.testing.assert_array_equal(index, [0, 1, 2])
        self.assertEqual(seeds, 2)
        with tempfile.TemporaryDirectory() as directory:
            _point, result = solve_subspace_trust(
                engine, engine, Guard(), origin, lambda row: None, adjacency,
                archive=directory, maxiter=3, maxeval=200, wall_seconds=30, max_cg=4)
            self.assertTrue(result["hvp_verification"]["accepted"])
            self.assertLess(result["best_ordinary_projection"], 2e-3)
            self.assertLess(result["objective"], 10.)
            self.assertEqual(result["zero_edges"], 1)
            self.assertLessEqual(abs(_point[2] - 1.), 0.03 + 1e-9)
            self.assertGreater(abs(_point[0] - origin[0]), 1e-3)
            self.assertEqual(_point[3], 0.)
            self.assertGreater(result["accepted"], 0)
            names = {path.name for path in Path(directory).glob("*.npz")}
            self.assertTrue({"initial.npz", "best_projection.npz", "best_objective.npz", "final.npz"} <= names)

    def test_armijo_step_is_kept_when_the_maximum_residual_moves(self):
        hessian = np.array([[2., 3.], [3., 2.]])
        origin = np.array([0.2, 0.2])
        gradient = np.array([0.01, -0.004])
        engine = Quadratic(hessian, origin, gradient, 1.)
        adjacency = incident_edges(np.array([[1, 0], [2, 0]]))
        point, result = solve_subspace_trust(
            engine, engine, Guard(), origin, lambda row: None, adjacency,
            maxiter=1, maxeval=40, wall_seconds=30, max_cg=4)
        self.assertGreater(result["accepted"], 0)
        self.assertLess(result["objective"], 1.)
        self.assertGreater(result["projection"], 0.01)

    def test_negative_curvature_does_not_take_the_undamped_step(self):
        hessian = np.array([[-1.]])
        origin = np.array([1.])
        engine = Quadratic(hessian, origin, [0.2], 1.)
        adjacency = incident_edges(np.array([[1, 0]]))
        point, result = solve_subspace_trust(
            engine, engine, Guard(), origin, lambda row: None, adjacency,
            maxiter=20, maxeval=400, wall_seconds=30, max_cg=4)
        self.assertEqual(result["status"], "trust_collapsed")
        self.assertEqual(result["accepted"], 0)
        np.testing.assert_allclose(point, origin)
        # The undamped stationary point of this negative curvature is a move of 0.2.
        self.assertLess(abs(point[0] - origin[0]), 0.01)


if __name__ == "__main__":
    unittest.main()

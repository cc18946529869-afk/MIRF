"""Independent checks for the Potts conditional objective, solver, and gauge."""

from __future__ import annotations

import sys
import math
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import energy_components
import gauge
import potts_energy
import potts_fit


def reference_site_terms(theta, msa, weights, site, weight_total, lambda_h, lambda_j_prime, q):
    """Separate loop implementation. It is only used on small alignments."""
    n_rows, length = msa.shape
    fields = theta[:q].astype(np.float64).copy()
    couplings = theta[q:].reshape(q, length - 1, q).astype(np.float64).copy()
    partners = [column for column in range(length) if column != site]
    nll = 0.0
    grad_h = np.zeros(q, dtype=np.float64)
    grad_j = np.zeros_like(couplings)
    for row in range(n_rows):
        logits = fields.copy()
        for partner, column in enumerate(partners):
            logits += couplings[:, partner, int(msa[row, column])]
        maximum = float(logits.max())
        exponential = np.exp(logits - maximum)
        normalizer = float(exponential.sum())
        observed = int(msa[row, site])
        nll -= float(weights[row]) * (float(logits[observed]) - (maximum + np.log(normalizer)))
        probability = exponential / normalizer
        for state in range(q):
            signed = float(weights[row]) * (float(probability[state]) - (1.0 if state == observed else 0.0))
            grad_h[state] += signed
            for partner, column in enumerate(partners):
                grad_j[state, partner, int(msa[row, column])] += signed
    penalty = float(weight_total) * (
        float(lambda_h) * float(np.dot(fields, fields))
        + float(lambda_j_prime) * float(np.dot(couplings.ravel(), couplings.ravel()))
    )
    grad_h += 2.0 * float(weight_total) * float(lambda_h) * fields
    grad_j += 2.0 * float(weight_total) * float(lambda_j_prime) * couplings
    return nll, penalty, np.concatenate((grad_h, grad_j.ravel()))


class ObjectiveTests(unittest.TestCase):
    def test_hand_calculated_terms_include_the_factor_two(self):
        q = 2
        msa = np.array([[0, 1]], dtype=np.int8)
        weights = np.array([2.0])
        fields = np.array([0.0, np.log(3.0)])
        couplings = np.zeros((q, 1, q))
        theta = np.concatenate((fields, couplings.ravel()))
        weight_total = 5.0
        expected_nll = -2.0 * np.log(0.25)
        expected_penalty = weight_total * 0.01 * float(np.dot(fields, fields))
        expected_grad = np.array(
            [
                -1.5,
                1.5 + 2.0 * weight_total * 0.01 * np.log(3.0),
                0.0,
                -1.5,
                0.0,
                1.5,
            ]
        )
        for evaluator in (potts_fit.site_terms_numpy, potts_fit.site_terms_c, reference_site_terms):
            nll, penalty, gradient = evaluator(
                theta, msa, weights, 0, weight_total, 0.01, 0.005, q
            )
            self.assertAlmostEqual(nll, expected_nll)
            self.assertAlmostEqual(penalty, expected_penalty)
            np.testing.assert_allclose(gradient, expected_grad)

    def test_weight_total_scales_the_penalty_once(self):
        rng = np.random.default_rng(7)
        q = 4
        msa = rng.integers(0, q, size=(5, 3), dtype=np.int8)
        weights = np.array([0.5, 1.0, 2.0, 0.25, 4.0])
        theta = rng.normal(size=potts_fit.parameter_count(3, q))
        nll_a, penalty_a, grad_a = potts_fit.site_terms_numpy(
            theta, msa, weights, 1, 3.5, 0.01, 0.005, q
        )
        nll_b, penalty_b, grad_b = potts_fit.site_terms_c(
            theta, msa, weights, 1, 7.0, 0.01, 0.005, q
        )
        nll_0, penalty_0, grad_0 = reference_site_terms(
            theta, msa, weights, 1, 0.0, 0.01, 0.005, q
        )
        self.assertAlmostEqual(nll_a, nll_b)
        self.assertAlmostEqual(nll_a, nll_0)
        self.assertEqual(penalty_0, 0.0)
        self.assertAlmostEqual(penalty_b, 2.0 * penalty_a)
        fields, couplings = potts_fit.unpack_parameters(theta, 3, q)
        self.assertAlmostEqual(
            penalty_a,
            3.5 * (0.01 * float(np.dot(fields, fields)) + 0.005 * float(np.dot(couplings.ravel(), couplings.ravel()))),
        )
        np.testing.assert_allclose(grad_b - grad_0, 2.0 * (grad_a - grad_0))

    def test_numpy_and_c_match_the_independent_loop(self):
        rng = np.random.default_rng(11)
        q = 5
        length = 4
        msa = rng.integers(0, q, size=(6, length), dtype=np.int8)
        msa[0, 2] = 0
        weights = rng.uniform(0.1, 2.0, size=6)
        theta = rng.normal(scale=0.2, size=potts_fit.parameter_count(length, q))
        for site in range(length):
            reference = reference_site_terms(theta, msa, weights, site, 19.0, 0.01, 0.005, q)
            numpy_terms = potts_fit.site_terms_numpy(theta, msa, weights, site, 19.0, 0.01, 0.005, q)
            c_terms = potts_fit.site_terms_c(theta, msa, weights, site, 19.0, 0.01, 0.005, q)
            for terms in (numpy_terms, c_terms):
                self.assertAlmostEqual(terms[0], reference[0])
                self.assertAlmostEqual(terms[1], reference[1])
                np.testing.assert_allclose(terms[2], reference[2], atol=1e-10)

    def test_analytic_gradient_matches_finite_differences(self):
        rng = np.random.default_rng(13)
        q = 3
        length = 3
        msa = rng.integers(0, q, size=(4, length), dtype=np.int8)
        weights = np.array([1.0, 0.5, 1.5, 0.25])
        theta = rng.normal(scale=0.15, size=potts_fit.parameter_count(length, q))
        weight_total = potts_fit.OFFICIAL_M_EFF
        _nll, _penalty, gradient = potts_fit.site_terms(
            theta, msa, weights, 2, weight_total, potts_fit.LAMBDA_H, potts_fit.LAMBDA_J_PRIME, q
        )

        def value(point):
            nll, penalty, _gradient = reference_site_terms(
                point, msa, weights, 2, weight_total, potts_fit.LAMBDA_H, potts_fit.LAMBDA_J_PRIME, q
            )
            return nll + penalty

        step = 1e-6
        numerical = np.empty_like(theta)
        for index in range(theta.size):
            plus = theta.copy()
            minus = theta.copy()
            plus[index] += step
            minus[index] -= step
            numerical[index] = (value(plus) - value(minus)) / (2.0 * step)
        np.testing.assert_allclose(gradient, numerical, rtol=1e-7, atol=1e-7)

    def test_conditional_probabilities_sum_to_one(self):
        rng = np.random.default_rng(17)
        q = 21
        length = 5
        msa = rng.integers(0, q, size=(9, length), dtype=np.int8)
        fields = rng.normal(size=q)
        couplings = rng.normal(size=(q, length - 1, q))
        probabilities = potts_fit.conditional_probabilities(msa, fields, couplings, 3)
        np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(9))
        self.assertTrue(np.all(probabilities >= 0.0))


class SolverTests(unittest.TestCase):
    def test_only_the_gradient_test_is_success(self):
        def status(**overrides):
            values = dict(
                finite=True,
                grad_inf=1.0,
                step_inf=1.0,
                function_change=1.0,
                direction_derivative=None,
                opt_tol=1e-5,
                prog_tol=1e-9,
                evaluations=3,
                max_fun_evals=1000,
                iteration=2,
                max_iter=500,
            )
            values.update(overrides)
            return potts_fit.stopping_status(**values)

        self.assertEqual(status(grad_inf=1e-8, step_inf=1e-12, function_change=1e-12), "converged")
        self.assertIsNone(status(grad_inf=1e-3, step_inf=1e-12))
        self.assertIsNone(status(grad_inf=1e-3, function_change=1e-12))
        self.assertIsNone(status(step_inf=None, function_change=None, direction_derivative=-1e-12))
        self.assertIsNone(status(step_inf=None, function_change=None, direction_derivative=-1e-4))
        self.assertEqual(status(finite=False, grad_inf=0.0), "nonfinite")
        self.assertEqual(status(evaluations=1000), "max_eval")
        self.assertEqual(status(iteration=500), "max_iter")

    def test_wolfe_step_on_a_quadratic(self):
        def objective(point):
            return float((point[0] - 0.3) ** 2), np.array([2.0 * (point[0] - 0.3)])

        position = np.array([0.0])
        value, gradient = objective(position)
        direction = -gradient
        found = potts_fit.strong_wolfe(objective, position, value, gradient, direction, 1.0)
        self.assertIsNotNone(found)
        self.assertEqual(found.acceptance, "strong_wolfe")
        slope0 = float(np.dot(gradient, direction))
        self.assertLessEqual(found.delta, potts_fit.C1 * found.alpha * slope0)
        self.assertLessEqual(abs(found.slope), potts_fit.C2 * abs(slope0))

    def test_trace_keeps_descent_directions_and_positive_curvature(self):
        def objective(point):
            return 0.5 * float(np.dot(point, point)), np.array(point, copy=True)

        solved = potts_fit.lbfgs(objective, np.array([1.0, -2.0, 0.5]))
        directions = [row for row in solved.trace if row["event"] == "direction"]
        accepted = [row for row in solved.trace if row["event"] == "accepted"]
        self.assertTrue(directions)
        self.assertTrue(all(row["descent"] and row["gtd"] < 0.0 for row in directions))
        self.assertTrue(accepted)
        self.assertTrue(all(row["acceptance"] == "strong_wolfe" for row in accepted))
        self.assertTrue(all(row["curvature_positive"] and row["curvature_sy"] > 0.0 and row["curvature_stored"] for row in accepted))

    def test_lbfgs_quadratic_reaches_the_gradient_tolerance(self):
        def objective(point):
            return 0.5 * float(np.dot(point, point)), np.array(point, copy=True)

        solved = potts_fit.lbfgs(objective, np.array([1.0, -2.0, 0.5, 4.0]), prog_tol=1e9)
        self.assertEqual(solved.status, "converged")
        self.assertLessEqual(solved.grad_inf, potts_fit.OPT_TOL)
        np.testing.assert_allclose(solved.x, np.zeros(4), atol=1e-6)

    def test_iteration_limit_is_not_success(self):
        def objective(point):
            return 0.5 * float(np.dot(point, point)), np.array(point, copy=True)

        solved = potts_fit.lbfgs(
            objective,
            np.ones(3),
            opt_tol=-1.0,
            prog_tol=0.0,
            max_iter=1,
        )
        self.assertEqual(solved.status, "max_iter")
        self.assertNotEqual(solved.status, "converged")

    def test_unbounded_linear_objective_is_not_success(self):
        def objective(point):
            gradient = np.zeros_like(point)
            gradient[0] = 1.0
            return float(point[0]), gradient

        solved = potts_fit.lbfgs(objective, np.zeros(2), max_iter=20, max_fun_evals=40)
        self.assertNotEqual(solved.status, "converged")

    def test_small_conditional_model_converges_from_zero(self):
        rng = np.random.default_rng(19)
        q = 21
        length = 3
        msa = rng.integers(0, q, size=(12, length), dtype=np.int8)
        weights = rng.uniform(0.2, 1.5, size=12)
        self.assertEqual(potts_fit.parameter_count(263, 21), 115563)
        easy = potts_fit.fit_site(msa, weights, 1, 1.0)
        self.assertEqual(easy.status, "converged")
        self.assertLessEqual(easy.grad_inf, potts_fit.OPT_TOL)
        self.assertEqual(easy.h.shape, (q,))
        self.assertEqual(easy.couplings.shape, (q, length - 1, q))
        official_scale = potts_fit.fit_site(msa, weights, 1, potts_fit.OFFICIAL_M_EFF)
        self.assertEqual(official_scale.status, "converged")
        self.assertLessEqual(official_scale.grad_inf, potts_fit.OPT_TOL)
        _nll, _penalty, gradient = potts_fit.site_terms_numpy(
            official_scale.theta,
            msa,
            weights,
            1,
            potts_fit.OFFICIAL_M_EFF,
            potts_fit.LAMBDA_H,
            potts_fit.LAMBDA_J_PRIME,
            q,
        )
        self.assertLessEqual(float(np.max(np.abs(gradient))), potts_fit.OPT_TOL)


class GaugeTests(unittest.TestCase):
    def test_recovered_field_uses_the_fitted_row_means(self):
        length = 4
        q = 5
        fields = np.zeros((length, q))
        fields[:, 0] = 1.0
        fields[:, 1] = -1.0
        directional = np.zeros((length, length, q, q))
        ratio = potts_fit.LAMBDA_H / potts_fit.LAMBDA_J_PRIME
        for i in range(length):
            for j in range(length):
                if i == j:
                    continue
                directional[i, j] = (ratio / q) * fields[i][:, None]
        recovered = gauge.recover_fields(fields, directional)
        factor = 1.0 + (length - 1) * ratio / q
        np.testing.assert_allclose(recovered, factor * fields)
        directional[0, 1, 2, :] += 0.3
        shifted = gauge.recover_fields(fields, directional)
        self.assertAlmostEqual(shifted[0, 2] - recovered[0, 2], 0.3)

    def test_zero_sum_couplings_do_not_change_the_recovered_field(self):
        rng = np.random.default_rng(23)
        length = 5
        q = 7
        fields = rng.normal(size=(length, q))
        raw = rng.normal(size=(length, length, q, q))
        averaged = gauge.average_couplings(raw)
        recovered = gauge.recover_fields(fields, averaged)
        np.testing.assert_allclose(recovered, fields, atol=1e-10)
        for i in range(length):
            for j in range(i + 1, length):
                np.testing.assert_allclose(averaged[i, j].sum(axis=0), np.zeros(q), atol=1e-10)
                np.testing.assert_allclose(averaged[i, j].sum(axis=1), np.zeros(q), atol=1e-10)
                np.testing.assert_allclose(averaged[j, i], averaged[i, j].T)

    def test_single_mutation_matches_the_full_energy_difference(self):
        rng = np.random.default_rng(29)
        length = 6
        q = 21
        fields = rng.normal(size=(length, q))
        couplings = gauge.average_couplings(rng.normal(size=(length, length, q, q)))
        sequence = rng.integers(0, q, size=length)
        mutant = sequence.copy()
        mutant[2] = (int(sequence[2]) + 3) % q
        delta = potts_energy.mutation_energy(sequence, 2, int(mutant[2]), fields, couplings)
        full = potts_energy.sequence_energy(mutant, fields, couplings) - potts_energy.sequence_energy(
            sequence, fields, couplings
        )
        self.assertAlmostEqual(delta, full)


class SolverV2Tests(unittest.TestCase):
    def test_full_msa_run_is_refused(self):
        with self.assertRaises(RuntimeError):
            potts_fit.run_official_point_estimate(Path("/tmp/potts-v2-not-authorized"))

    def test_large_progress_tolerance_does_not_stop_a_quadratic(self):
        def objective(point):
            return 0.5 * float(np.dot(point, point)), np.array(point, copy=True)

        solved = potts_fit.lbfgs(objective, np.array([0.2, -0.4]), prog_tol=1e9)
        self.assertEqual(solved.status, "converged")
        self.assertTrue(any(row.get("legacy_progress") for row in solved.trace if row.get("event") == "accepted"))

    def test_ascent_direction_fails_and_is_not_converged(self):
        def objective(point):
            return 0.5 * float(np.dot(point, point)), np.array(point, copy=True)

        original = potts_fit._two_loop

        def ascent(gradient, steps, changes, curvatures):
            del steps, changes, curvatures
            return np.array(gradient, copy=True)

        potts_fit._two_loop = ascent
        try:
            solved = potts_fit.lbfgs(objective, np.array([1.0, -0.5]))
        finally:
            potts_fit._two_loop = original
        self.assertEqual(solved.status, "ascent_direction")
        self.assertFalse(solved.n_strong_wolfe or solved.n_armijo_fallback)
        np.testing.assert_allclose(solved.x, np.array([1.0, -0.5]))

    def test_site_103_candidate_rule_selects_the_lowest_decrease(self):
        candidates = [
            {"index": 0, "alpha": 1.0, "delta": -1.6e-10, "value": 0.0, "armijo": True, "curvature": False},
            {"index": 1, "alpha": 0.617, "delta": 1.0e-9, "value": 0.0, "armijo": False, "curvature": False},
            {"index": 2, "alpha": 0.99999743, "delta": -2.0e-10, "value": 0.0, "armijo": True, "curvature": False},
            {"index": 3, "alpha": 0.99999745, "delta": -2.0e-10, "value": 0.0, "armijo": True, "curvature": False},
        ]
        chosen = potts_fit.choose_armijo_fallback(candidates)
        self.assertEqual(chosen["alpha"], 0.99999743)
        self.assertIsNone(potts_fit.choose_armijo_fallback([]))

    def test_budget_exhaustion_accepts_fallback_and_not_wolfe(self):
        def objective(point):
            x = float(point[0])
            value = -0.05 * x + 0.2 * (x - 0.3) ** 2
            gradient = np.array([4.0 if x >= 0.5 else -4.0])
            return value, gradient

        position = np.zeros(1)
        value, gradient = 0.0, np.array([-1.0])
        events = []
        found = potts_fit.strong_wolfe(
            objective,
            position,
            value,
            gradient,
            np.array([1.0]),
            1.0,
            max_evals=6,
            events=events,
        )
        self.assertIsNotNone(found)
        self.assertEqual(found.acceptance, "armijo_fallback")
        self.assertNotEqual(found.acceptance, "strong_wolfe")
        armijo = [row for row in events if row.get("armijo") and row.get("alpha", 0.0) > 0.0]
        self.assertGreaterEqual(len(armijo), 2)
        self.assertAlmostEqual(found.delta, min(row["delta"] for row in armijo))
        self.assertGreater(abs(found.slope), potts_fit.C2 * 1.0)
        self.assertFalse(any(row.get("acceptance") == "strong_wolfe" for row in events))

    def test_no_sufficient_decrease_is_a_line_search_failure(self):
        def objective(point):
            x = float(point[0])
            return 10.0 * x * x, np.array([20.0 * x])

        found = potts_fit.strong_wolfe(
            objective,
            np.zeros(1),
            0.0,
            np.array([-1.0]),
            np.array([1.0]),
            1.0,
            max_evals=5,
        )
        self.assertIsNone(found)

    def test_nonpositive_curvature_is_not_stored(self):
        def objective(point):
            x = float(point[0])
            return -0.01 * x, np.array([-1.0 - 4.0 * x])

        solved = potts_fit.lbfgs(objective, np.zeros(1), max_iter=1, max_fun_evals=8, line_search_evals=3)
        accepted = [row for row in solved.trace if row["event"] == "accepted"]
        self.assertTrue(accepted)
        self.assertEqual(accepted[0]["acceptance"], "armijo_fallback")
        self.assertLessEqual(accepted[0]["curvature_sy"], 0.0)
        self.assertFalse(accepted[0]["curvature_stored"])
        self.assertEqual(solved.n_curvature_skipped, 1)
        self.assertEqual(solved.n_strong_wolfe, 0)
        self.assertGreater(len(solved.curvature_sy), 0)

    def test_direct_delta_matches_the_objective_difference(self):
        rng = np.random.default_rng(11)
        q = 21
        length = 4
        rows = 7
        msa = rng.integers(0, q, size=(rows, length), dtype=np.int8)
        weights = rng.uniform(0.2, 1.0, size=rows)
        theta = rng.normal(scale=0.05, size=potts_fit.parameter_count(length, q))
        direction = rng.normal(scale=0.05, size=theta.shape)
        alpha = 0.37
        weight_total = 3.5
        base_nll, base_penalty, _gradient = potts_fit.site_terms_numpy(
            theta, msa, weights, 2, weight_total
        )
        moved_nll, moved_penalty, _gradient = potts_fit.site_terms_numpy(
            theta + alpha * direction, msa, weights, 2, weight_total
        )
        delta_fn = potts_fit.make_direct_delta(msa, weights, 2, weight_total)
        direct = delta_fn(theta, direction, alpha)
        self.assertAlmostEqual(direct, (moved_nll + moved_penalty) - (base_nll + base_penalty), places=8)

    def test_tied_absolute_values_still_accept_the_lower_delta_inside_zoom(self):
        f0 = 85825.06840460224
        slope0 = -1e-12
        curvature_scale = 1.95e-12

        def phi(step):
            return slope0 * step + 0.5 * curvature_scale * step * step

        def objective(point):
            step = float(point[0])
            return f0, np.array([slope0 + curvature_scale * step])

        def delta_fn(position, direction, alpha):
            start = float(position[0])
            moved = start + alpha * float(direction[0])
            return phi(moved) - phi(start)

        self.assertEqual(f0 + phi(1.0), f0)
        self.assertEqual(f0 + phi(1.0 / 1.95), f0)
        events = []
        found = potts_fit.strong_wolfe(
            objective,
            np.zeros(1),
            f0,
            np.array([slope0]),
            np.array([1.0]),
            1.0,
            events=events,
            delta_fn=delta_fn,
        )
        self.assertIsNotNone(found)
        self.assertEqual(found.acceptance, "strong_wolfe")
        self.assertLess(found.delta, phi(1.0))
        self.assertTrue(any(row.get("stage") == "zoom_trial" and row.get("reason") == "strong_wolfe" for row in events))
        self.assertFalse(any(row.get("reason") == "budget" for row in events))

    def test_nonfinite_absolute_objective_is_rejected_before_the_direct_delta(self):
        def objective(point):
            del point
            return math.inf, np.array([1.0])

        def delta_fn(position, direction, alpha):
            del position, direction, alpha
            raise AssertionError("a nonfinite objective must not become a candidate")

        found = potts_fit.strong_wolfe(
            objective,
            np.zeros(1),
            0.0,
            np.array([-1.0]),
            np.array([1.0]),
            1.0,
            max_evals=2,
            delta_fn=delta_fn,
        )
        self.assertIsNone(found)

    def test_direct_delta_accepts_wolfe_and_fallback_on_the_same_objective(self):
        rng = np.random.default_rng(23)
        q = 21
        length = 3
        msa = rng.integers(0, q, size=(8, length), dtype=np.int8)
        weights = np.ones(8)
        weight_total = 4.0
        theta = np.zeros(potts_fit.parameter_count(length, q))
        nll, penalty, gradient = potts_fit.site_terms(theta, msa, weights, 0, weight_total)
        f0 = nll + penalty
        direction = -gradient
        delta_fn = potts_fit.make_direct_delta(msa, weights, 0, weight_total)

        def objective(point):
            value_nll, value_penalty, value_gradient = potts_fit.site_terms(point, msa, weights, 0, weight_total)
            return value_nll + value_penalty, value_gradient

        wolfe = potts_fit.strong_wolfe(objective, theta, f0, gradient, direction, 1.0, delta_fn=delta_fn)
        self.assertEqual(wolfe.acceptance, "strong_wolfe")
        moved_nll, moved_penalty, _gradient = potts_fit.site_terms(
            theta + wolfe.alpha * direction, msa, weights, 0, weight_total
        )
        self.assertAlmostEqual(wolfe.delta, (moved_nll + moved_penalty) - f0, places=8)

        fallback = None
        for alpha0 in (1e-6, 1e-3, 1.0, 1e2, 1e4):
            trial = potts_fit.strong_wolfe(
                objective,
                theta,
                f0,
                gradient,
                direction,
                alpha0,
                max_evals=1,
                delta_fn=delta_fn,
            )
            if trial is not None and trial.acceptance == "armijo_fallback":
                fallback = trial
                break
        self.assertIsNotNone(fallback)
        self.assertNotEqual(fallback.acceptance, "strong_wolfe")
        checked_nll, checked_penalty, _gradient = potts_fit.site_terms(
            theta + fallback.alpha * direction, msa, weights, 0, weight_total
        )
        self.assertAlmostEqual(fallback.delta, (checked_nll + checked_penalty) - f0, places=8)

    def test_budget_exit_selects_only_an_evaluated_sufficient_decrease(self):
        def objective(point):
            return 0.0, np.array([-4.0])

        def delta_fn(position, direction, alpha):
            del position, direction
            return -0.1 * alpha * (1.0 - alpha)

        events = []
        found = potts_fit.strong_wolfe(
            objective,
            np.zeros(1),
            10.0,
            np.array([-1.0]),
            np.array([1.0]),
            1.0,
            max_evals=4,
            events=events,
            delta_fn=delta_fn,
        )
        self.assertEqual(found.acceptance, "armijo_fallback")
        self.assertEqual(events[-1]["line_search_exit"], "budget_exhausted")
        evaluated = [row["alpha"] for row in events if row.get("delta") is not None]
        self.assertIn(found.alpha, evaluated)
        armijo = [row for row in events if row.get("armijo") and row.get("alpha", 0.0) > 0.0]
        self.assertAlmostEqual(found.delta, min(row["delta"] for row in armijo))

    def test_alpha_cap_does_not_use_an_unevaluated_step(self):
        def objective(point):
            del point
            return 0.0, np.array([-1.0])

        def delta_fn(position, direction, alpha):
            del position
            return -alpha * float(direction[0])

        events = []
        found = potts_fit.strong_wolfe(
            objective,
            np.zeros(1),
            0.0,
            np.array([-1.0]),
            np.array([1.0]),
            1.0,
            max_evals=40,
            events=events,
            delta_fn=delta_fn,
        )
        self.assertEqual(found.acceptance, "armijo_fallback")
        self.assertEqual(events[-1]["line_search_exit"], "alpha_cap")
        self.assertLessEqual(found.alpha, 1e8)
        self.assertFalse(any(row.get("alpha", 0.0) > 1e8 and row.get("delta") is not None for row in events))
        self.assertAlmostEqual(found.delta, -found.alpha)

    def test_bracket_collapse_keeps_the_evaluated_candidate(self):
        def objective(point):
            del point
            return 0.0, np.array([1.0])

        def delta_fn(position, direction, alpha):
            del position, direction
            return -0.05 * alpha

        original = potts_fit._cubic_minimizer

        def collapsed(left, f_left, slope_left, right, f_right, slope_right):
            del left, f_left, slope_left, right, f_right, slope_right
            return math.nan

        potts_fit._cubic_minimizer = collapsed
        events = []
        try:
            found = potts_fit.strong_wolfe(
                objective,
                np.zeros(1),
                0.0,
                np.array([-1.0]),
                np.array([1.0]),
                1.0,
                events=events,
                delta_fn=delta_fn,
            )
        finally:
            potts_fit._cubic_minimizer = original
        self.assertEqual(found.acceptance, "armijo_fallback")
        self.assertEqual(found.alpha, 1.0)
        self.assertEqual(events[-1]["line_search_exit"], "bracket_collapse")
        self.assertEqual([row["alpha"] for row in events if row.get("stage") == "trial"], [1.0])

    def test_nonfinite_step_collapse_has_no_invented_candidate(self):
        def objective(point):
            del point
            return math.nan, np.array([math.nan])

        events = []
        found = potts_fit.strong_wolfe(
            objective,
            np.zeros(1),
            0.0,
            np.array([-1.0]),
            np.array([1.0]),
            1e-306,
            max_evals=20,
            events=events,
        )
        self.assertIsNone(found)
        self.assertFalse(any(row.get("acceptance") for row in events))
        self.assertEqual(events[-1]["line_search_exit"], "nonfinite_step")

    def test_lbfgs_records_curvature_scale_and_descent_directions(self):
        def objective(point):
            return 0.5 * float(np.dot(point, point)), np.array(point, copy=True)

        solved = potts_fit.lbfgs(objective, np.array([1.0, -2.0]), max_iter=4)
        directions = [row for row in solved.trace if row["event"] == "direction"]
        accepted = [row for row in solved.trace if row["event"] == "accepted"]
        self.assertTrue(directions and accepted)
        self.assertTrue(all(row["direction_finite"] and row["descent"] and math.isfinite(row["direction_l2"]) for row in directions))
        self.assertTrue(all(row["gtd"] < 0.0 for row in directions))
        for row in accepted:
            self.assertGreater(row["step_l2"], 0.0)
            self.assertGreater(row["y_l2"], 0.0)
            if row["curvature_stored"]:
                self.assertAlmostEqual(row["rho"], 1.0 / row["curvature_sy"])
                self.assertAlmostEqual(row["cos_phi"], row["curvature_sy"] / (row["step_l2"] * row["y_l2"]))
            else:
                self.assertIsNone(row["rho"])
        later = [row for row in directions if row["history_pairs"] > 0]
        self.assertTrue(later)
        self.assertTrue(all(math.isfinite(row["history_rho_max"]) and row["history_rho_max"] > 0.0 for row in later))


class ComponentCheckTests(unittest.TestCase):
    def test_paired_block_matches_the_gauge_average(self):
        rng = np.random.default_rng(13)
        length = 4
        q = 5
        directional = rng.normal(size=(length, length, q, q))
        for site in range(length):
            directional[site, site] = 0.0
        averaged = gauge.average_couplings(directional)
        fields = [directional[site, site, 0].copy() for site in range(length)]
        del fields
        packed = []
        for site in range(length):
            partners = [column for column in range(length) if column != site]
            block = np.stack([directional[site, partner] for partner in partners], axis=1)
            packed.append(block)
        for site in range(length):
            for partner in range(length):
                if partner == site:
                    continue
                block = energy_components.paired_ising_block(
                    energy_components.packed_block(packed[site], site, partner),
                    energy_components.packed_block(packed[partner], partner, site),
                )
                np.testing.assert_allclose(block, averaged[site, partner])

    def test_components_sum_to_the_mutation_energy_and_change(self):
        rng = np.random.default_rng(31)
        length = 5
        q = 6
        directional = rng.normal(size=(length, length, q, q))
        for site in range(length):
            directional[site, site] = 0.0
        raw_fields = rng.normal(size=(length, q))
        recovered = gauge.recover_fields(raw_fields, directional)
        couplings = gauge.average_couplings(directional)
        sequence = rng.integers(0, q, size=length)
        background = rng.random((length, q))
        background /= background.sum(axis=1, keepdims=True)
        groups = [np.array([0, 1, 2]), np.array([3, 4])]
        new_state = (int(sequence[1]) + 2) % q
        background_term, group_terms = energy_components.mutation_components(
            sequence, 1, new_state, recovered, couplings, background, groups
        )
        energy = potts_energy.mutation_energy(sequence, 1, new_state, recovered, couplings)
        self.assertAlmostEqual(background_term + sum(group_terms), energy)
        packed = []
        field_list = []
        for site in range(length):
            field_list.append(raw_fields[site].copy())
            partners = [column for column in range(length) if column != site]
            packed.append(np.stack([directional[site, partner] for partner in partners], axis=1))
        split_background, split_groups, split_energy = energy_components.components_from_directional(
            sequence, 1, new_state, field_list, packed, background, groups
        )
        self.assertAlmostEqual(split_background + sum(split_groups), split_energy)
        self.assertAlmostEqual(split_energy, energy)
        packed[1] = np.array(packed[1], copy=True)
        packed[1][new_state, 0, int(sequence[0])] += 1.0
        changed_background, changed_groups, changed_energy = energy_components.components_from_directional(
            sequence, 1, new_state, field_list, packed, background, groups
        )
        self.assertNotAlmostEqual(changed_energy, split_energy)
        self.assertTrue(
            changed_background != split_background or changed_groups != split_groups
        )


if __name__ == "__main__":
    unittest.main()

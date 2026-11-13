"""Independent checks for the Potts conditional objective, solver, and gauge."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

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
        self.assertEqual(status(grad_inf=1e-3, step_inf=1e-12), "stagnation_step")
        self.assertEqual(status(grad_inf=1e-3, function_change=1e-12), "stagnation_function")
        self.assertEqual(
            status(step_inf=None, function_change=None, direction_derivative=-1e-12),
            "stagnation_direction",
        )
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
        alpha, new_value, new_gradient, _used = found
        slope0 = float(np.dot(gradient, direction))
        self.assertLessEqual(new_value, value + potts_fit.C1 * alpha * slope0)
        self.assertLessEqual(abs(float(np.dot(new_gradient, direction))), potts_fit.C2 * abs(slope0))

    def test_trace_keeps_descent_directions_and_positive_curvature(self):
        def objective(point):
            return 0.5 * float(np.dot(point, point)), np.array(point, copy=True)

        solved = potts_fit.lbfgs(objective, np.array([1.0, -2.0, 0.5]))
        directions = [row for row in solved.trace if row["event"] == "direction"]
        accepted = [row for row in solved.trace if row["event"] == "accepted"]
        self.assertTrue(directions)
        self.assertTrue(all(row["descent"] and row["gtd"] < 0.0 for row in directions))
        self.assertTrue(accepted)
        self.assertTrue(all(row["curvature_positive"] and row["curvature_sy"] > 0.0 for row in accepted))

    def test_lbfgs_quadratic_reaches_the_gradient_tolerance(self):
        def objective(point):
            return 0.5 * float(np.dot(point, point)), np.array(point, copy=True)

        solved = potts_fit.lbfgs(objective, np.array([1.0, -2.0, 0.5, 4.0]))
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
        self.assertEqual(official_scale.status, "stagnation_direction")
        self.assertGreater(official_scale.grad_inf, potts_fit.OPT_TOL)
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
        self.assertGreater(float(np.max(np.abs(gradient))), potts_fit.OPT_TOL)


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


if __name__ == "__main__":
    unittest.main()

"""The per-edge scale uses Hd, but the step that is taken is p = D_alpha d.

H D_alpha d equals D_alpha H d when every moved edge in a block shares one
scale. The five-edge path is that case. An unequal scale, or a frozen edge
next to a moved edge, makes the two products differ. The directional
derivative g·p stays negative because each scale is nonnegative and d points
against g. That is the descent condition. It is not a prediction of the
updated gradient.
"""
import unittest

import numpy as np

from .f81_stage28_trust import capped_diagonal_step, hessian_product


def scales_of(gradient, direction, product_direction):
    component = direction * product_direction
    scales = np.zeros_like(direction)
    usable = component > 0.
    scales[usable] = np.clip(
        -(gradient[usable] * direction[usable]) / component[usable], 0., 1.)
    return scales


class ScaleIdentityTests(unittest.TestCase):
    def test_equal_scales_on_the_five_edge_path_commute(self):
        hessian = 3.5 * np.ones((5, 5))
        gradient = np.full(5, -0.00184)
        diagonal = np.diag(hessian)
        direction = -gradient / diagonal
        product_direction = hessian @ direction
        scales = scales_of(gradient, direction, product_direction)
        step = scales * direction
        np.testing.assert_allclose(scales, np.full(5, 0.2))
        np.testing.assert_allclose(hessian @ step, scales * product_direction, atol=1e-15)
        np.testing.assert_allclose(gradient + hessian @ step, 0., atol=1e-12)
        self.assertLess(float(gradient @ step), 0.)

    def test_a_frozen_neighbor_is_moved_by_the_cross_term(self):
        # Positive definite. Edge 2 is frozen because d_2 (Hd)_2 is negative.
        # D_alpha Hd predicts no change there. H p does change it.
        hessian = np.array([[10., 8.], [8., 10.]])
        gradient = np.array([0.01, -0.002])
        diagonal = np.diag(hessian)
        direction = -gradient / diagonal
        product_direction = hessian @ direction
        scales = scales_of(gradient, direction, product_direction)
        step = scales * direction
        self.assertEqual(float(scales[1]), 0.)
        self.assertGreater(float(scales[0]), 0.)
        predicted_by_scaled_product = scales * product_direction
        actual = hessian @ step
        self.assertGreater(abs(float(actual[1] - predicted_by_scaled_product[1])), 1e-4)
        self.assertLess(float(gradient @ step), 0.)
        updated = gradient + actual
        self.assertGreater(abs(float(updated[1])), abs(float(gradient[1])))
        change = float(gradient @ step + 0.5 * step @ actual)
        self.assertLess(change, 0.)

    def test_random_blocks_keep_descent_and_separate_the_two_products(self):
        generator = np.random.default_rng(29)
        gaps = []
        for _ in range(40):
            size = int(generator.integers(2, 7))
            factor = generator.normal(size=(size, size))
            hessian = factor.T @ factor + np.diag(generator.uniform(0.2, 12., size=size))
            gradient = generator.normal(size=size) * generator.uniform(1e-4, 2e-2, size=size)
            lengths = generator.uniform(1e-4, 0.2, size=size)
            cap = np.minimum(0.25 * lengths, 0.01)
            index = np.arange(size)

            def product(full, hessian=hessian):
                return hessian @ full

            step, info = capped_diagonal_step(gradient, np.diag(hessian), product, index, cap)
            if info["hit"] == "not_descent":
                np.testing.assert_allclose(step, 0.)
                continue
            self.assertLess(float(gradient @ step), 0.)
            actual = hessian @ step
            raw = np.divide(-gradient, np.maximum(np.diag(hessian), 1e-3))
            raw = np.clip(raw, -cap, cap)
            predicted = scales_of(gradient, raw, hessian @ raw) * (hessian @ raw)
            gaps.append(float(np.max(np.abs(actual - predicted))))
            change = float(gradient @ step + 0.5 * step @ actual)
            self.assertLess(change, 0.)
            # The quadratic's gradient at the new point is g + H p, exactly.
            def gradient_at(point, hessian=hessian, gradient=gradient, lengths=lengths):
                return gradient + hessian @ (point - lengths)
            probed = hessian_product(lengths, step, gradient_at)
            np.testing.assert_allclose(probed, actual, rtol=1e-6, atol=1e-10)
        self.assertGreater(len(gaps), 30)
        self.assertGreater(max(gaps), 1e-8)


if __name__ == "__main__":
    unittest.main()

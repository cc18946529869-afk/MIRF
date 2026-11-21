"""Boundary, projection, Armijo accounting and binary64 ULP tests."""
from decimal import Decimal, localcontext
import math
import unittest

from .check_f81_edge_conditioning import ordered_binary64
from .f81_edge_conditioning import EdgeObjective, physical, projected, solve_edge


class EdgeConditioningTests(unittest.TestCase):
    def test_ulp_order_across_sign_and_adjacent_values(self):
        for value in (-7., -1., 0., 1., 7.):
            next_value = math.nextafter(value, math.inf)
            self.assertEqual(ordered_binary64(next_value)-ordered_binary64(value), 1)
        self.assertEqual(ordered_binary64(0.), ordered_binary64(-0.))
        self.assertEqual(ordered_binary64(-math.ulp(0.))+1, ordered_binary64(0.))

    def test_structural_zero_has_finite_log_gradient_limit(self):
        with localcontext() as ctx:
            ctx.prec = 115
            point = physical(1e-28)
            rows = [(Decimal(0), Decimal(1))]
            objective = EdgeObjective(rows, point)
            _, gradient, diagonal = objective.evaluate(point)
            self.assertFalse(objective.zero_feasible)
            self.assertIsNone(objective.evaluate(Decimal(0)))
            self.assertLess(abs(point*gradient+1), Decimal("1e-27"))
            self.assertLess(abs(point*point*diagonal+point*gradient), Decimal("1e-27"))
            fit = solve_edge(rows, 1e-28, "hybrid_log_diag")
            self.assertEqual(fit["status"], "local_projected_gradient")
            self.assertGreater(fit["end_length"], 0.)
            self.assertLessEqual(fit["evaluations"], 96)
            self.assertTrue(all(r["length_trial"] > 0 for r in fit["trials"]))
            self.assertEqual(len(fit["trials"])+1, fit["evaluations"])
            for row in fit["trials"]:
                if row["accepted"]:
                    self.assertLess(Decimal(row["projected_slope"]), 0)
                    self.assertLessEqual(Decimal(row["objective"]), Decimal(row["armijo_rhs"]))

    def test_boundary_kkt_retention_activation_and_return(self):
        with localcontext() as ctx:
            ctx.prec = 115
            should_activate = [(Decimal(".25"), Decimal(1)), (Decimal(4), Decimal(1))]
            fit = solve_edge(should_activate, 0., "hybrid_log_diag")
            self.assertEqual(fit["trials"][0]["mode"], "activation")
            self.assertGreater(fit["end_length"], 0.)
            self.assertEqual(fit["status"], "local_projected_gradient")
            should_be_zero = [(Decimal(4), Decimal(1))]
            stays = solve_edge(should_be_zero, 0., "hybrid_log_diag")
            enters = solve_edge(should_be_zero, .5, "hybrid_log_diag")
            self.assertEqual(stays["evaluations"], 1)
            self.assertEqual(enters["trials"][0]["mode"], "boundary")
            self.assertEqual(enters["end_length"], 0.)
            self.assertEqual(projected(Decimal(0), Decimal(enters["gradient_s"])), 0)


if __name__ == "__main__":
    unittest.main()

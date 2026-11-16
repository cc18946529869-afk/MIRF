import unittest

import numpy as np

import audit_ctx_parity as audit


class ReferenceFormulaTests(unittest.TestCase):
    def test_spring_uses_the_squared_distance_written_in_the_repository(self):
        coordinates = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        hessian = audit.official_hessian(coordinates)
        self.assertAlmostEqual(hessian[0, 0], 1_000_000.0)
        self.assertAlmostEqual(hessian[0, 3], -1_000_000.0)

    def test_seven_directions_include_the_normalized_face_diagonals(self):
        directions = audit.unit_directions(audit.SEVEN_DIRECTIONS)
        self.assertEqual(len(directions), 7)
        self.assertAlmostEqual(np.linalg.norm(directions[3]), 1.0)
        self.assertTrue(np.allclose(directions[3], np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)))

    def test_forward_dci_matches_the_repository_ratio(self):
        responses = np.array(
            [[1.0, 4.0, 1.0], [10.0, 1.0, 1.0], [1.0, 1.0, 8.0]],
            dtype=np.float64,
        )
        _, forward, reverse = audit.official_profiles(responses, [0, 2])
        self.assertAlmostEqual(forward[1], (5.5 / 4.0))
        self.assertNotAlmostEqual(reverse[1], forward[1])

    def test_ambler_numbers_skip_239_and_253(self):
        self.assertEqual(audit.ambler_to_search(26), 24)
        self.assertEqual(audit.ambler_to_search(290), 286)
        with self.assertRaises(ValueError):
            audit.ambler_to_search(239)


if __name__ == "__main__":
    unittest.main()

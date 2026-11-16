import unittest

import numpy as np

import audit_pipeline as audit


class DirectionTests(unittest.TestCase):
    def test_two_directions_differ_on_an_asymmetric_matrix(self):
        responses = np.array([[1.0, 0.0], [10.0, 1.0]], dtype=np.float64)
        active_to, residue_to, asymmetry = audit.directional_coupling(responses, [0])
        self.assertAlmostEqual(active_to[1], 10.0 / 5.5)
        self.assertAlmostEqual(residue_to[1], 0.0)
        self.assertGreater(abs(asymmetry[1]), 1.0)

    def test_response_matrix_uses_columns_as_the_perturbed_site(self):
        coordinates = np.array(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [2.0, 2.0, 0.4]],
            dtype=np.float64,
        )
        values, vectors = audit.vibration_basis(audit.anm_hessian(coordinates, 3.0))
        covariance, used = audit.covariance_from_modes(values, vectors, None)
        self.assertEqual(used, 6)
        responses = audit.response_matrix(covariance)
        self.assertFalse(np.allclose(responses, responses.T))

    def test_ordered_map_keeps_the_known_1zg4_mismatches(self):
        atoms = [
            {"pdb_number": 26, "residue": "H", "coordinate": np.zeros(3)},
            {"pdb_number": 84, "residue": "I", "coordinate": np.ones(3)},
        ]
        wild_type = "X" * 23 + "HV"
        mapped = audit.map_atoms(atoms, wild_type)
        self.assertEqual(mapped[0]["sequence_position"], 24)
        self.assertTrue(mapped[0]["residue_agrees"])
        self.assertEqual(mapped[1]["sequence_position"], 25)
        self.assertFalse(mapped[1]["residue_agrees"])

    def test_cefotaxime_is_not_inferred_from_an_ampicillin_name(self):
        text = "BLAT_ECOLX_Stiffler_2015 ampicillin"
        self.assertIn("ampicillin", text.lower())
        self.assertNotIn("cefotaxime", text.lower())


if __name__ == "__main__":
    unittest.main()

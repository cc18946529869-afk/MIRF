import unittest

import numpy as np

import audit_dynamics_prior as audit


class AssayGateTests(unittest.TestCase):
    def test_rejects_the_proteingym_ampicillin_label(self):
        with self.assertRaises(ValueError):
            audit.require_cefotaxime_assay("Growth (10-2500 ug/mL ampicillin)")

    def test_rejects_an_unspecified_assay(self):
        with self.assertRaises(ValueError):
            audit.require_cefotaxime_assay("antibiotic resistance, MIC")

    def test_accepts_an_explicit_cefotaxime_assay(self):
        self.assertIn(
            "cefotaxime",
            audit.require_cefotaxime_assay("cefotaxime selection, single mutants"),
        )


class StructureTests(unittest.TestCase):
    def test_small_network_has_six_rigid_modes(self):
        coordinates = np.array(
            [
                [0.0, 0.0, 0.0],
                [4.0, 0.2, 0.0],
                [0.3, 4.1, 0.2],
                [0.1, 0.2, 4.0],
                [3.8, 3.7, 0.4],
                [3.6, 0.3, 3.9],
                [0.4, 3.5, 3.8],
                [3.2, 3.1, 3.4],
            ],
            dtype=np.float64,
        )
        hessian = audit.anm_hessian(coordinates, 8.0)
        _covariance, values = audit.mode_covariance(hessian, 2)
        self.assertLess(values[5], 1e-6)
        self.assertGreater(values[6], 1e-6)

    def test_coupling_is_one_when_every_site_is_functional(self):
        magnitudes = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
        _flexibility, coupling = audit.flexibility_and_coupling(magnitudes, [0, 1])
        self.assertTrue(np.allclose(coupling, [1.0, 1.0]))


class MappingTests(unittest.TestCase):
    def test_rank_correlation_of_identical_vectors(self):
        self.assertAlmostEqual(audit.spearman([1.0, 3.0, 2.0], [1.0, 3.0, 2.0]), 1.0)


if __name__ == "__main__":
    unittest.main()

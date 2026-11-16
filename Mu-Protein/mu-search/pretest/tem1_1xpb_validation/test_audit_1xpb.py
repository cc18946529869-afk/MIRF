import unittest

import numpy as np

import audit_1xpb as audit


class AlignmentTests(unittest.TestCase):
    def test_alignment_finds_an_internal_match_without_a_fixed_offset(self):
        pairs = audit.align_sequences("ACWWWDEFG", "DEFG")
        mapped = {target: query for query, target in pairs if query and target}
        self.assertEqual(mapped[1], 6)
        self.assertEqual(mapped[4], 9)

    def test_alignment_opens_a_gap_for_a_missing_segment(self):
        pairs = audit.align_sequences("ACDEFGH", "ACDGH")
        gaps = [query for query, target in pairs if target is None]
        self.assertEqual(gaps, [4, 5])


class CouplingTests(unittest.TestCase):
    def test_equation_six_and_its_reverse_use_different_denominators(self):
        responses = np.array(
            [[1.0, 4.0, 1.0], [10.0, 1.0, 1.0], [1.0, 1.0, 8.0]],
            dtype=np.float64,
        )
        forward = audit.dci_functional_to_residue(responses, [0, 2])
        reverse = audit.dci_residue_to_functional(responses, [0, 2])
        old = audit.column_mean_ratio(responses, [0, 2])
        self.assertAlmostEqual(forward[1], 5.5 / 4.0)
        self.assertAlmostEqual(reverse[1], (2.0 + 0.3) / 2.0)
        self.assertAlmostEqual(old[1], 2.5 / 2.0)
        self.assertNotAlmostEqual(reverse[1], old[1])

    def test_superposition_removes_rotation_and_translation(self):
        left = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        angle = np.pi / 2.0
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]]
        )
        right = left @ rotation.T + np.array([3.0, -2.0, 5.0])
        self.assertLess(audit.kabsch_rmsd(left, right), 1e-8)


if __name__ == "__main__":
    unittest.main()

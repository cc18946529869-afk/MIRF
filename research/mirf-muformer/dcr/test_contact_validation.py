"""Checks for contact-ranking rules, not for a new biological claim."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

import contact_validation


HEADER = (
    "ref_pos\tmature_pos\tpdb_chain\tpdb_auth_resid\tlobe_id\tlobe_name\t"
    "mapped\twt_aa\tpdb_aa\thas_ca\taa_match\n"
)


def atom_line(serial: int, atom: str, residue: str, resid: int, x: float, y: float, z: float) -> str:
    return f"ATOM  {serial:5d}  {atom:<3s} {residue:>3s} A{resid:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 10.00           C"


class ContactValidationTests(unittest.TestCase):
    def test_amino_block_norm_drops_the_gap_state(self) -> None:
        couplings = np.zeros((2, 2, 21, 21), dtype=np.float64)
        couplings[0, 1, 0, 0] = 10.0
        couplings[0, 1, 1, 2] = 3.0
        couplings[0, 1, 2, 1] = 4.0
        norms = contact_validation.amino_block_norm(couplings)
        self.assertAlmostEqual(norms[0, 1], 5.0)

    def test_average_product_correction_removes_a_constant_background(self) -> None:
        scores = np.full((4, 4), 2.0)
        np.fill_diagonal(scores, 0.0)
        corrected = contact_validation.average_product_correction(scores)
        self.assertTrue(np.allclose(corrected, 0.0, atol=1e-8))

    def test_average_product_correction_keeps_every_site_in_the_background(self) -> None:
        scores = np.zeros((4, 4))
        scores[0, 1] = scores[1, 0] = 1.0
        scores[0, 2] = scores[2, 0] = 1.0
        scores[0, 3] = scores[3, 0] = 5.0
        scores[1, 2] = scores[2, 1] = 1.0
        scores[1, 3] = scores[3, 1] = 1.0
        scores[2, 3] = scores[3, 2] = 1.0
        full = contact_validation.average_product_correction(scores)
        without_last = scores.copy()
        without_last[3, :] = 0.0
        without_last[:, 3] = 0.0
        reduced = contact_validation.average_product_correction(without_last)
        self.assertGreater(abs(full[0, 1] - reduced[0, 1]), 1e-8)

    def test_precision_reads_the_ranked_prefix(self) -> None:
        contact = np.array([1, 0, 1, 0], dtype=bool)
        score = np.array([4.0, 3.0, 2.0, 1.0])
        metrics = contact_validation.ranking_metrics(
            np.arange(4),
            np.arange(1, 5),
            contact,
            score,
            length=4,
        )
        self.assertEqual(metrics["precision_at_l"], 0.5)
        self.assertEqual(metrics["precision_at_l_half"], 0.5)
        self.assertAlmostEqual(metrics["average_precision"], (1.0 + 2.0 / 3.0) / 2.0)

    def test_equal_scores_break_toward_the_smaller_site_pair(self) -> None:
        left = np.array([1, 0, 0])
        right = np.array([2, 2, 1])
        score = np.array([1.0, 1.0, 1.0])
        order = contact_validation.ranking_order(left, right, score)
        ranked = [(int(left[index]), int(right[index])) for index in order]
        self.assertEqual(ranked, [(0, 1), (0, 2), (1, 2)])

    def test_a_tie_across_the_cutoff_is_reported(self) -> None:
        left = np.array([0, 0, 1])
        right = np.array([1, 2, 2])
        contact = np.array([True, False, True])
        score = np.array([2.0, 1.0, 1.0])
        metrics = contact_validation.ranking_metrics(left, right, contact, score, length=2)
        boundary = metrics["boundaries"]["l"]
        self.assertTrue(boundary["boundary_tie"])
        self.assertEqual(boundary["tied_group_size"], 2)
        self.assertEqual(boundary["precision"], 0.5)
        self.assertEqual(boundary["precision_if_tied_group_excluded"], 1.0)
        self.assertAlmostEqual(boundary["precision_if_tied_group_included"], 2.0 / 3.0)

    def test_sequence_separation_boundaries(self) -> None:
        included = np.ones(30, dtype=bool)
        left, right = contact_validation.pair_table(included)
        separation = right - left
        self.assertEqual(int(np.count_nonzero(separation >= 6)), 300)
        self.assertEqual(int(np.count_nonzero(separation >= 12)), 171)
        self.assertEqual(int(np.count_nonzero((separation >= 6) & (separation <= 11))), 129)
        self.assertEqual(int(np.count_nonzero((separation >= 12) & (separation <= 23))), 150)
        self.assertEqual(int(np.count_nonzero(separation >= 24)), 21)
        self.assertTrue(np.any(separation == 6))
        self.assertTrue(np.any(separation == 11))
        self.assertFalse(np.any((separation >= 6) & (separation <= 11) & (separation == 5)))
        self.assertFalse(np.any((separation >= 6) & (separation <= 11) & (separation == 12)))
        self.assertFalse(np.any((separation >= 12) & (separation <= 23) & (separation == 24)))
        self.assertFalse(np.any((separation >= 24) & (separation == 23)))

    def test_distance_of_eight_is_not_a_contact(self) -> None:
        points = np.array([[0.0, 0.0, 0.0], [8.0, 0.0, 0.0], [7.9, 0.0, 0.0]])
        contact = contact_validation.contact_matrix(points)
        self.assertFalse(contact[0, 1])
        self.assertTrue(contact[0, 2])

    def test_nonfinite_coordinates_raise_before_a_contact_decision(self) -> None:
        points = np.array([[0.0, 0.0, 0.0], [np.nan, 0.0, 0.0]])
        with self.assertRaises(ValueError):
            contact_validation.contact_matrix(points)

    def test_glycine_uses_ca_and_a_missing_atom_raises(self) -> None:
        expected = (
            {"ref_pos": 1, "mature_pos": 1, "pdb_auth_resid": 10, "wt_aa": "G", "pdb_aa": "G"},
        )
        rows = "1\t1\tA\t10\t0\tAB\t1\tG\tG\t1\t0\n"
        second = "2\t2\tA\t11\t0\tAB\t1\tA\tA\t1\t1\n"
        pdb = "\n".join(
            [
                atom_line(1, "CA", "GLY", 10, 0.0, 0.0, 0.0),
                atom_line(2, "CB", "GLY", 10, 50.0, 0.0, 0.0),
                atom_line(3, "CA", "ALA", 11, 40.0, 0.0, 0.0),
                atom_line(4, "CB", "ALA", 11, 1.0, 0.0, 0.0),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            domain = root / "domain_map.tsv"
            structure = root / "structure.pdb"
            domain.write_text(HEADER + rows + second)
            structure.write_text(pdb + "\n")
            points, mismatched = contact_validation.site_points(domain, structure, length=2, expected_mismatches=expected)
            contact = contact_validation.contact_matrix(points)
            self.assertTrue(contact[0, 1])
            self.assertTrue(mismatched[0])
            self.assertFalse(mismatched[1])
            structure.write_text(atom_line(1, "CA", "GLY", 10, 0.0, 0.0, 0.0) + "\n")
            with self.assertRaises(ValueError):
                contact_validation.site_points(domain, structure, length=2, expected_mismatches=expected)

    def test_mapping_rejects_a_repeated_mature_position(self) -> None:
        row = {
            "ref_pos": 82,
            "mature_pos": 1,
            "chain": "A",
            "pdb_auth_resid": 84,
            "wt_aa": "V",
            "pdb_aa": "I",
            "has_ca": "1",
            "aa_match": "0",
        }
        with self.assertRaises(ValueError):
            contact_validation.require_site_mapping([row, dict(row)], 2, (row,))

    def test_an_insertion_code_is_refused(self) -> None:
        line = atom_line(1, "CA", "GLY", 10, 0.0, 0.0, 0.0)
        line = line[:26] + "A" + line[27:]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "structure.pdb"
            path.write_text(line + "\n")
            with self.assertRaises(ValueError):
                contact_validation.coordinates_from_pdb(path)

    def test_mapping_rejects_a_different_mismatch(self) -> None:
        rows = [
            {
                "ref_pos": 3,
                "mature_pos": 1,
                "chain": "A",
                "pdb_auth_resid": 1,
                "wt_aa": "A",
                "pdb_aa": "C",
                "has_ca": "1",
                "aa_match": "0",
            }
        ]
        with self.assertRaises(ValueError):
            contact_validation.require_site_mapping(rows, 1, contact_validation.EXPECTED_MISMATCHES)

    def test_real_domain_map_has_the_two_known_mismatches_and_finite_coordinates(self) -> None:
        root = Path(__file__).resolve().parents[1]
        points, mismatched = contact_validation.site_points(root / "prep" / "domain_map.tsv", root / "prep" / "1BTL.pdb")
        self.assertEqual(points.shape, (263, 3))
        self.assertTrue(np.isfinite(points).all())
        self.assertEqual(np.flatnonzero(mismatched).tolist(), [58, 158])


if __name__ == "__main__":
    unittest.main()

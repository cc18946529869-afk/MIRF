import csv
import tempfile
import unittest
from pathlib import Path

import audit_paper_parity as audit


class MutationTests(unittest.TestCase):
    def test_four_sites_match_wild_type(self):
        for position, residue in ((40, "A"), (102, "E"), (180, "M"), (236, "G")):
            self.assertEqual(audit.WILD_TYPE[position - 1], residue)

    def test_apply_one_and_four(self):
        single = audit.apply_mutations(audit.WILD_TYPE, ["A40G"])
        self.assertEqual(single[39], "G")
        self.assertEqual(audit.hamming(single, audit.WILD_TYPE), 1)
        quadruple = audit.apply_mutations(
            audit.WILD_TYPE, ["A40G", "E102K", "M180T", "G236S"]
        )
        self.assertEqual(audit.hamming(quadruple, audit.WILD_TYPE), 4)

    def test_refuses_wrong_residue(self):
        with self.assertRaises(ValueError):
            audit.apply_mutations(audit.WILD_TYPE, ["K40G"])


class CorrelationTests(unittest.TestCase):
    def test_perfect_and_reversed(self):
        self.assertAlmostEqual(audit.spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)
        self.assertAlmostEqual(audit.spearman([1, 2, 3, 4], [40, 30, 20, 10]), -1.0)

    def test_tied_ranks_use_the_average(self):
        self.assertEqual(audit.rank_descending([5, 5, 1]), [1.5, 1.5, 3])


class TableTests(unittest.TestCase):
    def test_synthetic_panel_matches_published_numbers(self):
        rows = []
        for key, mic in audit.WEINREICH_MIC.items():
            name = "WT" if not key else ";".join(sorted(key))
            rows.append("{0}\t{1}\n".format(name, mic))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "panel.tsv"
            path.write_text("mutation\tscore\n" + "".join(rows), encoding="utf-8")
            parsed = audit.read_mic_table(path)
            self.assertTrue(audit.weinreich_match_report(parsed)["complete"])

    def test_rejects_a_changed_label(self):
        text = "mutation\tscore\nWT\t0.088\nA40G\t9\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "panel.tsv"
            path.write_text(text, encoding="utf-8")
            parsed = audit.read_mic_table(path)
            self.assertFalse(audit.weinreich_match_report(parsed)["complete"])

    def test_write_marks_validation_inference(self):
        rows = [
            {"mutation": "WT", "order": 0, "mic": 0.088, "weinreich_mic": 0.088},
            {"mutation": "G236S", "order": 1, "mic": 1.4, "weinreich_mic": 1.4},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "out.csv"
            audit.write_validation(rows, [0.1, 0.2], path)
            with path.open(encoding="utf-8") as handle:
                written = list(csv.DictReader(handle))
        self.assertEqual(written[1]["purpose"], "validation_inference")
        self.assertEqual(written[1]["prediction_rank"], "1")


if __name__ == "__main__":
    unittest.main()

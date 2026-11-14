"""Checks for bootstrap sampling, gauge assembly, and the R/D decision."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bootstrap_fit
import bootstrap_gauge
import bootstrap_stats
import gauge
import potts_fit


class GaugeAssemblyTests(unittest.TestCase):
    def test_packed_partners_round_trip_in_column_order(self):
        length, q = 4, 3
        generator = np.random.default_rng(3)
        directional = generator.normal(size=(length, length, q, q))
        for site in range(length):
            directional[site, site] = 0.0
        fields = generator.normal(size=(length, q))
        packed_fields = []
        packed_couplings = []
        for site in range(length):
            partners = [partner for partner in range(length) if partner != site]
            block = np.zeros((q, length - 1, q), dtype=np.float64)
            for slot, partner in enumerate(partners):
                self.assertEqual(bootstrap_gauge.partner_index(site, partner), slot)
                block[:, slot, :] = directional[site, partner]
            packed_fields.append(fields[site])
            packed_couplings.append(block)
        restored_fields, restored = bootstrap_gauge.assemble_directional(packed_fields, packed_couplings)
        np.testing.assert_allclose(restored_fields, fields)
        np.testing.assert_allclose(restored, directional)

    def test_recovered_field_uses_the_pretransform_couplings(self):
        directional = np.zeros((3, 3, 2, 2), dtype=np.float64)
        directional[0, 1, 0, 0] = 4.0
        fields = np.zeros((3, 2), dtype=np.float64)
        recovered, averaged = bootstrap_gauge.gauge_parameters(fields, directional)
        direct = gauge.recover_fields(fields, directional)
        np.testing.assert_allclose(recovered, direct)
        self.assertNotAlmostEqual(float(recovered[0, 0]), 0.0)
        fed_back = gauge.recover_fields(fields, averaged)
        self.assertGreater(abs(float(recovered[0, 0] - fed_back[0, 0])), 0.1)

    def test_averaged_couplings_keep_both_triangles(self):
        generator = np.random.default_rng(5)
        directional = generator.normal(size=(5, 5, 2, 2))
        for site in range(5):
            directional[site, site] = 0.0
        averaged = gauge.average_couplings(directional)
        transposed = np.transpose(averaged, (1, 0, 3, 2))
        np.testing.assert_allclose(averaged, transposed)
        upper = np.sum(averaged[np.triu_indices(5, k=1)] ** 2)
        self.assertAlmostEqual(float(np.sum(averaged ** 2)), float(2.0 * upper))


class StatisticTests(unittest.TestCase):
    def test_zero_deviation_and_zero_reference_are_distinct(self):
        samples = np.ones((6, 2), dtype=np.float64)
        self.assertEqual(bootstrap_stats.relative_frobenius(samples, np.array([2.0, 0.0])), 0.0)
        self.assertIsNone(bootstrap_stats.relative_frobenius(samples, np.zeros(2)))
        self.assertEqual(bootstrap_stats.relative_change(0.0, 0.0), 0.0)

    def test_half_range_is_the_preassigned_ids(self):
        success = [False] * 10 + [True] * 90
        fields = {identifier: np.array([1.0, 2.0]) for identifier in range(1, 101) if success[identifier - 1]}
        couplings = {identifier: np.zeros((2, 2, 1, 1)) for identifier in fields}
        couplings_reference = np.zeros((2, 2, 1, 1))
        couplings_reference[0, 1, 0, 0] = 1.0
        couplings_reference[1, 0, 0, 0] = 1.0
        outcome = bootstrap_stats.evaluate_stage(
            success,
            fields,
            couplings,
            np.array([1.0, 2.0]),
            couplings_reference,
            100,
        )
        self.assertEqual(outcome["success_count"], 90)
        self.assertEqual(outcome["action"], "pass")
        self.assertEqual(outcome["d_h"], 0.0)

    def test_too_few_successes_do_not_compute_d(self):
        success = [True] * 89 + [False] * 11
        outcome = bootstrap_stats.evaluate_stage(success, {}, {}, np.ones(2), np.ones((1, 1)), 100)
        self.assertEqual(outcome["action"], "extend")
        self.assertIsNone(outcome["r_h"])
        self.assertIsNone(outcome["d_h"])

    def test_stable_deviation_with_large_parameter_spread_does_not_add_replicates(self):
        success = [True] * 100
        fields = {}
        for identifier in range(1, 101):
            fields[identifier] = np.array([0.0, 0.0] if identifier % 2 else [10.0, 0.0])
        couplings = {identifier: np.ones((1, 1)) for identifier in fields}
        outcome = bootstrap_stats.evaluate_stage(success, fields, couplings, np.array([1.0, 0.0]), np.ones((1, 1)), 100)
        self.assertLessEqual(outcome["d_h"], 0.10)
        self.assertGreater(outcome["r_h"], 0.10)
        self.assertEqual(outcome["action"], "parameter_unstable")
        self.assertFalse(outcome["passed"])

    def test_mismatched_shape_or_nonfinite_values_are_rejected(self):
        with self.assertRaises(ValueError):
            bootstrap_stats.relative_frobenius(np.ones((4, 2)), np.ones(3))
        with self.assertRaises(ValueError):
            bootstrap_stats.sample_deviation(np.array([[1.0, np.nan], [0.0, 1.0]]))
        success = [True] * 100
        fields = {identifier: np.array([1.0, 2.0]) for identifier in range(1, 101)}
        fields[1] = np.array([np.nan, 2.0])
        couplings = {identifier: np.ones((1, 1)) for identifier in fields}
        with self.assertRaises(ValueError):
            bootstrap_stats.evaluate_stage(success, fields, couplings, np.array([1.0, 2.0]), np.ones((1, 1)), 100)

    def test_two_hundred_without_enough_successes_has_no_parameter_verdict(self):
        success = [True] * 179 + [False] * 21
        outcome = bootstrap_stats.evaluate_stage(success, {}, {}, np.ones(2), np.ones((1, 1)), 200)
        self.assertEqual(outcome["action"], "no_conclusion")
        self.assertIsNone(outcome["parameter_stable"])


class SamplingTests(unittest.TestCase):
    def test_duplicate_draws_remain_separate_rows(self):
        alignment = np.arange(12, dtype=np.int8).reshape(4, 3)
        sample = bootstrap_fit.replicate_sample(alignment, np.array([0, 0, 2]))
        self.assertEqual(sample.shape, (3, 3))
        np.testing.assert_array_equal(sample[0], sample[1])
        self.assertFalse(np.array_equal(sample[1], sample[2]))

    def test_official_probabilities_match_the_weight_total(self):
        _alignment, weights, total = potts_fit.load_official_alignment()
        self.assertEqual(total, potts_fit.OFFICIAL_M_EFF)
        generator = np.random.default_rng(11)
        indices = bootstrap_fit.draw_indices(generator, weights, total, count=1000)
        self.assertEqual(indices.shape, (1000,))
        self.assertTrue(np.all((indices >= 0) & (indices < weights.shape[0])))

    def test_formal_replicate_streams_are_independent_of_order(self):
        first = bootstrap_fit.formal_seed_sequence(19, 1).integers(0, 1000, size=4)
        again = bootstrap_fit.formal_seed_sequence(19, 1).integers(0, 1000, size=4)
        later = bootstrap_fit.formal_seed_sequence(19, 101).integers(0, 1000, size=4)
        np.testing.assert_array_equal(first, again)
        self.assertFalse(np.array_equal(first, later))

    def test_timing_seed_is_rejected_for_verification(self):
        with self.assertRaises(ValueError):
            bootstrap_fit.run_verification(20261114)

    def test_identical_sequences_must_share_neighbor_counts(self):
        alignment = np.array(
            [
                [0, 0, 0],
                [0, 0, 0],
                [1, 1, 1],
                [1, 0, 0],
            ],
            dtype=np.int8,
        )
        counts = np.array([2, 2, 1, 1], dtype=np.int64)
        self.assertEqual(bootstrap_fit.identical_sequences_share_counts(alignment, counts), 1)
        self.assertEqual(bootstrap_fit.recount_neighbors(alignment, 0, threshold=3), 2)
        self.assertEqual(bootstrap_fit.recount_neighbors(alignment, 3, threshold=3), 1)
        with self.assertRaises(RuntimeError):
            bootstrap_fit.identical_sequences_share_counts(alignment, np.array([2, 3, 1, 1]))
        with self.assertRaises(RuntimeError):
            bootstrap_fit.require_msa_row_order(np.array([0, 2, 1, 3]), 4)

    def test_official_weight_rows_match_the_alignment(self):
        report = bootstrap_fit.verify_official_weight_rows()
        self.assertTrue(report["row_index_is_msa_order"])
        self.assertTrue(report["identical_sequences_share_counts"])
        self.assertGreater(report["duplicate_groups"], 0)
        self.assertFalse(report["full_pair_recount"])
        self.assertFalse(report["spot_check_covers_every_row"])
        self.assertEqual(report["spot_check_row_count"], 32)
        self.assertTrue(report["stored_weight_column_matches_inverse_count"])
        self.assertEqual(report["query_m"], 720)
        self.assertEqual(report["recount_rows"][0], {"row": 0, "m_i": 720})


def _fit_row(site: int, passed: bool, partners: int, fallback: int = 0) -> dict:
    return {
        "site": site,
        "status": "converged" if passed else "line_search",
        "grad_inf": 1e-6 if passed else 1.0,
        "objective": 1.0,
        "iterations": 3,
        "evaluations": 4,
        "passed": passed,
        "finite_parameters": True,
        "seconds": 0.1,
        "n_strong_wolfe": 2,
        "n_armijo_fallback": fallback,
        "n_curvature_skipped": 1,
        "curvature_sy": [0.2],
        "line_search_exits": ["budget_exhausted"] if fallback else [],
        "h": np.array([0.1, -0.1], dtype=np.float64),
        "couplings": np.full((2, partners, 2), float(site + 1), dtype=np.float64),
    }


@contextmanager
def _temporary_output():
    original = bootstrap_fit.OUTPUT
    with tempfile.TemporaryDirectory() as tmp:
        bootstrap_fit.OUTPUT = Path(tmp)
        try:
            yield Path(tmp)
        finally:
            bootstrap_fit.OUTPUT = original


class IntegrityTests(unittest.TestCase):
    def test_failed_replicate_does_not_keep_gauge_parameters(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            indices = np.arange(bootstrap_fit.DRAW_COUNT, dtype=np.int64)
            identity = bootstrap_fit.make_identity("formal", 7, 1, indices)
            rows = [_fit_row(site, passed=site != 0, partners=1) for site in range(bootstrap_fit.SITE_COUNT)]
            np.save(directory / "h_hat.npy", np.zeros(1))
            np.save(directory / "j_star.npy", np.zeros(1))
            manifest = bootstrap_fit.finalize_replicate(
                directory,
                None,
                indices,
                rows,
                identity,
                {"formal_seed": 7, "formal_replicate_id": 1, "workers": 40, "threads": 1},
            )
            self.assertFalse(manifest["replicate_success"])
            self.assertFalse(manifest["entered_formal_statistics"])
            self.assertIsNone(manifest["both_triangles"])
            self.assertEqual(manifest["failed_sites"], [0])
            self.assertFalse((directory / "h_hat.npy").exists())
            self.assertFalse((directory / "j_star.npy").exists())
            self.assertTrue((directory / "indices.npy").exists())
            saved = json.loads((directory / "diagnostics.json").read_text())
            self.assertEqual(len(saved), bootstrap_fit.SITE_COUNT)
            self.assertEqual(saved[0]["status"], "line_search")

    def test_pass_flag_must_match_the_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            indices = np.arange(bootstrap_fit.DRAW_COUNT, dtype=np.int64)
            identity = bootstrap_fit.make_identity("verification", 1, None, indices)
            rows = [_fit_row(site, passed=True, partners=1) for site in range(bootstrap_fit.SITE_COUNT)]
            rows[0]["grad_inf"] = 1.0
            with self.assertRaises(RuntimeError):
                bootstrap_fit.finalize_replicate(directory, None, indices, rows, identity, {"workers": 40})

    def test_resume_rejects_another_sample_and_an_empty_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            row = _fit_row(0, passed=True, partners=1, fallback=4)
            bootstrap_fit._write_site(directory, row, "sample-a")
            self.assertFalse(any("tmp" in path.name for path in directory.iterdir()))
            self.assertEqual(bootstrap_fit._read_token(directory / "site_000.npz"), "sample-a")
            original = bootstrap_fit.SITE_COUNT
            bootstrap_fit.SITE_COUNT = 2
            try:
                self.assertEqual(bootstrap_fit.pending_sites(directory, "sample-a"), [1])
                with self.assertRaises(RuntimeError):
                    bootstrap_fit.pending_sites(directory, "sample-b")
            finally:
                bootstrap_fit.SITE_COUNT = original
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "site_000.npz").write_bytes(b"")
            with self.assertRaises(RuntimeError):
                bootstrap_fit.pending_sites(directory, "sample-a")

    def test_saved_indices_from_another_draw_stop_the_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            indices = np.arange(bootstrap_fit.DRAW_COUNT, dtype=np.int64)
            np.save(directory / "indices.npy", indices + 1)
            with self.assertRaises(RuntimeError):
                bootstrap_fit.ensure_indices(directory, indices)
            identity = bootstrap_fit.make_identity("verification", 1, None, indices)
            (directory / "manifest.json").write_text('{"state": "running"}\n')
            before = (directory / "manifest.json").read_text()
            with self.assertRaises(RuntimeError):
                bootstrap_fit.open_replicate(directory, identity, indices, {})
            self.assertEqual((directory / "manifest.json").read_text(), before)
            np.save(directory / "indices.npy", indices)
            with self.assertRaises(RuntimeError):
                bootstrap_fit.open_replicate(directory, identity, indices, {})
            self.assertEqual((directory / "manifest.json").read_text(), before)

    def test_successful_replicate_writes_both_triangles_and_keeps_fallback_counts(self):
        original = bootstrap_fit.SITE_COUNT
        bootstrap_fit.SITE_COUNT = 2
        try:
            with tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                indices = np.arange(bootstrap_fit.DRAW_COUNT, dtype=np.int64)
                identity = bootstrap_fit.make_identity("verification", 1, None, indices)
                for site in range(2):
                    bootstrap_fit._write_site(directory, _fit_row(site, True, partners=1, fallback=4), identity["identity_token"])
                rows = [bootstrap_fit._read_site(directory / f"site_{site:03d}.npz", site) for site in range(2)]
                sample = np.zeros((bootstrap_fit.DRAW_COUNT, 2), dtype=np.int8)
                manifest = bootstrap_fit.finalize_replicate(
                    directory,
                    sample,
                    indices,
                    rows,
                    identity,
                    {"workers": 40, "verification_seed": 1, "formal_seed": None, "formal_replicate_id": None},
                )
                self.assertTrue(manifest["replicate_success"])
                self.assertTrue(manifest["both_triangles"])
                self.assertFalse(manifest["entered_formal_statistics"])
                saved = np.load(directory / "j_star.npy")
                np.testing.assert_array_equal(saved, np.transpose(saved, (1, 0, 3, 2)))
                diagnostics = json.loads((directory / "diagnostics.json").read_text())
                self.assertEqual(diagnostics[0]["n_armijo_fallback"], 4)
                self.assertEqual(diagnostics[0]["n_curvature_skipped"], 1)
                self.assertEqual(diagnostics[0]["line_search_exits"], ["budget_exhausted"])
        finally:
            bootstrap_fit.SITE_COUNT = original

    def test_formal_timing_seed_and_missing_seed_do_not_start(self):
        with self.assertRaises(ValueError):
            bootstrap_fit.run_formal(20261114)
        self.assertFalse((bootstrap_fit.OUTPUT / "formal").exists())
        with self.assertRaises(SystemExit):
            bootstrap_fit.main(["formal"])
        with self.assertRaises(SystemExit):
            bootstrap_fit.main(["decide", "--seed", "3"])

    def test_formal_seed_lock_rejects_a_second_seed(self):
        with _temporary_output():
            data = {"alignment_sha256": "a", "neighbor_counts_sha256": "b", "code_revision": "abc"}
            bootstrap_fit.lock_formal_seed(5, data)
            with self.assertRaises(RuntimeError):
                bootstrap_fit.lock_formal_seed(6, data)

    def test_code_revision_mismatch_rejects_resume(self):
        with _temporary_output():
            data = {"alignment_sha256": "a", "neighbor_counts_sha256": "b", "code_revision": "aaa"}
            bootstrap_fit.lock_formal_seed(5, data)
            changed = dict(data, code_revision="bbb")
            with self.assertRaises(RuntimeError):
                bootstrap_fit.lock_formal_seed(5, changed)

    def test_dirty_worktree_rejects_a_formal_run(self):
        with self.assertRaises(RuntimeError):
            bootstrap_fit.require_clean_worktree([" M research/mirf-muformer/dcr/bootstrap_fit.py"])
        with self.assertRaises(RuntimeError):
            bootstrap_fit.require_clean_worktree(["?? research/mirf-muformer/dcr/extra.py"])
        bootstrap_fit.require_clean_worktree(["?? research/mirf-muformer/prep/early_abort_histogram.tsv"])
        bootstrap_fit.require_clean_worktree([])

    def test_initial_command_cannot_cross_100(self):
        bootstrap_fit.validate_formal_span(20, 40)
        bootstrap_fit.validate_formal_span(1, 100)
        with self.assertRaises(RuntimeError):
            bootstrap_fit.validate_formal_span(90, 110)
        with self.assertRaises(RuntimeError):
            bootstrap_fit.run_formal(4, 1, 200)
        self.assertFalse((bootstrap_fit.OUTPUT / "formal").exists())

    def test_extension_follows_the_stage_100_decision(self):
        with _temporary_output():
            with self.assertRaises(RuntimeError):
                bootstrap_fit.require_stage_extension(9)
            for action in ("pass", "parameter_unstable", "undefined"):
                _write_stage_decision(action, 9)
                with self.assertRaises(RuntimeError):
                    bootstrap_fit.require_stage_extension(9)
            _write_stage_decision("extend", 8)
            with self.assertRaises(RuntimeError):
                bootstrap_fit.require_stage_extension(9)
            _write_stage_decision("extend", 9)
            bootstrap_fit.require_stage_extension(9)

    def test_extension_requires_the_decision_to_match_the_lock(self):
        with _temporary_output():
            _write_stage_decision("extend", 9)
            stage_path = bootstrap_fit.OUTPUT / "formal" / "stage_100.json"
            lock_path = bootstrap_fit.OUTPUT / "formal" / "seed.json"
            stage = json.loads(stage_path.read_text())
            stage["code_revision"] = "other-commit"
            stage_path.write_text(json.dumps(stage))
            with self.assertRaises(RuntimeError):
                bootstrap_fit.require_stage_extension(9)
            stage["code_revision"] = "rev"
            stage["alignment_sha256"] = "other-alignment"
            stage_path.write_text(json.dumps(stage))
            with self.assertRaises(RuntimeError):
                bootstrap_fit.require_stage_extension(9)
            lock_path.unlink()
            stage["alignment_sha256"] = "a"
            stage_path.write_text(json.dumps(stage))
            with self.assertRaises(RuntimeError):
                bootstrap_fit.require_stage_extension(9)
            self.assertFalse(lock_path.exists())

    def test_existing_replicates_without_a_lock_are_not_adopted(self):
        with _temporary_output():
            directory = bootstrap_fit._formal_directory(1)
            directory.mkdir(parents=True)
            (directory / "manifest.json").write_text("{}\n")
            data = {"alignment_sha256": "a", "neighbor_counts_sha256": "b", "code_revision": "abc"}
            with self.assertRaises(RuntimeError):
                bootstrap_fit.lock_formal_seed(5, data)
            self.assertFalse((bootstrap_fit.OUTPUT / "formal" / "seed.json").exists())

    def test_decide_checks_the_worktree_and_does_not_create_a_lock(self):
        original = bootstrap_fit.require_clean_worktree
        calls = []

        def stop(lines=None):
            calls.append(lines)
            raise RuntimeError("dirty worktree")

        bootstrap_fit.require_clean_worktree = stop
        try:
            with _temporary_output():
                with self.assertRaises(RuntimeError):
                    bootstrap_fit.write_stage_decision(100, 3)
                self.assertEqual(calls, [None])
                self.assertFalse((bootstrap_fit.OUTPUT / "formal").exists())
        finally:
            bootstrap_fit.require_clean_worktree = original

    def test_stage_file_records_the_locked_identity(self):
        originals = (
            bootstrap_fit.require_clean_worktree,
            bootstrap_fit.require_reference,
            bootstrap_fit.decide_formal_stage,
        )
        bootstrap_fit.require_clean_worktree = lambda lines=None: None
        bootstrap_fit.require_reference = lambda data: (np.zeros(1), np.zeros(1))
        bootstrap_fit.decide_formal_stage = lambda *args: {"action": "extend", "stage": 100}
        try:
            with _temporary_output():
                data = bootstrap_fit.data_record()
                bootstrap_fit.lock_formal_seed(3, data)
                saved = bootstrap_fit.write_stage_decision(100, 3)
                self.assertEqual(saved["action"], "extend")
                self.assertEqual(saved["formal_seed"], 3)
                self.assertEqual(saved["code_revision"], data["code_revision"])
                self.assertEqual(saved["alignment_sha256"], data["alignment_sha256"])
                self.assertEqual(saved["neighbor_counts_sha256"], data["neighbor_counts_sha256"])
                on_disk = json.loads((bootstrap_fit.OUTPUT / "formal" / "stage_100.json").read_text())
                self.assertEqual(on_disk, saved)
        finally:
            (
                bootstrap_fit.require_clean_worktree,
                bootstrap_fit.require_reference,
                bootstrap_fit.decide_formal_stage,
            ) = originals

    def test_stage_decision_does_not_create_a_missing_lock(self):
        original = bootstrap_fit.require_clean_worktree
        bootstrap_fit.require_clean_worktree = lambda lines=None: None
        try:
            with _temporary_output():
                with self.assertRaises(RuntimeError):
                    bootstrap_fit.write_stage_decision(100, 3)
                self.assertFalse((bootstrap_fit.OUTPUT / "formal" / "seed.json").exists())
                self.assertFalse((bootstrap_fit.OUTPUT / "formal" / "stage_100.json").exists())
        finally:
            bootstrap_fit.require_clean_worktree = original

    def test_stage_decision_refuses_a_lock_from_another_revision(self):
        original = bootstrap_fit.require_clean_worktree
        bootstrap_fit.require_clean_worktree = lambda lines=None: None
        try:
            with _temporary_output():
                data = bootstrap_fit.data_record()
                lock_path = bootstrap_fit.OUTPUT / "formal" / "seed.json"
                lock_path.parent.mkdir(parents=True)
                bootstrap_fit.atomic_json(
                    lock_path,
                    {
                        "formal_seed": 3,
                        "alignment_sha256": data["alignment_sha256"],
                        "neighbor_counts_sha256": data["neighbor_counts_sha256"],
                        "code_revision": "not-this-commit",
                    },
                )
                with self.assertRaises(RuntimeError):
                    bootstrap_fit.write_stage_decision(100, 3)
                self.assertFalse((bootstrap_fit.OUTPUT / "formal" / "stage_100.json").exists())
        finally:
            bootstrap_fit.require_clean_worktree = original

    def test_stage_200_decision_requires_the_stage_100_extension(self):
        original_clean = bootstrap_fit.require_clean_worktree
        original_reference = bootstrap_fit.require_reference
        original_decide = bootstrap_fit.decide_formal_stage
        bootstrap_fit.require_clean_worktree = lambda lines=None: None
        bootstrap_fit.require_reference = lambda data: (np.zeros(1), np.zeros(1))
        bootstrap_fit.decide_formal_stage = lambda *args: {"action": "no_conclusion", "stage": 200}
        try:
            with _temporary_output():
                data = bootstrap_fit.data_record()
                bootstrap_fit.lock_formal_seed(11, data)
                with self.assertRaises(RuntimeError):
                    bootstrap_fit.write_stage_decision(200, 11)
                identity = {
                    "code_revision": data["code_revision"],
                    "alignment_sha256": data["alignment_sha256"],
                    "neighbor_counts_sha256": data["neighbor_counts_sha256"],
                }
                stage_path = bootstrap_fit.OUTPUT / "formal" / "stage_100.json"
                bootstrap_fit.atomic_json(
                    stage_path,
                    {"action": "pass", "formal_seed": 11, "stage": 100, **identity},
                )
                with self.assertRaises(RuntimeError):
                    bootstrap_fit.write_stage_decision(200, 11)
                self.assertFalse((bootstrap_fit.OUTPUT / "formal" / "stage_200.json").exists())
                bootstrap_fit.atomic_json(
                    stage_path,
                    {"action": "extend", "formal_seed": 11, "stage": 100, **identity},
                )
                saved = bootstrap_fit.write_stage_decision(200, 11)
                self.assertEqual(saved["action"], "no_conclusion")
                self.assertEqual(saved["code_revision"], data["code_revision"])
                self.assertTrue((bootstrap_fit.OUTPUT / "formal" / "stage_200.json").is_file())
        finally:
            bootstrap_fit.require_clean_worktree = original_clean
            bootstrap_fit.require_reference = original_reference
            bootstrap_fit.decide_formal_stage = original_decide

    def test_reference_identity_must_match_before_it_is_used(self):
        with _temporary_output():
            data = bootstrap_fit.data_record()
            reference = bootstrap_fit.OUTPUT / "reference"
            reference.mkdir()
            manifest = {
                "role": "reference",
                "source": str(bootstrap_fit.FULL_FIT),
                "refit": False,
                "sites": bootstrap_fit.SITE_COUNT,
                "both_triangles": True,
                **data,
            }
            manifest["alignment_sha256"] = "different-alignment"
            bootstrap_fit.atomic_json(reference / "manifest.json", manifest)
            with self.assertRaises(RuntimeError):
                bootstrap_fit.require_reference(data)
            self.assertEqual(
                json.loads((reference / "manifest.json").read_text())["alignment_sha256"],
                "different-alignment",
            )
            manifest["alignment_sha256"] = data["alignment_sha256"]
            manifest["code_revision"] = "other-commit"
            bootstrap_fit.atomic_json(reference / "manifest.json", manifest)
            with self.assertRaises(RuntimeError):
                bootstrap_fit.require_reference(data)
            manifest["code_revision"] = data["code_revision"]
            bootstrap_fit.atomic_json(reference / "manifest.json", manifest)
            with self.assertRaises(RuntimeError):
                bootstrap_fit.require_reference(data)
            original = bootstrap_fit.SITE_COUNT
            bootstrap_fit.SITE_COUNT = 2
            try:
                manifest["sites"] = 2
                bootstrap_fit.atomic_json(reference / "manifest.json", manifest)
                for name in ("fields.npy", "directional.npy", "h_hat.npy", "j_star.npy"):
                    array = np.zeros((2, 21)) if name == "h_hat.npy" else np.zeros((2, 2, 21, 21))
                    if name in ("fields.npy", "directional.npy"):
                        array = np.zeros(1)
                    np.save(reference / name, array)
                recovered, averaged = bootstrap_fit.require_reference(data)
                self.assertEqual(recovered.shape, (2, 21))
                self.assertEqual(averaged.shape, (2, 2, 21, 21))
                broken = np.zeros((2, 2, 21, 21))
                broken[0, 1, 0, 0] = 1.0
                np.save(reference / "j_star.npy", broken)
                with self.assertRaises(RuntimeError):
                    bootstrap_fit.require_reference(data)
            finally:
                bootstrap_fit.SITE_COUNT = original

    def test_count_audit_reports_every_difference(self):
        stored = np.array([2, 3, 4], dtype=np.int64)
        recomputed = np.array([2, 1, 4], dtype=np.int64)
        summary = bootstrap_fit.summarize_count_audit(stored, recomputed)
        self.assertEqual(summary["rows"], 3)
        self.assertEqual(summary["n_mismatch"], 1)
        self.assertEqual(summary["max_abs_difference"], 2)
        self.assertEqual(summary["mismatch_rows"], [1])
        self.assertEqual(summary["stored_m_eff"], float(np.sum(np.float64(1.0) / stored)))
        self.assertEqual(summary["recomputed_m_eff"], float(np.sum(np.float64(1.0) / recomputed)))
        matched = bootstrap_fit.summarize_count_audit(stored, stored)
        self.assertEqual(matched["n_mismatch"], 0)
        self.assertEqual(matched["max_abs_difference"], 0)
        with self.assertRaises(ValueError):
            bootstrap_fit.summarize_count_audit(stored, stored[:2])

    def test_later_formal_ids_do_not_redraw_a_missing_prefix(self):
        with _temporary_output():
            with self.assertRaises(RuntimeError):
                bootstrap_fit._require_prefix(5, 1)

    def test_stage_decision_uses_preassigned_ids_and_rejects_bad_files(self):
        with _temporary_output():
            field = np.array([1.0, 2.0])
            coupling = np.ones((1, 1))
            indices_seed = 11
            for replicate_id in range(1, 101):
                _plant_formal(replicate_id, indices_seed, True, field, coupling)
            data = bootstrap_fit.data_record()
            bootstrap_fit.atomic_json(
                bootstrap_fit.OUTPUT / "formal" / "seed.json",
                {
                    "formal_seed": indices_seed,
                    "alignment_sha256": data["alignment_sha256"],
                    "neighbor_counts_sha256": data["neighbor_counts_sha256"],
                    "code_revision": data["code_revision"],
                },
            )
            outcome = bootstrap_fit.decide_formal_stage(100, indices_seed, field, coupling, data)
            self.assertEqual(outcome["action"], "pass")
            self.assertEqual(outcome["success_count"], 100)
            changed = bootstrap_fit._formal_directory(1) / "manifest.json"
            changed_manifest = json.loads(changed.read_text())
            changed_manifest["code_revision"] = "not-the-lock"
            changed.write_text(json.dumps(changed_manifest))
            with self.assertRaises(RuntimeError):
                bootstrap_fit.load_formal_samples(100, indices_seed, data)
            changed_manifest["code_revision"] = data["code_revision"]
            changed.write_text(json.dumps(changed_manifest))
            directory = bootstrap_fit._formal_directory(2)
            np.save(directory / "h_hat.npy", np.zeros((2, 2, 1, 1)))
            broken = np.zeros((2, 2, 1, 1))
            broken[0, 1, 0, 0] = 5.0
            np.save(directory / "j_star.npy", broken)
            manifest = json.loads((directory / "manifest.json").read_text())
            manifest["replicate_success"] = False
            manifest["entered_formal_statistics"] = False
            (directory / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(RuntimeError):
                bootstrap_fit.load_formal_samples(100, indices_seed, data)
            manifest["replicate_success"] = True
            manifest["entered_formal_statistics"] = True
            (directory / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(RuntimeError):
                bootstrap_fit.load_formal_samples(100, indices_seed, data)


def _write_stage_decision(action: str, seed: int) -> None:
    directory = bootstrap_fit.OUTPUT / "formal"
    directory.mkdir(parents=True, exist_ok=True)
    identity = {
        "code_revision": "rev",
        "alignment_sha256": "a",
        "neighbor_counts_sha256": "b",
    }
    bootstrap_fit.atomic_json(
        directory / "stage_100.json",
        {"action": action, "formal_seed": seed, "stage": 100, **identity},
    )
    bootstrap_fit.atomic_json(directory / "seed.json", {"formal_seed": seed, **identity})


def _plant_formal(replicate_id: int, seed: int, success: bool, field: np.ndarray, coupling: np.ndarray) -> None:
    directory = bootstrap_fit._formal_directory(replicate_id)
    directory.mkdir(parents=True, exist_ok=True)
    indices = np.full(bootstrap_fit.DRAW_COUNT, replicate_id, dtype=np.int64)
    identity = bootstrap_fit.make_identity("formal", seed, replicate_id, indices)
    manifest = {
        **identity,
        "state": "complete",
        "formal_seed": seed,
        "formal_replicate_id": replicate_id,
        "replicate_success": success,
        "entered_formal_statistics": success,
        "failed_sites": [] if success else [0],
    }
    bootstrap_fit.ensure_indices(directory, indices)
    bootstrap_fit.atomic_json(directory / "manifest.json", manifest)
    if success:
        for name in bootstrap_fit.GAUGE_FILES:
            array = field if name == "h_hat.npy" else coupling if name == "j_star.npy" else np.zeros(1)
            bootstrap_fit.atomic_npy(directory / name, array)


if __name__ == "__main__":
    unittest.main()

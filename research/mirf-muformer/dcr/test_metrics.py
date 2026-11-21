"""Synthetic checks for the paired metric program."""

from __future__ import annotations

import inspect
import math
import unittest

import numpy as np

import metrics


class MetricTests(unittest.TestCase):
    def test_k_for_the_stiffler_test_file_is_100(self) -> None:
        self.assertEqual(metrics.top_fraction_k(999), 100)
        self.assertEqual(metrics.top_fraction_k(999), math.ceil(0.1 * 999))
        self.assertEqual(inspect.signature(metrics.evaluate_paired).parameters["n_bootstrap"].default, 10000)

    def test_ties_break_by_sequence_and_ndcg_uses_the_supplied_minmax(self) -> None:
        order = metrics.ranking_order(np.array([1.0, 1.0, 0.0]), np.array(["b", "a", "c"]))
        self.assertEqual(order.tolist(), [1, 0, 2])
        y = np.array([4.0, 6.0])
        sequences = np.array(["a", "b"])
        scores = np.array([1.0, 0.0])
        fixed = metrics.ndcg_at_k(y, scores, sequences, 0.0, 10.0, 1)
        recomputed = metrics.ndcg_at_k(y, scores, sequences, 4.0, 6.0, 1)
        self.assertAlmostEqual(fixed, (2.0 ** 0.4 - 1.0) / (2.0 ** 0.6 - 1.0))
        self.assertEqual(recomputed, 0.0)
        self.assertNotEqual(fixed, recomputed)
        perfect = metrics.ndcg_at_k(np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array(["a", "b"]), 0.0, 1.0, 1)
        self.assertEqual(perfect, 1.0)
        self.assertEqual(metrics.topk_recall(np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array(["a", "b"]), 1), 1.0)
        self.assertEqual(metrics.topk_recall(np.array([0.0, 1.0]), np.array([1.0, 0.0]), np.array(["a", "b"]), 1), 0.0)

    def test_rmse_direction_does_not_replace_spearman(self) -> None:
        y = np.arange(12, dtype=np.float64)
        sequences = np.array([f"v{index:02d}" for index in range(12)])
        baseline = np.stack([y + 5.0 for _ in range(10)])
        dcr = np.stack([y for _ in range(10)])
        report = metrics.evaluate_paired(y, sequences, baseline, dcr, n_bootstrap=30, bootstrap_seed=1)
        self.assertGreater(report["metrics"]["rmse"]["mean_delta"], 0.0)
        self.assertEqual(report["metrics"]["spearman"]["mean_delta"], 0.0)
        self.assertFalse(report["spearman_passes"])
        self.assertFalse(report["formal_experiment"])

    def test_bootstrap_matches_one_shared_draw_and_the_fixed_minmax(self) -> None:
        y = np.array([0.0, 4.0, 6.0, 10.0, 2.0, 8.0, 1.0, 9.0, 3.0, 7.0])
        sequences = np.array([f"s{index:02d}" for index in range(y.shape[0])])
        generator = np.random.default_rng(2)
        baseline = generator.normal(size=(10, y.shape[0]))
        dcr = y[None, :] + 0.05 * generator.normal(size=(10, y.shape[0]))
        report = metrics.evaluate_paired(y, sequences, baseline, dcr, n_bootstrap=25, bootstrap_seed=3)
        self.assertEqual(report["label_min"], float(np.min(y)))
        self.assertEqual(report["label_max"], float(np.max(y)))
        self.assertEqual(report["k"], metrics.top_fraction_k(y.shape[0]))
        rng = np.random.default_rng(3)
        y_min = float(np.min(y))
        y_max = float(np.max(y))
        k = report["k"]
        boot = []
        for _ in range(25):
            seed_draw = rng.integers(0, 10, size=10)
            item_draw = rng.integers(0, y.shape[0], size=y.shape[0])
            deltas = []
            for seed_index in seed_draw.tolist():
                base_value = metrics.ndcg_at_k(y[item_draw], baseline[seed_index, item_draw], sequences[item_draw], y_min, y_max, k)
                dcr_value = metrics.ndcg_at_k(y[item_draw], dcr[seed_index, item_draw], sequences[item_draw], y_min, y_max, k)
                deltas.append(dcr_value - base_value)
            boot.append(float(np.mean(deltas)))
        low, high = np.quantile(np.asarray(boot), [0.025, 0.975])
        self.assertAlmostEqual(report["metrics"]["ndcg"]["ci_low"], float(low))
        self.assertAlmostEqual(report["metrics"]["ndcg"]["ci_high"], float(high))

    def test_spearman_three_sentences_pass_only_together(self) -> None:
        y = np.arange(20, dtype=np.float64)
        sequences = np.array([f"s{index:02d}" for index in range(20)])
        good = metrics.evaluate_paired(
            y,
            sequences,
            np.stack([-y for _ in range(10)]),
            np.stack([y for _ in range(10)]),
            n_bootstrap=40,
            bootstrap_seed=0,
        )
        self.assertGreater(good["metrics"]["spearman"]["mean_delta"], 0.0)
        self.assertGreater(good["metrics"]["spearman"]["ci_low"], 0.0)
        self.assertGreaterEqual(good["metrics"]["spearman"]["n_improved"], 8)
        self.assertTrue(good["spearman_passes"])
        bad = metrics.evaluate_paired(
            y,
            sequences,
            np.stack([y for _ in range(10)]),
            np.stack([-y for _ in range(10)]),
            n_bootstrap=40,
            bootstrap_seed=0,
        )
        self.assertFalse(bad["spearman_passes"])
        self.assertEqual(metrics.paired_delta("rmse", 2.0, 1.0), 1.0)
        self.assertEqual(metrics.paired_delta("spearman", 0.2, 0.5), 0.3)


if __name__ == "__main__":
    unittest.main()

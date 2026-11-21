"""Paired metrics for baseline and DCR scores.

The Spearman rule is the three sentences in the alignment record. Pearson,
NDCG@10%, Top-10% Recall and RMSE are reported beside it. This module does
not read an experiment directory and does not decide that a trained model
improved.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import pearsonr, spearmanr

N_TEST_STIFLER = 999
FORMAL_BOOTSTRAP = 10000
SPEARMAN_IMPROVED_SEEDS = 8
SPEARMAN_SEED_COUNT = 10


def top_fraction_k(n_test: int) -> int:
    """``K = ceil(0.1 N_test)``. For the Stiffler test file this is 100."""
    if int(n_test) < 1:
        raise ValueError("N_test must be positive")
    return int(math.ceil(0.1 * int(n_test)))


def ranking_order(scores, sequences) -> np.ndarray:
    """Higher score first. Ties break by the sequence, then by row position."""
    values = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(sequences)
    if labels.shape[0] != values.shape[0]:
        raise ValueError("sequences and scores have different lengths")
    return np.lexsort((np.arange(values.shape[0]), labels, -values))


def _mapped_relevance(y, y_min: float, y_max: float) -> np.ndarray:
    labels = np.asarray(y, dtype=np.float64)
    if float(y_max) == float(y_min):
        return np.zeros(labels.shape, dtype=np.float64)
    return (labels - float(y_min)) / (float(y_max) - float(y_min))


def _dcg(relevance, scores, sequences, k: int) -> float:
    order = ranking_order(scores, sequences)[: int(k)]
    chosen = np.asarray(relevance, dtype=np.float64)[order]
    ranks = np.arange(1, chosen.shape[0] + 1, dtype=np.float64)
    gain = np.power(2.0, chosen) - 1.0
    discount = np.log2(ranks + 1.0)
    return float(np.sum(gain / discount))


def ndcg_at_k(y, scores, sequences, y_min: float, y_max: float, k: int) -> float:
    """NDCG with a caller-supplied label min and max, gain ``2^rel-1`` and ``log2(rank+1)``."""
    relevance = _mapped_relevance(y, y_min, y_max)
    actual = _dcg(relevance, scores, sequences, k)
    ideal = _dcg(relevance, relevance, sequences, k)
    if ideal == 0.0:
        return 1.0 if actual == 0.0 else 0.0
    return actual / ideal


def topk_recall(y, scores, sequences, k: int) -> float:
    """Size of the intersection of the two top-``K`` sets, divided by ``K``."""
    predicted = set(ranking_order(scores, sequences)[: int(k)].tolist())
    truth = set(ranking_order(y, sequences)[: int(k)].tolist())
    return float(len(predicted & truth) / int(k))


def rmse(y, scores) -> float:
    labels = np.asarray(y, dtype=np.float64)
    pred = np.asarray(scores, dtype=np.float64)
    return float(np.sqrt(np.mean((labels - pred) ** 2)))


def _spearman(y, scores) -> float:
    labels = np.asarray(y, dtype=np.float64)
    pred = np.asarray(scores, dtype=np.float64)
    if labels.size < 2 or np.unique(labels).size < 2 or np.unique(pred).size < 2:
        return float("nan")
    value = spearmanr(labels, pred).correlation
    if value is None or not np.isfinite(value):
        return float("nan")
    return float(value)


def _pearson(y, scores) -> float:
    labels = np.asarray(y, dtype=np.float64)
    pred = np.asarray(scores, dtype=np.float64)
    if labels.size < 2 or np.unique(labels).size < 2 or np.unique(pred).size < 2:
        return float("nan")
    value = pearsonr(labels, pred).statistic
    if not np.isfinite(value):
        return float("nan")
    return float(value)


def metric_value(name: str, y, scores, sequences, y_min: float, y_max: float, k: int) -> float:
    if name == "spearman":
        return _spearman(y, scores)
    if name == "pearson":
        return _pearson(y, scores)
    if name == "ndcg":
        return ndcg_at_k(y, scores, sequences, y_min, y_max, k)
    if name == "recall":
        return topk_recall(y, scores, sequences, k)
    if name == "rmse":
        return rmse(y, scores)
    raise ValueError(f"unknown metric {name}")


def paired_delta(name: str, baseline_value: float, dcr_value: float) -> float:
    """DCR minus baseline, except RMSE, which is baseline minus DCR."""
    if name == "rmse":
        return float(baseline_value) - float(dcr_value)
    return float(dcr_value) - float(baseline_value)


METRIC_NAMES = ("spearman", "pearson", "ndcg", "recall", "rmse")


def evaluate_paired(
    y,
    sequences,
    baseline,
    dcr,
    *,
    n_bootstrap: int = FORMAL_BOOTSTRAP,
    bootstrap_seed: int = 0,
) -> dict:
    """Point estimates on the full test rows, then a paired bootstrap of their mean difference.

    ``n_bootstrap`` defaults to 10000. Min and max of ``y`` are computed once.
    Each bootstrap draw resamples seeds, then resamples ``N_test`` rows once;
    baseline and DCR use that same row draw. ``K`` stays ``ceil(0.1 N_test)``
    of the full test file. The Spearman pass flag is the three recorded sentences
    and is not a claim about a trained model.
    """
    labels = np.asarray(y, dtype=np.float64)
    names = np.asarray(sequences)
    base = np.asarray(baseline, dtype=np.float64)
    fused = np.asarray(dcr, dtype=np.float64)
    if labels.ndim != 1:
        raise ValueError("labels must be one test-file vector")
    if names.shape != labels.shape:
        raise ValueError("each test row needs its sequence")
    if base.shape != fused.shape or base.ndim != 2 or base.shape[1] != labels.shape[0]:
        raise ValueError("baseline and DCR must be seeds by test rows")
    if int(n_bootstrap) < 1:
        raise ValueError("the bootstrap count must be positive")
    n_seeds = int(base.shape[0])
    n_test = int(labels.shape[0])
    k = top_fraction_k(n_test)
    y_min = float(np.min(labels))
    y_max = float(np.max(labels))
    point = {name: np.empty(n_seeds, dtype=np.float64) for name in METRIC_NAMES}
    for seed_index in range(n_seeds):
        for name in METRIC_NAMES:
            base_value = metric_value(name, labels, base[seed_index], names, y_min, y_max, k)
            dcr_value = metric_value(name, labels, fused[seed_index], names, y_min, y_max, k)
            point[name][seed_index] = paired_delta(name, base_value, dcr_value)
    rng = np.random.default_rng(int(bootstrap_seed))
    boot = {name: np.empty(int(n_bootstrap), dtype=np.float64) for name in METRIC_NAMES}
    for draw in range(int(n_bootstrap)):
        seed_draw = rng.integers(0, n_seeds, size=n_seeds)
        item_draw = rng.integers(0, n_test, size=n_test)
        drawn_y = labels[item_draw]
        drawn_names = names[item_draw]
        for name in METRIC_NAMES:
            deltas = np.empty(n_seeds, dtype=np.float64)
            for position, seed_index in enumerate(seed_draw.tolist()):
                base_value = metric_value(name, drawn_y, base[seed_index, item_draw], drawn_names, y_min, y_max, k)
                dcr_value = metric_value(name, drawn_y, fused[seed_index, item_draw], drawn_names, y_min, y_max, k)
                deltas[position] = paired_delta(name, base_value, dcr_value)
            boot[name][draw] = float(np.mean(deltas))
    report = {}
    for name in METRIC_NAMES:
        deltas = point[name]
        low, high = np.quantile(boot[name], [0.025, 0.975])
        improved = int(np.sum(np.isfinite(deltas) & (deltas > 0.0)))
        report[name] = {
            "mean_delta": float(np.mean(deltas)),
            "std_delta": float(np.std(deltas, ddof=1)) if n_seeds > 1 else float("nan"),
            "ci_low": float(low),
            "ci_high": float(high),
            "n_improved": improved,
            "deltas": [float(value) for value in deltas.tolist()],
        }
    spearman = report["spearman"]
    spearman_passes = bool(
        n_seeds == SPEARMAN_SEED_COUNT
        and spearman["mean_delta"] > 0.0
        and spearman["ci_low"] > 0.0
        and spearman["n_improved"] >= SPEARMAN_IMPROVED_SEEDS
    )
    return {
        "n_seeds": n_seeds,
        "n_test": n_test,
        "k": k,
        "n_bootstrap": int(n_bootstrap),
        "label_min": y_min,
        "label_max": y_max,
        "spearman_passes": spearman_passes,
        "metrics": report,
        "formal_experiment": False,
    }

"""Attribute large fixed-topology gradients without fitting or generating observations."""
from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal, localcontext
import gzip
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.stats import spearmanr

from .check_f81_direction_scale import curvature_is_usable, reconstruct_direction
from .check_f81_observed_bridge import load_case
from .check_f81_split_gradient import Reference
from .f81_optimize import projected_norm


def recover_warm_terminal(engine, point, archive, record):
    """Re-evaluate previously accepted steps only, checking their recorded direction."""
    assert record["search"]["history_resets"] == 0
    events = [json.loads(x) for x in archive.read_text().splitlines()]
    searches = {x["search"]: x for x in events if x["event"] == "line_search"}
    evals = {x["call"]: x for x in events if x["event"] == "evaluation"}
    calls = [x["calls"] for x in events if x["event"] == "accepted"]
    trials = [x for x in events if x.get("outcome") == "accepted"]
    assert len(trials) == len(calls) == record["search"]["iterations"]
    n = len(engine.pruning.states)
    history = []
    current = engine.evaluate(point, threads=4)
    assert current.log_likelihood == evals[1]["log_likelihood"]
    for iteration, (trial, call) in enumerate(zip(trials, calls), 1):
        g = -current.length_gradient / n
        direction, _, details = reconstruct_direction(point, g, history)
        archived = searches[trial["search"]]
        assert archived["iteration"] == iteration - 1
        for key in ("directional_derivative", "direction_inf", "direction_l2"):
            assert details[key] == archived[key]
        next_point = np.maximum(0, point + trial["step"] * direction)
        delta = next_point - point
        assert float(g @ delta) == trial["projected_directional_derivative"]
        next_call = engine.evaluate(next_point, threads=4)
        assert next_call.log_likelihood == evals[call]["log_likelihood"]
        change = -next_call.length_gradient/n - g
        sy = float(delta @ change)
        if curvature_is_usable(delta, change, sy):
            history.append((iteration, delta, change, sy))
            history = history[-10:]
        current, point = next_call, next_point
    return point, len(calls) + 1


def leaves_on_child_side(engine):
    t = engine.topology
    counts = np.zeros(len(t.layout.parent) + 1, dtype=np.int64)
    counts[:t.layout.leaves] = 1
    for child, parent in enumerate(t.layout.parent):
        counts[parent] += counts[child]
    lengths_order = np.empty(len(t.child_to_edge), dtype=np.int64)
    lengths_order[t.child_to_edge] = counts[:-1]
    assert counts[-1] == t.layout.leaves
    return lengths_order


class StructuralSupport:
    """Check exact support from paths made exclusively of equality constraints."""

    def __init__(self, leaves, pairs, lengths, states):
        self.leaves, self.pairs, self.states = leaves, pairs, states
        self.parent = np.arange(len(pairs) + 1)
        self.size = np.ones(len(pairs) + 1, dtype=int)
        for i in np.flatnonzero(lengths == 0):
            a, b = map(self.find, pairs[i])
            if a == b:
                continue
            if self.size[a] < self.size[b]:
                a, b = b, a
            self.parent[b] = a
            self.size[a] += self.size[b]
        self.labels = {}
        for leaf in range(leaves):
            root = self.find(leaf)
            if root in self.labels:
                assert np.array_equal(self.labels[root], states[:, leaf])
            else:
                self.labels[root] = states[:, leaf]

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def a_is_zero(self, edge):
        a, b = map(self.find, self.pairs[edge])
        if a == b or a not in self.labels or b not in self.labels:
            return np.zeros(len(self.states), dtype=bool)
        return self.labels[a] != self.labels[b]


def precision_edge(reference, support, edge, length, left_count, selection, ordinary, logged):
    names = ("A", "B", "mass", "gradient", "diagonal_curvature")
    zero = support.a_is_zero(edge)
    started = perf_counter()
    mean192, s192 = reference.site_terms(edge, 192)
    mean256, s256 = reference.site_terms(edge, 256)
    leaf_count = reference.pruning.states.shape[1]
    with localcontext() as context:
        context.prec = 115
        count = len(s256)
        total = sum(t[3] for t in s256)
        norm = sum(abs(t[3]) for t in s256)
        diagonal = sum(t[4] for t in s256) / count
        assert abs(total/count - mean256) <= max(Decimal("1e-70"), abs(mean256)*Decimal("1e-71"))
        assert all(t[1] > 0 and (t[0] == 0) == expected and (r[0] == 0) == expected
                   for t, r, expected in zip(s256, s192, zero))
        exact_length = Decimal.from_float(length)
        redraw_ratio = exact_length.exp() - 1
        measures = []
        groups = Counter()
        absolute = Counter()
        for i, (low, row) in enumerate(zip(s192, s256)):
            a, b, mass, g, curvature = row
            ratio = a / b
            if zero[i]:
                label = "structural_A_zero"
            elif exact_length > 0 and ratio <= redraw_ratio:
                label = "positive_A_redraw_dominated"
            elif ratio <= Decimal("1e-12"):
                label = "positive_A_tiny_ratio"
            else:
                label = "other_positive_A"
            groups[label] += 1
            absolute[label] += abs(g)
            measures.append({"site_index": i, "classification": label, "A_over_B": str(ratio),
                             "A_positive_but_rounds_to_zero_in_binary64": bool(a > 0 and float(a) == 0.),
                             "bits192": dict(zip(names, map(str, low))),
                             "bits256": dict(zip(names, map(str, row))),
                             "absolute_gradient_fraction": float(abs(g)/norm) if norm else 0.})
        ordered = sorted(measures, key=lambda r: r["absolute_gradient_fraction"], reverse=True)
        rounded = Decimal.from_float(float(mean256))
        return {"edge_index": edge, "selection": selection, "branch_length": length,
                "left_leaves": left_count, "right_leaves": leaf_count - left_count,
                "structural_zero_support_check_passed": True,
                "site_class_counts": dict(groups),
                "absolute_gradient_fraction_by_class": {k: float(v/norm) if norm else 0. for k, v in absolute.items()},
                "summed_site_gradient_256bit": str(total), "mean_gradient_192bit": str(mean192),
                "mean_gradient_256bit": str(mean256), "mean_local_diagonal_curvature_256bit": str(diagonal),
                "mean_cross_precision_abs_difference": str(abs(mean192 - mean256)),
                "ordinary_gradient": ordinary, "rerooted_log_gradient": logged,
                "cross_path_gradient_abs_difference": abs(ordinary - logged),
                "ordinary_abs_error_vs_reference": str(abs(Decimal.from_float(ordinary) - mean256)),
                "rerooted_log_abs_error_vs_reference": str(abs(Decimal.from_float(logged) - mean256)),
                "nearest_binary64_abs_representation_error": str(abs(rounded - mean256)),
                "top_site_indices": [r["site_index"] for r in ordered[:5]],
                "top_one_absolute_gradient_fraction": ordered[0]["absolute_gradient_fraction"],
                "top_five_absolute_gradient_fraction": sum(r["absolute_gradient_fraction"] for r in ordered[:5]),
                "elapsed_seconds": perf_counter() - started, "site_terms": measures}


def summarize_groups(point, normal, agreement, side, n):
    bins = [(0, 0), (0, 1e-20), (1e-20, 1e-12), (1e-12, 1e-6), (1e-6, 1e-2), (1e-2, np.inf)]
    strata = []
    for low, high in bins:
        select = (point == 0) if high == 0 else ((point > low) & (point <= high))
        if select.any():
            strata.append({"length_lower_exclusive": low, "length_upper_inclusive": high if np.isfinite(high) else None,
                           "count": int(select.sum()), "abs_gradient_median": float(np.median(np.abs(normal[select]))),
                           "abs_gradient_max": float(np.abs(normal[select]).max()),
                           "cross_path_difference_max": float(agreement[select].max())})
    smaller = np.minimum(side, n - side)
    positive = point > 0
    return {"length_strata": strata,
            "spearman_abs_gradient_vs_smaller_subtree": float(spearmanr(np.abs(normal), smaller).statistic),
            "spearman_abs_gradient_vs_positive_length": float(spearmanr(np.abs(normal[positive]), point[positive]).statistic)}


def run(args):
    engine, auditor, facts = load_case(args.prep, args.gate, args.rows, args.library)
    point = np.load(args.checkpoint, allow_pickle=False)
    original = json.loads(args.manifest.read_text())
    for key in ("alignment_weights_sha256", "indices_sha256", "reference_profile_sha256", "topology_sha256", "library_sha256"):
        assert facts[key] == original[key]
    replay = 0
    if args.warm_trace:
        assert hashlib.sha256(point.tobytes()).hexdigest() == original["initial_sha256"]
        point, replay = recover_warm_terminal(engine, point, args.warm_trace, original)
    fingerprint = hashlib.sha256(point.tobytes()).hexdigest()
    assert fingerprint == (original.get("terminal_sha256") if args.warm_trace else original["checkpoint_sha256"])

    n = facts["sites"]
    normal = -engine.evaluate(point, threads=4).length_gradient / n
    logged = -auditor.evaluate(point, threads=4, log_domain=True).length_gradient / n
    agreement = np.abs(normal - logged)
    counts = leaves_on_child_side(engine)
    zero = point == 0
    residual = point - np.maximum(0, point - normal)
    roles = [("max_abs_gradient", int(np.abs(normal).argmax())),
             ("max_projected_residual", int(np.abs(residual).argmax())),
             ("max_cross_path_difference", int(agreement.argmax()))]
    positive = np.flatnonzero(~zero)
    if len(positive):
        roles.append(("shortest_positive", int(positive[np.argmin(point[positive])])))
    if zero.any():
        indices = np.flatnonzero(zero)
        roles.append(("largest_gradient_zero_edge", int(indices[np.argmax(np.abs(normal[zero]))])))
    ref = Reference(engine, point, args.reference)
    support = StructuralSupport(facts["sample_size"], ref.pairs, point, engine.pruning.states)
    top = np.argsort(-np.abs(normal), kind="stable")[:10]
    # Full scalar per-edge ledger remains on the compute host; all audited
    # precision terms and the rank/length summaries are independently published.
    with gzip.open(args.edge_ledger, "wt", encoding="utf-8") as log:
        for i in range(len(point)):
            log.write(json.dumps({"edge": i, "length": float(point[i]), "smaller_side_leaves": int(min(counts[i], facts["sample_size"]-counts[i])),
                                  "ordinary_gradient": float(normal[i]), "rerooted_log_gradient": float(logged[i]),
                                  "projected_residual": float(residual[i])}, allow_nan=False) + "\n")

    cases = {}
    for label, edge in roles:
        cases.setdefault(edge, []).append(label)
    selected = []
    for edge, label in cases.items():
        left = int(counts[edge])
        selected.append(precision_edge(ref, support, edge, float(point[edge]), left, label,
                                       float(normal[edge]), float(logged[edge])))
    return {**facts, "checkpoint_sha256": fingerprint, "replay_evaluations": replay,
            "all_edge_scan_evaluations": 2, "edges_scanned": len(point),
            "edge_ledger_sha256": hashlib.sha256(args.edge_ledger.read_bytes()).hexdigest(),
            "zero_edges": int(zero.sum()), "active_edges": int(np.count_nonzero(zero & (normal >= 0))),
            "ordinary_projected_gradient_inf": projected_norm(point, normal),
            "rerooted_projected_gradient_inf": projected_norm(point, logged),
            "cross_path_gradient_max_abs_difference": float(agreement.max()),
            "passes_frozen_absolute_gradient_gate": bool(agreement.max() <= 1e-8),
            "top_absolute_gradient_edges": [{"edge": int(i), "length": float(point[i]),
                                             "smaller_side_leaves": int(min(counts[i], facts["sample_size"]-counts[i])),
                                             "ordinary_gradient": float(normal[i])} for i in top],
            **summarize_groups(point, normal, agreement, counts, facts["sample_size"]),
            "selected_edge_site_attributions": selected}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, choices=(1000, 10000), required=True)
    for key in ("prep", "gate", "library", "reference", "checkpoint", "manifest", "edge-ledger", "warm-trace"):
        parser.add_argument("--" + key, type=Path, required=key != "warm-trace")
    args = parser.parse_args()
    print(json.dumps(run(args), allow_nan=False, indent=2))


if __name__ == "__main__":
    main()

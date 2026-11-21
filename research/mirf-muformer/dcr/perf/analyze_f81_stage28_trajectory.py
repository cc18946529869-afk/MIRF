"""Per-edge projected-gradient trajectory of the 208 accepted Stage 2.8 steps.

Gradients are the ones stored when each step was accepted. This does not
rerun the likelihood. The projected residual is the existing definition
r = s - max(0, s - g).
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def residual(lengths, gradient):
    return lengths - np.maximum(0., lengths - gradient)


def load_step(path):
    saved = np.load(path)
    lengths = np.ascontiguousarray(saved["lengths"], dtype=np.float64)
    gradient = np.ascontiguousarray(saved["physical_gradient"], dtype=np.float64)
    diagonal = np.ascontiguousarray(saved["hessian_diagonal"], dtype=np.float64)
    return {
        "lengths": lengths,
        "gradient": gradient,
        "diagonal": diagonal,
        "objective": float(saved["objective"]),
        "projection": float(saved["projection"]),
        "iteration": int(saved["iteration"]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=208)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    states = [load_step(args.archive / "initial.npz")]
    for step in range(1, args.steps + 1):
        states.append(load_step(args.archive / f"accepted-{step:04d}.npz"))
    edges = states[0]["lengths"].size
    top_counts = np.zeros(edges, dtype=np.int32)
    max_counts = np.zeros(edges, dtype=np.int32)
    rows = []
    previous_top = None
    max_ids = []
    for index, state in enumerate(states):
        absolute = np.abs(residual(state["lengths"], state["gradient"]))
        order = np.argsort(absolute)[::-1]
        top10 = order[:10]
        top50 = order[:50]
        top_counts[top50] += 1
        peak = int(order[0])
        max_counts[peak] += 1
        max_ids.append(peak)
        active = int(np.count_nonzero((state["lengths"] == 0) & (state["gradient"] >= 0)))
        overlap = "" if previous_top is None else f"{len(set(top10.tolist()) & previous_top) / 10:.2f}"
        previous_top = set(top10.tolist())
        if index == 0:
            changed = 0
            length_rms = 0.
            objective_change = 0.
            slope = 0.
        else:
            before = states[index - 1]
            delta = state["lengths"] - before["lengths"]
            changed = int(np.count_nonzero(delta != 0))
            length_rms = float(np.sqrt(np.mean(delta * delta)))
            objective_change = state["objective"] - before["objective"]
            slope = float(before["gradient"] @ delta)
        stored = state["projection"]
        recomputed = float(absolute.max())
        rows.append({
            "step": state["iteration"],
            "objective": state["objective"],
            "objective_change": objective_change,
            "projection_stored": stored,
            "projection": recomputed,
            "projection_abs_difference": abs(recomputed - stored),
            "peak_edge": peak,
            "peak_length": float(state["lengths"][peak]),
            "peak_gradient": float(state["gradient"][peak]),
            "peak_diagonal": float(state["diagonal"][peak]),
            "peak_abs_residual": float(absolute[peak]),
            "edges_above_1e-3": int(np.count_nonzero(absolute >= 1e-3)),
            "edges_above_1e-4": int(np.count_nonzero(absolute >= 1e-4)),
            "edges_above_1e-5": int(np.count_nonzero(absolute >= 1e-5)),
            "median_abs_residual": float(np.median(absolute)),
            "active_boundary": active,
            "zero_edges": int(np.count_nonzero(state["lengths"] == 0)),
            "changed_edges": changed,
            "length_rms": length_rms,
            "slope_previous_gradient": slope,
            "top10_overlap_with_previous": overlap,
            "top10": ",".join(str(int(edge)) for edge in top10),
        })
    with (args.out / "trajectory.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in writer.fieldnames})
    frequent = np.argsort(top_counts)[::-1]
    with (args.out / "top50_frequency.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["edge_id", "steps_in_top50", "steps_as_maximum"])
        for edge in frequent:
            if top_counts[edge] == 0:
                break
            writer.writerow([int(edge), int(top_counts[edge]), int(max_counts[edge])])
    named = {"start": states[0], "step92": states[92], "step208": states[208]}
    with (args.out / "compare_start_92_208.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow([
            "edge_id", "length_start", "length_92", "length_208",
            "gradient_start", "gradient_92", "gradient_208",
            "abs_residual_start", "abs_residual_92", "abs_residual_208",
            "diagonal_92", "diagonal_208",
            "residual_change_92_to_208", "length_change_92_to_208"])
        start_r = np.abs(residual(named["start"]["lengths"], named["start"]["gradient"]))
        mid_r = np.abs(residual(named["step92"]["lengths"], named["step92"]["gradient"]))
        end_r = np.abs(residual(named["step208"]["lengths"], named["step208"]["gradient"]))
        interesting = np.flatnonzero((start_r >= 1e-4) | (mid_r >= 1e-4) | (end_r >= 1e-4)
                                     | (np.abs(end_r - mid_r) >= 1e-4))
        order = interesting[np.argsort(end_r[interesting] - mid_r[interesting])[::-1]]
        for edge in order:
            writer.writerow([
                int(edge),
                float(named["start"]["lengths"][edge]), float(named["step92"]["lengths"][edge]),
                float(named["step208"]["lengths"][edge]),
                float(named["start"]["gradient"][edge]), float(named["step92"]["gradient"][edge]),
                float(named["step208"]["gradient"][edge]),
                float(start_r[edge]), float(mid_r[edge]), float(end_r[edge]),
                float(named["step92"]["diagonal"][edge]), float(named["step208"]["diagonal"][edge]),
                float(end_r[edge] - mid_r[edge]),
                float(named["step208"]["lengths"][edge] - named["step92"]["lengths"][edge])])
    peak_sequence = [row["peak_edge"] for row in rows]
    switches = sum(a != b for a, b in zip(peak_sequence, peak_sequence[1:]))
    worsened = end_r - mid_r
    worsened_order = np.argsort(worsened)[::-1][:20]
    summary = {
        "steps": args.steps,
        "stored_projection_max_abs_difference": max(row["projection_abs_difference"] for row in rows),
        "unique_peak_edges": int(len(set(peak_sequence))),
        "peak_switches": int(switches),
        "most_frequent_peak": int(np.argmax(max_counts)),
        "most_frequent_peak_steps": int(max_counts.max()),
        "edges_ever_in_top50": int(np.count_nonzero(top_counts)),
        "edges_in_top50_at_least_half_the_steps": int(np.count_nonzero(top_counts >= (args.steps + 1) / 2)),
        "step92": {
            "projection": rows[92]["projection"],
            "objective": rows[92]["objective"],
            "peak_edge": rows[92]["peak_edge"],
            "edges_above_1e-3": rows[92]["edges_above_1e-3"],
            "edges_above_1e-4": rows[92]["edges_above_1e-4"],
            "median_abs_residual": rows[92]["median_abs_residual"],
        },
        "step208": {
            "projection": rows[208]["projection"],
            "objective": rows[208]["objective"],
            "peak_edge": rows[208]["peak_edge"],
            "edges_above_1e-3": rows[208]["edges_above_1e-3"],
            "edges_above_1e-4": rows[208]["edges_above_1e-4"],
            "median_abs_residual": rows[208]["median_abs_residual"],
        },
        "edges_residual_increased_92_to_208": int(np.count_nonzero(worsened > 1e-4)),
        "edges_residual_decreased_92_to_208": int(np.count_nonzero(worsened < -1e-4)),
        "largest_residual_increases_92_to_208": [
            {"edge": int(edge), "from": float(mid_r[edge]), "to": float(end_r[edge]),
             "length_92": float(named["step92"]["lengths"][edge]),
             "length_208": float(named["step208"]["lengths"][edge]),
             "diagonal_92": float(named["step92"]["diagonal"][edge])}
            for edge in worsened_order],
        "negative_slope_steps": int(sum(row["slope_previous_gradient"] < 0 for row in rows[1:])),
        "median_changed_edges": float(np.median([row["changed_edges"] for row in rows[1:]])),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in (
        "unique_peak_edges", "peak_switches", "most_frequent_peak", "most_frequent_peak_steps",
        "edges_ever_in_top50", "edges_residual_increased_92_to_208",
        "edges_residual_decreased_92_to_208", "median_changed_edges")}, indent=2))


if __name__ == "__main__":
    main()

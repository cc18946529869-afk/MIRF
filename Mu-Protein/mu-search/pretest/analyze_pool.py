"""Full-pool mutation and fitness summary of the six finished TEM-1 runs.

Reads queries.csv only. Does not rewrite a run directory and does not score.
"""

import csv
import json
import math
import os
import statistics
from collections import Counter


RUNS = os.path.join(os.path.dirname(__file__), "runs")
METHODS = ("musearch_gt", "adalead_gt")
SEEDS = (0, 1, 2)


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def median(values):
    return float(statistics.median(values))


def mutations(wild, sequence):
    changes = []
    for index, (left, right) in enumerate(zip(wild, sequence), start=1):
        if left != right:
            changes.append({"pos": index, "from": left, "to": right})
    return changes


def load_rows(path):
    with open(path, encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def summarize(path):
    rows = load_rows(path)
    starts = [row for row in rows if row["role"] == "start"]
    candidates = [row for row in rows if row["role"] == "candidate"]
    if len(starts) != 1 or len(candidates) != 4998:
        raise RuntimeError("{0}: start={1} candidates={2}".format(path, len(starts), len(candidates)))
    wild = starts[0]["sequence"]
    wild_score = float(starts[0]["score"])
    if len(wild) != 286:
        raise RuntimeError("wild-type length {0}".format(len(wild)))

    first_score = {}
    first_order = []
    repeat_calls = []
    score_mismatch = 0
    site_unique = Counter()
    site_calls = Counter()
    substitution_unique = Counter()
    hamming_calls = Counter()
    hamming_unique = Counter()
    by_hamming = {}
    improvements = []
    best_so_far = wild_score
    half = {"first": [], "second": []}

    for row in candidates:
        sequence = row["sequence"]
        score = float(row["score"])
        index = int(row["candidate_index"])
        if len(sequence) != 286:
            raise RuntimeError("candidate length {0}".format(len(sequence)))
        distance = sum(left != right for left, right in zip(wild, sequence))
        hamming_calls[distance] += 1
        which = "first" if index <= 2499 else "second"
        bucket = half[which]
        bucket.append((distance, score, sequence == wild))
        if sequence in first_score:
            if abs(first_score[sequence] - score) > 1e-8:
                score_mismatch += 1
            repeat_calls.append(score)
        else:
            first_score[sequence] = score
            first_order.append(sequence)
            hamming_unique[distance] += 1
            changes = mutations(wild, sequence)
            for change in changes:
                site_unique[change["pos"]] += 1
                substitution_unique[(change["from"], change["pos"], change["to"])] += 1
            slot = by_hamming.setdefault(distance, [])
            slot.append(score)
        for change_pos in (i + 1 for i, (left, right) in enumerate(zip(wild, sequence)) if left != right):
            site_calls[change_pos] += 1
        if score > best_so_far:
            best_so_far = score
            improvements.append(
                {
                    "candidate_index": index,
                    "half": which,
                    "hamming": distance,
                    "score": score,
                    "sequence": sequence,
                }
            )

    best_score = max(first_score.values())
    best_sequences = [sequence for sequence in first_order if first_score[sequence] == best_score]
    best_details = []
    for sequence in best_sequences:
        changes = mutations(wild, sequence)
        best_details.append(
            {
                "score": best_score,
                "hamming": len(changes),
                "mutations": [
                    "{0}{1}{2}".format(change["from"], change["pos"], change["to"])
                    for change in changes
                ],
            }
        )

    def half_summary(items):
        distances = [item[0] for item in items]
        scores = [item[1] for item in items]
        return {
            "n": len(items),
            "mean_hamming": sum(distances) / len(distances),
            "median_hamming": median(distances),
            "mean_fitness": sum(scores) / len(scores),
            "max_fitness": max(scores),
            "wt_calls": sum(1 for item in items if item[2]),
        }

    fitness_by_hamming = {}
    for distance, scores in sorted(by_hamming.items()):
        fitness_by_hamming[str(distance)] = {
            "n_unique": len(scores),
            "mean": sum(scores) / len(scores),
            "median": median(scores),
            "max": max(scores),
            "min": min(scores),
            "frac_above_wt": sum(score > wild_score for score in scores) / len(scores),
        }

    ranked_sites = site_unique.most_common()
    rank_of = {pos: rank for rank, (pos, _count) in enumerate(ranked_sites, start=1)}
    best_site_ranks = []
    for detail in best_details:
        for token in detail["mutations"]:
            pos = int(token[1:-1])
            best_site_ranks.append(
                {
                    "mutation": token,
                    "unique_sequences": site_unique[pos],
                    "candidate_calls": site_calls[pos],
                    "rank_by_unique": rank_of.get(pos),
                }
            )

    unique_scores = [first_score[sequence] for sequence in first_order]
    unique_distances = []
    for sequence in first_order:
        unique_distances.append(sum(left != right for left, right in zip(wild, sequence)))

    return {
        "n_candidates": len(candidates),
        "wild_type_score": wild_score,
        "unique_sequences": len(first_score),
        "duplicate_calls": len(repeat_calls),
        "duplicate_rate": len(repeat_calls) / len(candidates),
        "duplicate_score_mismatches": score_mismatch,
        "duplicate_calls_that_are_wt": sum(1 for row in candidates if row["sequence"] == wild) - (1 if wild in first_score else 0),
        "wt_candidate_calls": sum(1 for row in candidates if row["sequence"] == wild),
        "mean_fitness_first_occurrence": sum(unique_scores) / len(unique_scores),
        "mean_fitness_repeat_calls": (sum(repeat_calls) / len(repeat_calls)) if repeat_calls else None,
        "mean_hamming_unique": sum(unique_distances) / len(unique_distances),
        "max_hamming": max(hamming_unique),
        "min_hamming": min(hamming_unique),
        "sites_used": len(site_unique),
        "pearson_fitness_hamming_unique": pearson(unique_distances, unique_scores),
        "hamming_hist_calls": {str(k): hamming_calls[k] for k in sorted(hamming_calls)},
        "hamming_hist_unique": {str(k): hamming_unique[k] for k in sorted(hamming_unique)},
        "fitness_by_hamming": fitness_by_hamming,
        "best": best_details,
        "best_site_ranks": best_site_ranks,
        "halves": {name: half_summary(items) for name, items in half.items()},
        "improvements": [
            {key: value for key, value in item.items() if key != "sequence"}
            for item in improvements
        ],
        "improvement_counts": {
            "first": sum(1 for item in improvements if item["half"] == "first"),
            "second": sum(1 for item in improvements if item["half"] == "second"),
        },
        "top_sites": [
            {"pos": pos, "unique_sequences": count, "candidate_calls": site_calls[pos]}
            for pos, count in ranked_sites[:15]
        ],
        "top_substitutions": [
            {
                "mutation": "{0}{1}{2}".format(src, pos, dst),
                "unique_sequences": count,
            }
            for (src, pos, dst), count in substitution_unique.most_common(15)
        ],
    }


def main():
    payload = {"runs": {}}
    for method in METHODS:
        for seed in SEEDS:
            path = os.path.join(
                RUNS, method, "seed{0}".format(seed),
                "{0}_seed{1}.queries.csv".format(method, seed),
            )
            key = "{0}_seed{1}".format(method, seed)
            payload["runs"][key] = summarize(path)
            print(key, "unique", payload["runs"][key]["unique_sequences"],
                  "best", payload["runs"][key]["best"][0]["mutations"])
    out_dir = os.path.join(os.path.dirname(__file__), "analysis")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "pool_stats.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Describe the six finished TEM-1 runs. Does not score new sequences.

Three seeds are a screen. The printed means are descriptive only.
"""

import json
import sys
from pathlib import Path

import pandas as pd

from plot_screen import summarize_one


def mutation_row(query_path):
    frame = pd.read_csv(query_path)
    start = frame.loc[frame["role"] == "start", "sequence"].iloc[0]
    candidates = frame.loc[frame["role"] == "candidate"]
    first = {}
    for sequence, score in zip(candidates["sequence"], candidates["score"]):
        sequence = str(sequence)
        if sequence not in first:
            first[sequence] = float(score)
    site_counts = [0] * len(start)
    hammings = []
    for sequence in first:
        differences = [index for index, (left, right) in enumerate(zip(start, sequence)) if left != right]
        hammings.append(len(differences))
        for index in differences:
            site_counts[index] += 1
    used = [count for count in site_counts if count]
    order = sorted(range(len(site_counts)), key=lambda index: site_counts[index], reverse=True)
    top = ["{0}:{1}".format(index + 1, site_counts[index]) for index in order[:8] if site_counts[index]]
    best_sequence = max(first, key=first.get)
    best_sites = [index + 1 for index, (left, right) in enumerate(zip(start, best_sequence)) if left != right]
    return {
        "unique_mean_hamming": sum(hammings) / len(hammings),
        "unique_median_hamming": float(pd.Series(hammings).median()),
        "mutated_unique_sequences": sum(1 for value in hammings if value > 0),
        "sites_used": len(used),
        "max_site_count": max(site_counts),
        "top_sites_1based": " ".join(top),
        "best_sequence_sites_1based": " ".join(str(site) for site in best_sites),
        "best_sequence_hamming": len(best_sites),
    }


def main(argv):
    result_dir = Path(argv[0])
    rows = []
    for query_path in sorted(result_dir.rglob("*_seed*.queries.csv")):
        stem = query_path.name.replace(".queries.csv", "")
        row = summarize_one(
            query_path,
            query_path.with_name(stem + ".budget.json"),
            query_path.with_name(stem + ".time.json"),
        )
        row.update(mutation_row(query_path))
        rows.append(row)
    table = pd.DataFrame(rows)
    destination = result_dir / "screen_summary.csv"
    table.to_csv(destination, index=False)
    columns = [
        "run",
        "best_fitness",
        "top_1pct_mean",
        "top_1pct_count",
        "search_auc",
        "gain_first_half",
        "gain_second_half",
        "unique_sequences",
        "candidate_duplicate_rate",
        "unique_mean_hamming",
        "sites_used",
        "best_sequence_hamming",
        "best_sequence_sites_1based",
        "wall_clock_search_seconds",
        "reward",
        "return_rescore",
    ]
    print(table[columns].to_string(index=False))
    print("写入 {0}".format(destination))
    grouped = table.copy()
    grouped["method"] = grouped["run"].str.replace(r"_seed[0-9]+$", "", regex=True)
    means = grouped.groupby("method")[columns[1:-2]].mean(numeric_only=True)
    print("三个种子的描述性平均，不是显著性检验")
    print(means.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

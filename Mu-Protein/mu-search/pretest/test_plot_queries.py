"""Query-level AUC rejects a two-point curve and keeps initialization out."""

import json
import os
import sys
import tempfile
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

import pandas as pd

from plot_screen import summarize_one, summarize_queries


def budget(reward):
    purposes = {
        "initialization": 1,
        "reward": reward,
        "return_rescore": 1,
        "search_measurement": 1,
        "gradient": 0,
    }
    zeros = {name: 0 for name in purposes}
    return {
        "events": {
            "muformer_oracle": purposes,
            "cnn_surrogate": dict(zeros),
            "other_landscape": dict(zeros),
            "other_surrogate": dict(zeros),
        },
        "total_by_channel": {
            "muformer_oracle": sum(purposes.values()),
            "cnn_surrogate": 0,
            "other_landscape": 0,
            "other_surrogate": 0,
        },
        "duplicate_calls": {"muformer_oracle": 0},
        "unique_sequences": {"muformer_oracle": reward},
    }


def write_run(directory, rows):
    path = directory / "musearch_gt_seed0.queries.csv"
    header = [
        "call_index",
        "purpose",
        "role",
        "sequence",
        "score",
        "t_sec",
        "model_input_sequences",
        "candidate_index",
        "candidate_best_so_far",
    ]
    lines = [",".join(header)]
    lines.extend(",".join(str(item) for item in row) for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main():
    with tempfile.TemporaryDirectory() as raw:
        directory = Path(raw)
        rows = [
            [1, "initialization", "initialization", "WT", 9.0, 0, 2, "", ""],
            [2, "search_measurement", "start", "WT", -0.07, 0, 2, "", ""],
            [3, "reward", "candidate", "AAAA", -0.05, 0, 2, 1, -0.05],
            [4, "reward", "candidate", "AAAB", -0.04, 0, 2, 2, -0.04],
            [5, "reward", "candidate", "AAAC", -0.06, 0, 2, 3, -0.04],
            [6, "reward", "candidate", "AAAD", -0.03, 0, 2, 4, -0.03],
            [7, "reward", "candidate", "AAAE", -0.08, 0, 2, 5, -0.03],
            [8, "return_rescore", "return_rescore", "AAAD", -0.03, 0, 2, "", ""],
        ]
        query_path = write_run(directory, rows)
        (directory / "musearch_gt_seed0.budget.json").write_text(
            json.dumps(budget(5)), encoding="utf-8"
        )
        (directory / "musearch_gt_seed0.time.json").write_text(
            json.dumps({"search_seconds": 12.5, "landscape_seconds": 1.5}),
            encoding="utf-8",
        )
        row = summarize_one(
            query_path,
            directory / "musearch_gt_seed0.budget.json",
            directory / "musearch_gt_seed0.time.json",
        )
        if abs(row["search_auc"] - (-0.21)) > 1e-9:
            raise SystemExit("auc {0}".format(row["search_auc"]))
        if row["candidate_duplicate_rate"] != 0:
            raise SystemExit("duplicate rate {0}".format(row["candidate_duplicate_rate"]))
        if abs(row["best_fitness"] - (-0.03)) > 1e-12:
            raise SystemExit("best included a non-candidate")
        if abs(row["start_fitness"] - (-0.07)) > 1e-12:
            raise SystemExit("start was not the measurement row")
        if row["top_1pct_count"] != 1 or abs(row["top_1pct_mean"] - (-0.03)) > 1e-12:
            raise SystemExit("top 1 percent mean is wrong")
        if abs(row["query_efficiency_per_1000"] - 8.0) > 1e-9:
            raise SystemExit("efficiency {0}".format(row["query_efficiency_per_1000"]))
        if row["unique_sequences"] != 5:
            raise SystemExit("unique candidate count is wrong")
        if abs(row["start_fitness"] - 9.0) < 1e-9:
            raise SystemExit("initialization score entered the curve")

        records = [{"role": "start", "sequence": "WT", "score": 0.0, "candidate_index": ""}]
        for index in range(1, 101):
            records.append(
                {
                    "role": "candidate",
                    "sequence": "BEST",
                    "score": 1.0,
                    "candidate_index": index,
                }
            )
        for offset, index in enumerate(range(101, 201), start=0):
            records.append(
                {
                    "role": "candidate",
                    "sequence": "U{0}".format(offset),
                    "score": 0.5 - offset * 0.001,
                    "candidate_index": index,
                }
            )
        records.append(
            {
                "role": "initialization",
                "sequence": "INIT",
                "score": 9.0,
                "candidate_index": "",
            }
        )
        deduped = summarize_queries(pd.DataFrame(records), budget(200), {"search_seconds": 1, "landscape_seconds": 1})
        if deduped["unique_sequences"] != 101:
            raise SystemExit("unique {0}".format(deduped["unique_sequences"]))
        if deduped["top_1pct_count"] != 2 or abs(deduped["top_1pct_mean"] - 0.75) > 1e-9:
            raise SystemExit(
                "top1 {0} {1}".format(deduped["top_1pct_count"], deduped["top_1pct_mean"])
            )
        if abs(deduped["candidate_duplicate_rate"] - (99 / 200)) > 1e-12:
            raise SystemExit("rate {0}".format(deduped["candidate_duplicate_rate"]))

        short = write_run(
            directory,
            [
                [1, "search_measurement", "start", "WT", -0.07, 0, 2, "", ""],
                [2, "reward", "candidate", "AAAA", -0.05, 0, 2, 1, -0.05],
                [3, "reward", "candidate", "AAAB", -0.04, 0, 2, 2, -0.04],
            ],
        )
        (directory / "musearch_gt_seed0.budget.json").write_text(
            json.dumps(budget(2)), encoding="utf-8"
        )
        try:
            summarize_one(
                short,
                directory / "musearch_gt_seed0.budget.json",
                directory / "musearch_gt_seed0.time.json",
            )
        except ValueError as error:
            if "不能当作" not in str(error):
                raise
        else:
            raise SystemExit("two candidate points were accepted")
    print("plot_queries_ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())

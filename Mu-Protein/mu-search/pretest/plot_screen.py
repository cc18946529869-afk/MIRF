"""Summarize query-level MuFormer traces. Does not train or score sequences.

Search AUC starts at the wild-type score and moves once per candidate score.
Initialization and return rescoring stay off that axis. Fewer than three
candidate scores is rejected.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import pandas as pd


def curve_from_wild_type(start, scores):
    """q=0 is the wild-type score. Each later point is one candidate query."""
    indexes = [0]
    values = [start]
    current = start
    for index, score in enumerate(scores, start=1):
        current = max(current, score)
        indexes.append(index)
        values.append(current)
    return indexes, values


def unique_first_scores(sequences, scores):
    first = {}
    for sequence, score in zip(sequences, scores):
        if sequence not in first:
            first[sequence] = score
    return first


def trapezoid(indexes, values):
    area = 0.0
    for left, right in zip(range(len(values) - 1), range(1, len(values))):
        width = indexes[right] - indexes[left]
        area += width * (values[left] + values[right]) / 2.0
    return area


def summarize_queries(frame, budget, timing):
    if budget["total_by_channel"]["cnn_surrogate"] != 0:
        raise ValueError("CNN 调用不是 0，不能和 MuFormer 混在这次比较里")
    events = budget["events"]["muformer_oracle"]
    if events["gradient"] != 0:
        raise ValueError("这次比较不应有 MuFormer gradient 调用")
    candidates = frame[frame["role"] == "candidate"].copy()
    if len(candidates) < 3:
        raise ValueError(
            "候选曲线只有 {0} 个点。起点和终点不能当作查询级搜索曲线。".format(
                len(candidates)
            )
        )
    indexes = [int(value) for value in candidates["candidate_index"]]
    expected = list(range(1, len(candidates) + 1))
    if indexes != expected:
        raise ValueError("候选序号不是从 1 开始的连续整数")
    if len(candidates) != events["reward"]:
        raise ValueError(
            "候选行数 {0} 与 reward 记账 {1} 不一致".format(
                len(candidates), events["reward"]
            )
        )
    scores = [float(value) for value in candidates["score"]]
    sequences = [str(value) for value in candidates["sequence"]]
    starts = frame[frame["role"] == "start"]
    if len(starts) != 1:
        raise ValueError("回合 0 测量必须正好一行")
    start_score = float(starts["score"].iloc[0])
    curve_index, best = curve_from_wild_type(start_score, scores)
    if curve_index[0] != 0 or curve_index[-1] != len(scores):
        raise ValueError("搜索曲线的横轴必须从野生型 q=0 走到最后一次候选评分")
    area = trapezoid(curve_index, best)
    unique_scores = unique_first_scores(sequences, scores)
    ranked_unique = sorted(unique_scores.values(), reverse=True)
    top_count = max(1, int(math.ceil(0.01 * len(ranked_unique))))
    half = len(scores) // 2
    best_at_half = best[half]
    queries_to_beat_start = ""
    for index, value in zip(curve_index[1:], best[1:]):
        if value > start_score:
            queries_to_beat_start = index
            break
    duplicate_calls = len(scores) - len(unique_scores)
    return {
        "best_fitness": best[-1],
        "start_fitness": start_score,
        "search_auc": area,
        "search_auc_mean": area / len(scores),
        "candidate_queries": len(candidates),
        "top_1pct_count": top_count,
        "top_1pct_mean": sum(ranked_unique[:top_count]) / top_count,
        "unique_sequences": len(unique_scores),
        "candidate_duplicate_calls": duplicate_calls,
        "candidate_duplicate_rate": duplicate_calls / len(scores),
        "query_efficiency_per_1000": 1000.0 * (best[-1] - start_score) / len(scores),
        "best_at_half_budget": best_at_half,
        "gain_first_half": best_at_half - start_score,
        "gain_second_half": best[-1] - best_at_half,
        "queries_to_beat_start": queries_to_beat_start,
        "initialization": events["initialization"],
        "search_measurement": events["search_measurement"],
        "reward": events["reward"],
        "return_rescore": events["return_rescore"],
        "gradient": events["gradient"],
        "cnn_forwards": budget["total_by_channel"]["cnn_surrogate"],
        "duplicate_muformer_calls": budget["duplicate_calls"]["muformer_oracle"],
        "unique_muformer_sequences": budget["unique_sequences"]["muformer_oracle"],
        "muformer_input_sequences": 2
        * (
            events["initialization"]
            + events["reward"]
            + events["return_rescore"]
            + events["search_measurement"]
        ),
        "wall_clock_search_seconds": timing["search_seconds"],
        "wall_clock_landscape_seconds": timing["landscape_seconds"],
    }


def summarize_one(query_path, budget_path, time_path):
    frame = pd.read_csv(query_path)
    budget = json.loads(budget_path.read_text(encoding="utf-8"))
    timing = json.loads(time_path.read_text(encoding="utf-8"))
    row = summarize_queries(frame, budget, timing)
    row["run"] = query_path.name.replace(".queries.csv", "")
    return row


def main(argv):
    parser = argparse.ArgumentParser(description="Summarize a finished TEM-1 screen")
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args(argv)
    rows = []
    errors = []
    paths = sorted(args.result_dir.rglob("*_seed*.queries.csv"))
    if not paths:
        print("目录里没有查询级 CSV：{0}".format(args.result_dir))
        return 1
    for query_path in paths:
        stem = query_path.name.replace(".queries.csv", "")
        budget_path = query_path.with_name(stem + ".budget.json")
        time_path = query_path.with_name(stem + ".time.json")
        if not budget_path.is_file() or not time_path.is_file():
            errors.append("缺少 {0} 的预算或时间文件".format(stem))
            continue
        try:
            rows.append(summarize_one(query_path, budget_path, time_path))
        except ValueError as error:
            errors.append("{0}: {1}".format(stem, error))
    if errors:
        print("\n".join(errors))
        return 1
    table = pd.DataFrame(rows)
    destination = args.result_dir / "screen_summary.csv"
    table.to_csv(destination, index=False)
    print(table.to_string(index=False))
    print("写入 {0}".format(destination))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

"""Read one finished run and reject a bad budget or a non-finite score.

This script does not score sequences and does not rewrite the run directory.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path


def fail(message):
    print("CHECK_FAIL {0}".format(message))
    return 1


def main(argv):
    parser = argparse.ArgumentParser(description="Check one TEM-1 run")
    parser.add_argument("--method", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    stem = "{0}_seed{1}".format(args.method, args.seed)
    directory = args.output_dir
    required = [
        directory / (stem + ".budget.json"),
        directory / (stem + ".queries.csv"),
        directory / (stem + ".time.json"),
        directory / (stem + ".seeds.json"),
        directory / (stem + ".csv"),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        return fail("缺少文件：{0}".format(", ".join(missing)))

    budget = json.loads(required[0].read_text(encoding="utf-8"))
    events = budget["events"]["muformer_oracle"]
    if events["reward"] != 4998:
        return fail("reward={0}".format(events["reward"]))
    if budget["total_by_channel"]["cnn_surrogate"] != 0 or events["gradient"] != 0:
        return fail("CNN 或 gradient 不是 0")
    if events["initialization"] != 105 or events["search_measurement"] != 1:
        return fail(
            "initialization={0} search_measurement={1}".format(
                events["initialization"], events["search_measurement"]
            )
        )
    if events["return_rescore"] > 100:
        return fail("return_rescore={0}".format(events["return_rescore"]))
    metadata = budget["metadata"]
    if metadata.get("seed") != args.seed or metadata.get("method") != args.method:
        return fail("预算里的方法和种子与目录不一致")
    if metadata.get("candidate_limit") != 4998:
        return fail("candidate_limit={0}".format(metadata.get("candidate_limit")))
    if args.method == "musearch_gt":
        schedule = metadata["ppo_schedule"]
        horizon = int(metadata.get("horizon", 5))
        expected = {
            "episodes_per_update": 1666,
            "n_steps": 1666 * horizon,
            "total_timesteps": 4998 * horizon,
            "rollouts": 3,
            "terminal_scores": 4998,
        }
        for key, value in expected.items():
            if schedule.get(key) != value:
                return fail("ppo_schedule {0}={1}".format(key, schedule.get(key)))
    cache = metadata.get("score_cache")
    if cache is not None:
        if cache.get("muformer_forwards") != 4998:
            return fail("缓存未命中={0}".format(cache.get("muformer_forwards")))
        if cache.get("cache_hits") != cache.get("requests", 0) - 4998:
            return fail("缓存请求数不一致：{0}".format(cache))
        if sum(budget.get("cache_hits", {}).values()) != cache.get("cache_hits"):
            return fail("账本缓存命中与包装器不一致")

    seeds = json.loads(required[3].read_text(encoding="utf-8"))
    for key in ("pythonhashseed", "python_random", "numpy", "torch", "tensorflow"):
        if str(seeds.get(key)) != str(args.seed):
            return fail("种子记录 {0}={1}".format(key, seeds.get(key)))

    candidate_indexes = []
    nonfinite = 0
    roles = {}
    with required[1].open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            roles[row["role"]] = roles.get(row["role"], 0) + 1
            try:
                score = float(row["score"])
            except ValueError:
                nonfinite += 1
                continue
            if not math.isfinite(score):
                nonfinite += 1
            if row["role"] == "candidate":
                candidate_indexes.append(int(row["candidate_index"]))
                if len(row["sequence"]) != 286:
                    return fail("候选序列长度不是 286")
    if nonfinite:
        return fail("非有限分数 {0} 个".format(nonfinite))
    if candidate_indexes != list(range(1, 4999)):
        return fail("候选序号不完整，行数 {0}".format(len(candidate_indexes)))
    if roles.get("start") != 1:
        return fail("回合 0 测量行数 {0}".format(roles.get("start")))
    print(
        "CHECK_OK {0} reward=4998 candidates=4998 init={1} start=1 return={2}".format(
            stem, events["initialization"], events["return_rescore"]
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

"""Run one TEM-1 search. Do not execute this until the launch is authorized.

One process scores one method and one seed. It does not read another run's
sequences. PYTHONHASHSEED must already be set to that seed before the
interpreter starts.
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path


CANDIDATE_LIMIT = 4998
REQUIRED_SCHEDULE = {
    "episodes_per_update": 1666,
    "n_steps": 8330,
    "total_timesteps": 24990,
    "rollouts": 3,
    "timesteps_collected": 24990,
    "terminal_scores": 4998,
}


def parse_args(argv):
    parser = argparse.ArgumentParser(description="One TEM-1 MuFormer search")
    parser.add_argument("--method", choices=("musearch_gt", "adalead_gt"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--score-cache", action="store_true")
    return parser.parse_args(argv)


def expected_schedule(horizon):
    return {
        "episodes_per_update": 1666,
        "n_steps": 1666 * horizon,
        "total_timesteps": CANDIDATE_LIMIT * horizon,
        "rollouts": 3,
        "timesteps_collected": CANDIDATE_LIMIT * horizon,
        "terminal_scores": CANDIDATE_LIMIT,
    }


def require_original_schedule(horizon):
    flexs_root = Path(__file__).resolve().parents[1] / "src" / "flexs"
    sys.path.insert(0, str(flexs_root))
    from flexs.musearch_baseline import floor_rollout_schedule

    schedule = floor_rollout_schedule(CANDIDATE_LIMIT, horizon, 3)
    for key, expected in expected_schedule(horizon).items():
        if schedule[key] != expected:
            print(
                "PPO 日程 {0}={1}，要求 {2}。四次更新的安排已停用。计算没有开始。".format(
                    key, schedule[key], expected
                )
            )
            return False
    if horizon == 5:
        for key, expected in REQUIRED_SCHEDULE.items():
            if schedule[key] != expected:
                print("H=5 日程与已经完成的三次不一致：{0}".format(schedule))
                return False
    return True


def require_hash_seed(seed):
    found = os.environ.get("PYTHONHASHSEED")
    if found != str(seed):
        print(
            "PYTHONHASHSEED 必须在解释器启动前设为 {0}。当前是 {1}。计算没有开始。".format(
                seed, found
            )
        )
        return False
    return True


def record_seeds(path, seed):
    import numpy
    import torch
    import tensorflow as tf

    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    for device in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(device, True)
    tf.random.set_seed(seed)
    payload = {
        "seed": seed,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "python_random": seed,
        "numpy": seed,
        "torch": seed,
        "tensorflow": seed,
        "versions": {
            "python": sys.version,
            "numpy": numpy.__version__,
            "torch": torch.__version__,
            "tensorflow": tf.__version__,
        },
        "note": (
            "同一编号只表示这四个生成器都被设成这个整数。"
            "μSearch 和 AdaLead 取随机数的次数和顺序不同，随机数流并不相同。"
        ),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def build_explorer(method, landscape, alphabet, starting_sequence, log_file, horizon, score_cache):
    import numpy as np
    import flexs
    from flexs import baselines
    from flexs.musearch_baseline import CandidateQueryCap, ExactScoreCache

    model = CandidateQueryCap(flexs.LandscapeAsModel(landscape), CANDIDATE_LIMIT)
    if score_cache:
        model = ExactScoreCache(model)
    common = dict(
        starting_sequence=starting_sequence,
        sequences_batch_size=100,
        model_queries_per_batch=CANDIDATE_LIMIT,
        rounds=1,
        log_file=str(log_file),
    )
    if method == "musearch_gt":
        musearch_args = argparse.Namespace(
            score_threshold=-np.inf,
            horizon=horizon,
            mixture_mode="marginal",
            exclude_measured=False,
        )
        return baselines.explorers.MuSearch(
            args=musearch_args,
            oracle_model=model,
            alphabet=alphabet,
            **common,
        )
    return baselines.explorers.Adalead(
        model,
        mu=1,
        recomb_rate=0,
        threshold=0.05,
        rho=0,
        eval_batch_size=20,
        alphabet=alphabet,
        **common,
    )


def check_result(budget, query_path, cap_calls):
    events = budget["events"]["muformer_oracle"]
    problems = []
    if events["reward"] != CANDIDATE_LIMIT:
        problems.append("reward 是 {0}，不是 {1}".format(events["reward"], CANDIDATE_LIMIT))
    if cap_calls != CANDIDATE_LIMIT:
        problems.append("候选包装计数是 {0}".format(cap_calls))
    cache = budget.get("metadata", {}).get("score_cache")
    if cache is not None:
        if cache.get("muformer_forwards") != CANDIDATE_LIMIT:
            problems.append("缓存未命中不是 {0}".format(CANDIDATE_LIMIT))
        if cache.get("cache_hits") != cache.get("requests", 0) - cache.get("muformer_forwards", 0):
            problems.append("缓存命中与请求数不一致：{0}".format(cache))
        recorded_hits = sum(budget.get("cache_hits", {}).values())
        if recorded_hits != cache.get("cache_hits"):
            problems.append(
                "账本缓存命中 {0}，包装器 {1}".format(recorded_hits, cache.get("cache_hits"))
            )
    if budget["total_by_channel"]["cnn_surrogate"] != 0:
        problems.append("CNN 调用不是 0")
    if events["gradient"] != 0:
        problems.append("gradient 不是 0")
    if events["initialization"] != 105:
        problems.append("initialization 是 {0}，预期 105".format(events["initialization"]))
    if events["search_measurement"] != 1:
        problems.append("search_measurement 是 {0}".format(events["search_measurement"]))
    text = query_path.read_text(encoding="utf-8")
    candidate_rows = sum(1 for line in text.splitlines() if ",candidate," in line)
    if candidate_rows != CANDIDATE_LIMIT:
        problems.append("查询级候选行数是 {0}".format(candidate_rows))
    return problems


def main(argv):
    args = parse_args(argv)
    if args.horizon < 1:
        print("horizon 必须是正整数。计算没有开始。")
        return 2
    if args.method != "musearch_gt" and args.horizon != 5:
        print("horizon 只随 μSearch 改变。AdaLead 没有这个参数。计算没有开始。")
        return 2
    if args.score_cache and args.method != "adalead_gt":
        print("精确序列缓存这一步只包装 AdaLead。计算没有开始。")
        return 2
    if not require_hash_seed(args.seed):
        return 2
    if not require_original_schedule(args.horizon):
        return 2
    runs = Path(__file__).resolve().parent / "runs"
    resolved = args.output_dir.resolve()
    for name in ("musearch_gt", "adalead_gt"):
        root = (runs / name).resolve()
        if resolved == root or root in resolved.parents:
            print("这个目录属于已经完成的六次结果，不写入：{0}".format(resolved))
            return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = "{0}_seed{1}".format(args.method, args.seed)
    occupied = [path for path in args.output_dir.rglob("*") if path.is_file()]
    if occupied:
        print("输出目录不是空的，不覆盖：{0}".format(args.output_dir))
        return 2
    record_seeds(args.output_dir / (stem + ".seeds.json"), args.seed)
    os.environ["MUSEARCH_TB_DIR"] = str(args.output_dir / "tensorboard")

    import flexs.utils.sequence_utils as s_utils
    from flexs.landscapes.muformer import MuformerLandscape

    landscape_started = time.perf_counter()
    landscape = MuformerLandscape()
    landscape_seconds = time.perf_counter() - landscape_started
    log_file = args.output_dir / (stem + ".csv")
    explorer = build_explorer(
        args.method,
        landscape,
        s_utils.AAS,
        landscape.starting_sequence,
        log_file,
        args.horizon,
        args.score_cache,
    )
    explorer.score_ledger.metadata["seed"] = args.seed
    explorer.score_ledger.metadata["method"] = args.method
    explorer.score_ledger.metadata["candidate_limit"] = CANDIDATE_LIMIT
    explorer.score_ledger.metadata["horizon"] = args.horizon
    explorer.score_ledger.metadata["score_cache_enabled"] = bool(args.score_cache)
    if args.score_cache:
        cache = explorer.model._model
        if type(cache).__name__ != "ExactScoreCache":
            print("精确缓存没有装到 AdaLead 的评分器上。计算没有开始。")
            return 2
        cache.ledger = explorer.score_ledger
    if args.method == "musearch_gt":
        schedule = explorer.score_ledger.metadata["ppo_schedule"]
        for key, expected in expected_schedule(args.horizon).items():
            if schedule[key] != expected:
                print("MuSearch 日程与三次 rollout 不一致：{0}".format(schedule))
                return 2
    search_started = time.perf_counter()
    explorer.run(landscape)
    search_seconds = time.perf_counter() - search_started
    timing = {
        "search_seconds": search_seconds,
        "landscape_seconds": landscape_seconds,
        "method": args.method,
        "seed": args.seed,
        "horizon": args.horizon,
        "score_cache": bool(args.score_cache),
    }
    (args.output_dir / (stem + ".time.json")).write_text(
        json.dumps(timing, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    budget_path = args.output_dir / (stem + ".budget.json")
    budget = json.loads(budget_path.read_text(encoding="utf-8"))
    problems = check_result(
        budget,
        args.output_dir / (stem + ".queries.csv"),
        explorer.model._model.candidate_calls,
    )
    print(
        "search_seconds={0:.1f} landscape_seconds={1:.1f}".format(
            search_seconds, landscape_seconds
        )
    )
    if problems:
        print("\n".join(problems))
        return 3
    print("候选预算 {0} 核对通过。".format(CANDIDATE_LIMIT))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

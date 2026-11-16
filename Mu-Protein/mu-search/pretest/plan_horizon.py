"""Print the horizon and cache commands. --launch does not start them."""

import argparse
import sys
from pathlib import Path


WORKTREE = Path("/data/cc233_data/work_dir/MIRF-musearch-pretest/Mu-Protein/mu-search")
PYTHON = "/tmp/musearch-ppo-isolated/bin/python"
WEIGHTS = "/data/cc233_data/tmp/musearch-convert/out"
CWD = "/data/cc233_data/work_dir/MIRF/Mu-Protein/mu-search"


def command(method, seed, output, horizon=None, score_cache=False):
    flags = [
        "--method", method,
        "--seed", str(seed),
        "--output-dir", str(output),
    ]
    if horizon is not None:
        flags.extend(["--horizon", str(horizon)])
    if score_cache:
        flags.append("--score-cache")
    tb = output / "tensorboard"
    return (
        "cd {cwd} && PYTHONHASHSEED={seed} CUDA_VISIBLE_DEVICES=1 "
        "MUSEARCH_CONVERTED_WEIGHTS={weights} "
        "MUSEARCH_TB_DIR={tb} "
        "PYTHONPATH={pythonpath} "
        "{python} -u {driver} {flags}".format(
            cwd=CWD,
            seed=seed,
            weights=WEIGHTS,
            tb=tb,
            pythonpath=WORKTREE / "src" / "flexs",
            python=PYTHON,
            driver=WORKTREE / "pretest" / "run_one.py",
            flags=" ".join(flags),
        )
    )


def jobs():
    runs = WORKTREE / "pretest" / "runs" / "horizon"
    rows = []
    for horizon in (7, 10):
        for seed in (0, 1, 2):
            output = runs / "musearch_h{0}".format(horizon) / "seed{0}".format(seed)
            rows.append(("musearch_gt", seed, horizon, False, output))
    for seed in (0, 1, 2):
        output = runs / "adalead_cache" / "seed{0}".format(seed)
        rows.append(("adalead_gt", seed, None, True, output))
    return rows


def main(argv):
    parser = argparse.ArgumentParser(description="Horizon ablation plan")
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args(argv)
    print("H=5 不重跑，使用 pretest/runs/musearch_gt/seed0,1,2。")
    print("无缓存 AdaLead 不重跑，使用 pretest/runs/adalead_gt/seed0,1,2。")
    print("新任务 9 个，顺序执行，物理 GPU 1。本脚本不启动。")
    print("预计新增 GPU 时间约 5 到 8 小时。这是按已完成的 H=5 耗时外推，不是新的计时。")
    for method, seed, horizon, score_cache, output in jobs():
        print(command(method, seed, output, horizon, score_cache))
        print("然后：{0} --method {1} --seed {2} --output-dir {3}".format(
            WORKTREE / "pretest" / "check_one.py", method, seed, output
        ))
    if args.launch:
        print("未授权，不启动。")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

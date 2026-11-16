"""Print the TEM-1 launch commands and refuse to start them.

This file does not import flexs, TensorFlow, or PyTorch, and it does not
create a landscape.
"""

import argparse
import json
import sys
from pathlib import Path


CONFIG = Path(__file__).with_name("tem1_screen.json")
REPORT = Path(__file__).with_name("budget_report.txt")


def load_config(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def output_directories(config):
    paths = config["paths"]
    budget = config["budget"]
    if budget["methods"] != ["musearch_gt", "adalead_gt"]:
        raise SystemExit("方法必须是 musearch_gt 和 adalead_gt")
    if budget["seeds"] != [0, 1, 2]:
        raise SystemExit("种子必须是 0、1、2")
    expected_total = (
        len(budget["methods"])
        * len(budget["seeds"])
        * budget["candidate_queries_per_run"]
    )
    if budget["nominal_candidate_total"] != expected_total:
        raise SystemExit(
            "名义总预算 {0} 与 {1} 不一致".format(
                budget["nominal_candidate_total"], expected_total
            )
        )
    directories = []
    for method in budget["methods"]:
        for seed in budget["seeds"]:
            directories.append("{0}/{1}/seed{2}".format(paths["runs"], method, seed))
    if len(directories) != len(set(directories)):
        raise SystemExit("输出目录重叠")
    return directories


def launch_command(config, method, seed):
    paths = config["paths"]
    run_dir = "{0}/{1}/seed{2}".format(paths["runs"], method, seed)
    lines = [
        "cd {0}".format(paths["cwd"]),
        "PYTHONHASHSEED={0} CUDA_VISIBLE_DEVICES=1 \\".format(seed),
        "MUSEARCH_CONVERTED_WEIGHTS={0} \\".format(
            config["landscape"]["converted_weights"]
        ),
        "MUSEARCH_TB_DIR={0}/tensorboard \\".format(run_dir),
        "PYTHONPATH={0} \\".format(paths["pythonpath"]),
        "{0} {1} --method {2} --seed {3} --output-dir {4}".format(
            paths["python"], paths["driver"], method, seed, run_dir
        ),
    ]
    return "\n".join(lines)


def report_lines(config):
    budget = config["budget"]
    schedule = budget["parameters_are_existing_defaults"]["musearch"]["actual_schedule"]
    adalead = budget["parameters_are_existing_defaults"]["adalead"]
    outside = budget["outside_candidate_budget_per_run"]
    construction = config["landscape"]["construction_queries_per_process"]
    runs = budget["planned_runs"]
    lines = [
        "TEM-1 方案 1 启动准备",
        "问题：{0}".format(config["question"]),
        "启动许可：{0}".format(config["launch_allowed"]),
        "方法：{0}".format(", ".join(budget["methods"])),
        "排除 GWG：{0}".format(config["excluded"]["gwg"]),
        "排除全 CNN：{0}".format(config["excluded"]["cnn_inner_model"]),
        "",
        "候选预算",
        "每次搜索 reward 评分：{0}".format(budget["candidate_queries_per_run"]),
        "方法数 {0}，种子 {1}，候选合计 {2}".format(
            len(budget["methods"]),
            ", ".join(str(seed) for seed in budget["seeds"]),
            budget["nominal_candidate_total"],
        ),
        "rounds={0}，model_queries_per_batch={1}，sequences_batch_size={2}".format(
            budget["rounds"],
            budget["model_queries_per_batch"],
            budget["sequences_batch_size"],
        ),
        "这 {0} 次不含初始化、回合 0 测量和返回重评分。".format(
            budget["nominal_candidate_total"]
        ),
        "",
        "每次运行另记，不计入 {0}".format(budget["candidate_queries_per_run"]),
        "initialization：{0}（野生型 {1} + ESBL {2}）。六个进程各自构造，合计 {3}。".format(
            outside["initialization"],
            construction["wild_type"],
            construction["esbl_table"],
            runs * outside["initialization"],
        ),
        "search_measurement：{0}。六个进程合计 {1}。".format(
            outside["search_measurement"], runs * outside["search_measurement"]
        ),
        "return_rescore：最多 {0}。六个进程最多 {1}。重复序列也计数。".format(
            outside["return_rescore_at_most"],
            runs * outside["return_rescore_at_most"],
        ),
        "因此六次运行的 MuFormer predict 上限是 {0}。CNN 必须是 0，gradient 必须是 0。".format(
            budget["nominal_candidate_total"]
            + runs
            * (
                outside["initialization"]
                + outside["search_measurement"]
                + outside["return_rescore_at_most"]
            )
        ),
        "一次 predict 的模型输入是 {0} 条序列、每条 {1} 个 token。".format(
            config["muformer_batch"]["input_sequences_per_candidate_query"],
            config["muformer_batch"]["tokens_per_sequence"],
        ),
        "",
        "μSearch 终点评分",
        schedule["reason"],
        "本轮实际：每批 {0} 条轨迹，{1} 批，n_steps={2}，total_timesteps={3}，终点评分 {4}。".format(
            schedule["episodes_per_update"],
            schedule["updates"],
            schedule["n_steps"],
            schedule["total_timesteps"],
            schedule["terminal_scores"],
        ),
        schedule["unchanged"],
        schedule["rejected_alternative"],
        schedule["executed_check"],
        budget["parameters_are_existing_defaults"]["musearch"]["ppo_batch_note"],
        "",
        "AdaLead",
        "参数保持 mu={0}，recomb_rate={1}，threshold={2}，rho={3}，eval_batch_size={4}。".format(
            adalead["mu"],
            adalead["recomb_rate"],
            adalead["threshold"],
            adalead["rho"],
            adalead["eval_batch_size"],
        ),
        adalead["last_batch"],
        adalead["parent_rescore"],
        "",
        "随机数",
        budget["same_seed_is_not_the_same_stream"],
        "每个方法的每个种子是单独进程，输出目录分开，不读另一个种子的搜索结果。",
        "六个输出目录：",
        *output_directories(config),
        "",
        "曲线和指标",
        config["search_auc"],
        "Best Fitness 是从野生型出发的 best-so-far 终点，即野生型和全部候选分数里的最大值。",
        config["top_1pct"],
        "Unique Sequences 是候选查询里的不同序列数。candidate_duplicate_rate 是候选重复调用占候选查询的比例。重复调用仍占用 4998。",
        "Query Efficiency 是（候选最终 best-so-far − 回合 0 测量）/（候选查询数 / 1000）。",
        "Wall-clock Time 是 explorer.run 的秒数，景观构造另记。",
        "",
        "时间估计",
        config["runtime_estimate"]["basis"],
        "一次搜索的前向大约 {0} 分钟，另加约 1 分钟构造和最多约 1 分钟返回重评分。".format(
            config["runtime_estimate"]["one_run_forward_minutes"]
        ),
        "六次顺序运行：{0}".format(config["runtime_estimate"]["six_runs_sequential_hours"]),
        config["runtime_estimate"]["gpu"],
        config["landscape"]["device_note"],
        "",
        "启动命令",
        "本文件不会执行这些命令。",
    ]
    for method in budget["methods"]:
        for seed in budget["seeds"]:
            lines.append("")
            lines.append("# {0} seed {1}".format(method, seed))
            lines.append(launch_command(config, method, seed))
    return lines


def main(argv):
    parser = argparse.ArgumentParser(description="TEM-1 screen plan")
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument(
        "--launch",
        action="store_true",
        help="Request a run. This script still refuses.",
    )
    args = parser.parse_args(argv)
    config = load_config(args.config)
    text = "\n".join(report_lines(config)) + "\n"
    REPORT.write_text(text, encoding="utf-8")
    print(text, end="")
    print("报告写入 {0}".format(REPORT))
    if not config["launch_allowed"]:
        print("配置里的 launch_allowed 仍是 false。计算没有启动。")
    if args.launch:
        print("拒绝启动。")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

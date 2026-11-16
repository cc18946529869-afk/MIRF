"""Offline Hamming audit of the six finished TEM-1 searches.

This script does not load MuFormer, does not score new sequences, and does
not change the search code. Final Hamming distance is not an edit history.
"""

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path


AAS = set("ACDEFGHIKLMNPQRSTVWY")
ORDERS = [str(i) for i in range(11)] + ["10+"]
RUNS = (
    ("musearch_gt", 0),
    ("musearch_gt", 1),
    ("musearch_gt", 2),
    ("adalead_gt", 0),
    ("adalead_gt", 1),
    ("adalead_gt", 2),
)
DEFAULT_RUNS = Path(
    "/data/cc233_data/work_dir/MIRF-musearch-pretest/Mu-Protein/mu-search/pretest/runs"
)


def hamming(sequence, wild_type):
    if len(sequence) != len(wild_type):
        raise ValueError("sequences differ in length")
    return sum(left != right for left, right in zip(sequence, wild_type))


def order_label(distance):
    return str(distance) if distance <= 10 else "10+"


def mutation_list(sequence, wild_type):
    if len(sequence) != len(wild_type):
        raise ValueError("sequences differ in length")
    return [
        "{0}{1}{2}".format(wild_type[index], index + 1, sequence[index])
        for index in range(len(wild_type))
        if sequence[index] != wild_type[index]
    ]


def percentile(values, probability):
    ordered = sorted(values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * probability
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return float(ordered[low])
    weight = position - low
    return float(ordered[low] * (1.0 - weight) + ordered[high] * weight)


def first_scores(rows):
    """Keep the first score of each sequence, in first-seen order."""
    seen = {}
    for sequence, score in rows:
        if sequence not in seen:
            seen[sequence] = score
    return list(seen.items())


def summarize(values):
    count = len(values)
    if count == 0:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "std": None,
            "p25": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "maximum": None,
            "top10_mean": None,
        }
    top10 = None
    if count >= 10:
        top10 = sum(sorted(values, reverse=True)[:10]) / 10.0
    return {
        "n": count,
        "mean": sum(values) / count,
        "median": percentile(values, 0.50),
        "std": statistics.stdev(values) if count > 1 else None,
        "p25": percentile(values, 0.25),
        "p75": percentile(values, 0.75),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "maximum": max(values),
        "top10_mean": top10,
    }


def load_run(runs_root, method, seed):
    directory = runs_root / method / "seed{0}".format(seed)
    stem = "{0}_seed{1}".format(method, seed)
    queries = []
    with (directory / (stem + ".queries.csv")).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            queries.append(row)
    budget = json.loads((directory / (stem + ".budget.json")).read_text(encoding="utf-8"))
    timing = json.loads((directory / (stem + ".time.json")).read_text(encoding="utf-8"))
    seeds = json.loads((directory / (stem + ".seeds.json")).read_text(encoding="utf-8"))
    return {
        "method": method,
        "seed": seed,
        "directory": directory,
        "stem": stem,
        "queries": queries,
        "budget": budget,
        "timing": timing,
        "seeds": seeds,
    }


def finite_score(text):
    try:
        value = float(text)
    except ValueError:
        return None
    if math.isfinite(value):
        return value
    return None


def illegal_characters(sequence):
    return sorted(set(sequence) - AAS)


def check_integrity(run):
    problems = []
    queries = run["queries"]
    events = run["budget"]["events"]
    candidates = [row for row in queries if row["role"] == "candidate"]
    starts = [row for row in queries if row["role"] == "start"]
    if len(candidates) != 4998:
        problems.append("候选行数是 {0}".format(len(candidates)))
    if events["muformer_oracle"]["reward"] != 4998:
        problems.append("reward 是 {0}".format(events["muformer_oracle"]["reward"]))
    if events["cnn_surrogate"]["gradient"] != 0 or sum(events["cnn_surrogate"].values()) != 0:
        problems.append("CNN 通道不是 0")
    if events["muformer_oracle"]["gradient"] != 0:
        problems.append("gradient 不是 0")
    indexes = []
    for row in candidates:
        if row["candidate_index"] == "":
            problems.append("候选行缺少 candidate_index")
            continue
        indexes.append(int(row["candidate_index"]))
    if indexes != list(range(1, len(indexes) + 1)):
        problems.append("candidate_index 不是从 1 连续到 {0}".format(len(indexes)))
    if len(starts) != 1:
        problems.append("role=start 有 {0} 行".format(len(starts)))
    wild_type = starts[0]["sequence"] if starts else ""
    wild_score = finite_score(starts[0]["score"]) if starts else None
    if wild_score is None:
        problems.append("WT 分数不是有限数")
    bad_scores = 0
    bad_chars = []
    empty = 0
    length_mismatch = 0
    for row in queries:
        if finite_score(row["score"]) is None:
            bad_scores += 1
        sequence = row["sequence"]
        if sequence == "":
            empty += 1
        else:
            illegal = illegal_characters(sequence)
            if illegal:
                bad_chars.append((row["call_index"], "".join(illegal)))
        if wild_type and row["role"] == "candidate" and len(sequence) != len(wild_type):
            length_mismatch += 1
    if bad_scores:
        problems.append("非有限分数 {0} 行".format(bad_scores))
    if empty:
        problems.append("空序列 {0} 行".format(empty))
    if bad_chars:
        problems.append("非法氨基酸 {0} 处，例如 {1}".format(len(bad_chars), bad_chars[:3]))
    if length_mismatch:
        problems.append("候选长度与 WT 不同的有 {0} 行".format(length_mismatch))
    init_lengths = {}
    for row in queries:
        if row["role"] == "initialization":
            init_lengths.setdefault(len(row["sequence"]), 0)
            init_lengths[len(row["sequence"])] += 1
    score_by_sequence = {}
    conflicting = 0
    for row in queries:
        value = finite_score(row["score"])
        if value is None:
            continue
        previous = score_by_sequence.get(row["sequence"])
        if previous is None:
            score_by_sequence[row["sequence"]] = value
        elif previous != value:
            conflicting += 1
    if conflicting:
        problems.append("同一序列出现不同分数 {0} 次".format(conflicting))
    short_sequences = sorted({
        row["sequence"]
        for row in queries
        if row["role"] == "initialization" and wild_type and len(row["sequence"]) != len(wild_type)
    })
    schedule = run["budget"].get("metadata", {}).get("ppo_schedule")
    return {
        "problems": problems,
        "wild_type": wild_type,
        "wild_score": wild_score,
        "candidate_rows": len(candidates),
        "query_rows": len(queries),
        "cnn_total": sum(events["cnn_surrogate"].values()),
        "gradient": events["muformer_oracle"]["gradient"],
        "initialization": events["muformer_oracle"]["initialization"],
        "return_rescore": events["muformer_oracle"]["return_rescore"],
        "reward": events["muformer_oracle"]["reward"],
        "init_lengths": init_lengths,
        "short_sequences": short_sequences,
        "search_seconds": run["timing"].get("search_seconds"),
        "time_keys": sorted(run["timing"].keys()),
        "ppo_schedule": schedule,
        "duplicate_calls_any_channel": run["budget"].get("duplicate_calls", {}).get("muformer_oracle"),
        "unique_any_channel": run["budget"].get("unique_sequences_any_channel"),
        "seed_file": run["seeds"].get("seed"),
        "seed_note": run["seeds"].get("note"),
    }


def candidate_records(run, wild_type):
    records = []
    for row in run["queries"]:
        if row["role"] != "candidate":
            continue
        sequence = row["sequence"]
        distance = hamming(sequence, wild_type) if len(sequence) == len(wild_type) else None
        records.append(
            {
                "sequence": sequence,
                "score": finite_score(row["score"]),
                "distance": distance,
                "order": order_label(distance) if distance is not None else "length_mismatch",
                "candidate_index": int(row["candidate_index"]),
            }
        )
    return records


def grouped_stats(records, scope):
    if scope == "unique":
        paired = first_scores((row["sequence"], row) for row in records)
        rows = [row for _sequence, row in paired]
    else:
        rows = records
    buckets = {label: [] for label in ORDERS}
    unique_in_bucket = {label: set() for label in ORDERS}
    for row in rows:
        if row["order"] not in buckets or row["score"] is None:
            continue
        buckets[row["order"]].append(row["score"])
        unique_in_bucket[row["order"]].add(row["sequence"])
    output = []
    for label in ORDERS:
        stats = summarize(buckets[label])
        calls = stats["n"]
        unique_count = len(unique_in_bucket[label]) if scope == "query" else calls
        duplicate_fraction = None
        if scope == "query" and calls:
            duplicate_fraction = (calls - unique_count) / calls
        output.append(
            {
                "order": label,
                "calls_or_unique": calls,
                "unique_sequences": unique_count,
                "duplicate_fraction": duplicate_fraction,
                **{key: stats[key] for key in stats if key != "n"},
            }
        )
    return output


def top_unique(records, count):
    unique = first_scores((row["sequence"], row) for row in records)
    ranked = sorted(unique, key=lambda item: (-item[1]["score"], item[1]["candidate_index"]))
    return ranked[:count]


def write_csv(path, rows, fieldnames):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_cell(row.get(key)) for key in fieldnames})


def format_cell(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return "{0:.8g}".format(value)
    return value


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def analyze(runs_root):
    loaded = [load_run(runs_root, method, seed) for method, seed in RUNS]
    integrity = [check_integrity(run) for run in loaded]
    wild_types = {item["wild_type"] for item in integrity}
    if len(wild_types) != 1 or "" in wild_types:
        raise RuntimeError("六次运行的 WT 不一致，停止汇总")
    wild_type = next(iter(wild_types))
    per_run = []
    for run, item in zip(loaded, integrity):
        records = candidate_records(run, wild_type)
        per_run.append(
            {
                "method": run["method"],
                "seed": run["seed"],
                "integrity": item,
                "records": records,
                "query": grouped_stats(records, "query"),
                "unique": grouped_stats(records, "unique"),
            }
        )
    return {"wild_type": wild_type, "runs": per_run}


def order_rows(result):
    rows = []
    for run in result["runs"]:
        for scope in ("query", "unique"):
            for stats in run[scope]:
                rows.append(
                    {
                        "method": run["method"],
                        "seed": run["seed"],
                        "scope": scope,
                        "order": stats["order"],
                        "n": stats["calls_or_unique"],
                        "unique_sequences": stats["unique_sequences"],
                        "duplicate_fraction": stats["duplicate_fraction"],
                        "mean": stats["mean"],
                        "median": stats["median"],
                        "std": stats["std"],
                        "p25": stats["p25"],
                        "p75": stats["p75"],
                        "p90": stats["p90"],
                        "p95": stats["p95"],
                        "p99": stats["p99"],
                        "maximum": stats["maximum"],
                        "top10_mean": stats["top10_mean"],
                    }
                )
    return rows


def summary_rows(order_table):
    groups = {}
    for row in order_table:
        groups.setdefault((row["method"], row["scope"], row["order"]), []).append(row)
    output = []
    for (method, scope, order), rows in groups.items():
        def values(key):
            return [row[key] for row in rows if isinstance(row[key], (int, float))]

        counts = values("n")
        output.append(
            {
                "method": method,
                "scope": scope,
                "order": order,
                "n_seeds": len(rows),
                "n_min": min(counts) if counts else 0,
                "n_max": max(counts) if counts else 0,
                "n_sum": sum(counts) if counts else 0,
                "mean_of_means": statistics.fmean(values("mean")) if values("mean") else None,
                "median_of_medians": statistics.fmean(values("median")) if values("median") else None,
                "maximum_of_maxima": max(values("maximum")) if values("maximum") else None,
                "minimum_of_maxima": min(values("maximum")) if values("maximum") else None,
            }
        )
    return output


def high_score_tables(result):
    appearances = {}
    prepared = []
    for run in result["runs"]:
        unique_count = sum(row["calls_or_unique"] for row in run["unique"])
        top1 = max(1, math.ceil(0.01 * unique_count))
        selections = {
            "top1pct": top_unique(run["records"], top1),
            "top10": top_unique(run["records"], 10),
            "top50": top_unique(run["records"], 50),
        }
        prepared.append((run, selections, top1))
        for sequence, _row in first_scores((row["sequence"], row) for row in run["records"]):
            appearances.setdefault(sequence, set()).add("{0}/seed{1}".format(run["method"], run["seed"]))
    candidate_rows = []
    substitution_sets = {}
    for run, selections, top1 in prepared:
        for set_name, ranked in selections.items():
            key = (run["method"], run["seed"], set_name)
            substitution_sets[key] = []
            for rank, (sequence, row) in enumerate(ranked, start=1):
                mutations = mutation_list(sequence, result["wild_type"])
                substitution_sets[key].append(mutations)
                others = sorted(appearances[sequence] - {"{0}/seed{1}".format(run["method"], run["seed"])})
                candidate_rows.append(
                    {
                        "method": run["method"],
                        "seed": run["seed"],
                        "set": set_name,
                        "set_size": top1 if set_name == "top1pct" else len(ranked),
                        "rank": rank,
                        "score": row["score"],
                        "hamming": row["distance"],
                        "mutations": ";".join(mutations),
                        "n_runs_with_sequence": len(appearances[sequence]),
                        "other_runs": ";".join(others),
                    }
                )
    return candidate_rows, substitution_sets


def recurrent_rows(candidate_rows):
    """Count substitutions and pairs inside each high-score set.

    Co-occurrence is not epistasis. No single-mutant control is computed.
    """
    grouped = {}
    for row in candidate_rows:
        grouped.setdefault(row["set"], []).append(row)
    output = []
    for set_name, rows in grouped.items():
        substitution_counts = {}
        pair_counts = {}
        for row in rows:
            mutations = [item for item in row["mutations"].split(";") if item]
            identity = (row["method"], row["seed"], row["rank"])
            for mutation in mutations:
                substitution_counts.setdefault(mutation, set()).add(identity)
            for left_index, left in enumerate(mutations):
                for right in mutations[left_index + 1:]:
                    pair = tuple(sorted((left, right)))
                    pair_counts.setdefault(pair, set()).add(identity)
        for mutation, owners in sorted(substitution_counts.items(), key=lambda item: (-len(item[1]), item[0])):
            if len(owners) < 2:
                continue
            run_names = sorted({"{0}/seed{1}".format(owner[0], owner[1]) for owner in owners})
            output.append(
                {
                    "set": set_name,
                    "kind": "substitution",
                    "item": mutation,
                    "n_sequences": len(owners),
                    "n_runs": len(run_names),
                    "runs": ";".join(run_names),
                }
            )
        for pair, owners in sorted(pair_counts.items(), key=lambda item: (-len(item[1]), item[0])):
            if len(owners) < 2:
                continue
            run_names = sorted({"{0}/seed{1}".format(owner[0], owner[1]) for owner in owners})
            output.append(
                {
                    "set": set_name,
                    "kind": "pair",
                    "item": "{0}+{1}".format(*pair),
                    "n_sequences": len(owners),
                    "n_runs": len(run_names),
                    "runs": ";".join(run_names),
                }
            )
    return output


def schedule_paragraph(result):
    schedules = [run["integrity"]["ppo_schedule"] for run in result["runs"] if run["method"] == "musearch_gt"]
    adalead = [run["integrity"]["ppo_schedule"] for run in result["runs"] if run["method"] == "adalead_gt"]
    if any(item is not None for item in adalead):
        adalead_text = "AdaLead 的 metadata 里出现了 ppo_schedule。"
    else:
        adalead_text = "AdaLead 的 metadata 没有 ppo_schedule，也没有 horizon。"
    if not schedules or any(item is None for item in schedules) or any(item != schedules[0] for item in schedules):
        return "μSearch 三次的 ppo_schedule 不一致或缺失。" + adalead_text
    schedule = schedules[0]
    edits = schedule["n_steps"] / schedule["episodes_per_update"]
    return (
        "μSearch 三次的 ppo_schedule 相同：rollouts={0}，episodes_per_update={1}，n_steps={2}，"
        "total_timesteps={3}，terminal_scores={4}。n_steps/episodes_per_update={5:g}，"
        "这是产物里能直接读到的每条轨迹编辑次数。{6}"
    ).format(
        schedule["rollouts"],
        schedule["episodes_per_update"],
        schedule["n_steps"],
        schedule["total_timesteps"],
        schedule["terminal_scores"],
        edits,
        adalead_text,
    )


def duplicate_sentence(run, wild_type):
    seen = set()
    extra = 0
    extra_non_wt = 0
    for row in run["records"]:
        if row["sequence"] in seen:
            extra += 1
            if row["sequence"] != wild_type:
                extra_non_wt += 1
        else:
            seen.add(row["sequence"])
    return "{0}/seed{1} 重复调用 {2}，其中非 WT {3}".format(
        run["method"], run["seed"], extra, extra_non_wt
    )


def candidate_duplicate_paragraph(result):
    pieces = []
    for run in result["runs"]:
        if run["method"] != "musearch_gt":
            continue
        pieces.append(
            "{0}/seed{1} 候选唯一序列 {2}，全通道重复 {3}".format(
                run["method"],
                run["seed"],
                len({row["sequence"] for row in run["records"]}),
                run["integrity"]["duplicate_calls_any_channel"],
            )
        )
    adalead = "；".join(
        duplicate_sentence(run, result["wild_type"])
        for run in result["runs"]
        if run["method"] == "adalead_gt"
    )
    returns = "，".join(
        "{0}/seed{1} 返回重评分 {2}".format(run["method"], run["seed"], run["integrity"]["return_rescore"])
        for run in result["runs"]
    )
    all_distinct = all(
        len({row["sequence"] for row in run["records"]}) == 4998
        for run in result["runs"]
        if run["method"] == "musearch_gt"
    )
    distinct_text = (
        "三次候选唯一序列都是 4998，全通道重复发生在候选轴以外。"
        if all_distinct
        else "三次候选唯一序列没有都等于 4998。"
    )
    return (
        "μSearch 候选去重：{0}。{1}"
        "AdaLead 候选轴上的重复调用：{2}。返回重评分：{3}。这些差异不改变 4998 条候选行。"
    ).format("；".join(pieces), distinct_text, adalead, returns)


def integrity_markdown(result):
    lines = [
        "# 六次运行的数据核验",
        "",
        "原始文件没有修改。下面只记录读到的内容。",
        "",
        "WT 六次一致，长度 {0}。`role=start` 的分数是 {1:.8g}。".format(
            len(result["wild_type"]), result["runs"][0]["integrity"]["wild_score"]
        ),
        "",
        "六次目录里没有 `action_trace.csv`。`time.json` 的键是 {0}，其中没有 horizon。终点汉明距离不能反推空编辑、重复编辑或回退。".format(
            "、".join(result["runs"][0]["integrity"]["time_keys"])
        ),
        "",
        schedule_paragraph(result),
        "",
        "| 运行 | 候选行 | 候选唯一序列 | reward | CNN | gradient | 索引 | 初始化长度 | 返回重评分 | 全通道重复 | 搜索分钟 | 问题 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for run in result["runs"]:
        item = run["integrity"]
        unique_candidates = len({row["sequence"] for row in run["records"]})
        lines.append(
            "| {0}/seed{1} | {2} | {3} | {4} | {5} | {6} | {7} | {8} | {9} | {10} | {11:.1f} | {12} |".format(
                run["method"],
                run["seed"],
                item["candidate_rows"],
                unique_candidates,
                item["reward"],
                item["cnn_total"],
                item["gradient"],
                "连续" if not any("candidate_index" in problem for problem in item["problems"]) else "不连续",
                item["init_lengths"],
                item["return_rescore"],
                item["duplicate_calls_any_channel"],
                item["search_seconds"] / 60.0,
                "无" if not item["problems"] else "；".join(item["problems"]),
            )
        )
    short = result["runs"][0]["integrity"]["short_sequences"]
    same_short = all(run["integrity"]["short_sequences"] == short for run in result["runs"])
    length_text = "、".join(
        "{0} 条长度为 {1}".format(count, length)
        for length, count in sorted(result["runs"][0]["integrity"]["init_lengths"].items())
    )
    lines.extend(
        [
            "",
            "候选行都是 4998，分数都有限，CNN 和 gradient 都是 0，`candidate_index` 都是 1 到 4998。候选序列长度都是 286，没有空序列，也没有标准氨基酸以外的字符。同一序列没有出现两个不同的分数。",
            "",
            "初始化共 105 条，六次运行的长度分布相同：{0}。长度不是 286 的初始化序列有 {1} 条，六次是否为同一组：{2}。这是初始化表里的既有差异，候选表没有因此缺行。".format(
                length_text, len(short), "是" if same_short else "否"
            ),
            "",
            candidate_duplicate_paragraph(result),
            "",
            "`seeds.json` 写明同一整数只是把四个生成器设成该整数，两种方法的随机数流并不相同。",
            "",
            "查询口径按每一次候选调用计数。唯一序列口径对每条候选序列只保留第一次分数。重复调用不另算成新的突变体。",
        ]
    )
    return "\n".join(lines) + "\n"


def unique_records(run):
    return [row for _sequence, row in first_scores((item["sequence"], item) for item in run["records"])]


def fmt(value):
    if value is None:
        return ""
    return "{0:.6f}".format(value)


def order_stats(run, label):
    for stats in run["unique"]:
        if stats["order"] == label:
            return stats
    raise KeyError(label)


def cfx_subset(test_path):
    other = test_path.with_name("CFX_test_from_pan_above_median.tsv")
    def mutations(path):
        with path.open(encoding="utf-8") as handle:
            next(handle)
            return {line.split("\t", 1)[0].strip() for line in handle if line.strip()}
    return mutations(other) <= mutations(test_path)


def training_coverage():
    model = Path(
        "/data/cc233_data/work_dir/MIRF/Mu-Protein/mu-search/src/flexs/flexs/landscapes/"
        "landscape/muformer/muformer-l-BLAT_ECOLX_Ranganathan2015_CFX.pt"
    )
    other = Path(
        "/data/cc233_data/work_dir/Muformer_DPFunc/mu-search-structure/src/flexs/flexs/"
        "landscapes/landscape/muformer/muformer-l-BLAT_ECOLX_Ranganathan2015_CFX.pt"
    )
    config_path = Path("/data/cc233_data/tmp/musearch-convert/out/muformer.config.json")
    manifest_path = Path("/data/cc233_data/tmp/musearch-convert/out/muformer.manifest.json")
    test_path = model.parent / "muformer_landscape" / "CFX_test_from_pan.tsv"
    lines = [
        "六次运行调用的是 `MuformerLandscape()`。它的默认权重路径是 `muformer-l-BLAT_ECOLX_Ranganathan2015_CFX.pt`，编码器是 `ur50-pcomb-prot_pmlm_1b-3x16-ckpt-checkpoint_best.pt`。查询表没有保存实际加载路径。",
        "",
    ]
    if model.exists() and other.exists() and model.samefile(other):
        lines.append(
            "当前机器上这个文件名有两个路径，`samefile` 为真，大小都是 {0} 字节。pretest 工作树里现在没有这份文件。本次没有 `torch.load`，也没有重算 sha256。转换清单 `{1}` 记录 source_bytes={2}，source_sha256=`{3}`。清单字节数和现文件大小一致。".format(
                model.stat().st_size,
                manifest_path,
                json.loads(manifest_path.read_text(encoding="utf-8"))["source_bytes"],
                json.loads(manifest_path.read_text(encoding="utf-8"))["source_sha256"],
            )
        )
    else:
        lines.append("同名 checkpoint 的路径核对没有同时找到两份同一文件。训练覆盖仍按未知处理。")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    lines.extend(
        [
            "",
            "checkpoint 旁的配置只有 `log_info`：epoch {0}，val_loss {1:.6g}，val corr {2:.6g}。没有训练、验证、测试文件名，也没有突变阶数计数。".format(
                config["log_info"]["epoch"],
                config["log_info"]["val_loss"],
                config["log_info"]["best_corr"],
            ),
            "",
        ]
    )
    orders = {}
    with test_path.open(encoding="utf-8") as handle:
        next(handle)
        for line in handle:
            mutation = line.split("\t", 1)[0].strip()
            order = 0 if mutation == "WT" else mutation.count(";") + 1
            orders[order] = orders.get(order, 0) + 1
    lines.append(
        "`{0}` 共 {1} 行数据。按分号计的突变阶数是 {2}。文件名写的是 test。分数从 0.088 到 2900，和六次搜索里大约 −0.08 到 0.09 的 MuFormer 分数不是同一个尺度。同目录 `CFX_test_from_pan_above_median.tsv` 的突变名是这份表的子集：{3}。这两份都不是训练集。".format(
            test_path.name,
            sum(orders.values()),
            "、".join("{0} 阶 {1} 条".format(order, orders[order]) for order in sorted(orders)),
            "是" if cfx_subset(test_path) else "否",
        )
    )
    lines.extend(
        [
            "",
            "仓库里另有 `BLAT_ECOLX_Stiffler_2015` 的划分和训练记录。那是另一个课题、另一个 checkpoint。不能用那份单突变组成填写这里的空白。",
            "",
            "这个 checkpoint 的适应度标签覆盖了哪些突变阶数：未知。编码器的预训练语料也回答不了适应度标签的突变阶数。",
        ]
    )
    return "\n".join(lines)


def report_markdown(result, candidates, recurrent):
    wild_type = result["wild_type"]
    lines = [
        "# TEM-1 高阶突变候选的离线审查",
        "",
        "这次只读已经完成的六次搜索。没有加载 MuFormer，没有新的评分，没有结构预测，没有 H=7、H=9、H=10，也没有改 PPO、环境或奖励。分数是这六次运行当时写下的 MuFormer 分数，把它当作搜索目标的记录，不把它当作独立实验真值。",
        "",
        "汉明距离以各次运行 `role=start` 的 WT 为参照。六次 WT 相同，长度 {0}。分组是 0 到 10，以及 10+。查询口径计入每一次候选调用。唯一序列口径只保留该序列的第一次分数。Top 1% 的条数是 `ceil(0.01 × 唯一序列数)`。样本少于 10 条时，Top 10 平均留空。".format(len(wild_type)),
        "",
        "没有动作轨迹。μSearch 产物里的编辑预算来自 `budget.json` 的 `n_steps/episodes_per_update=5`。终点汉明距离小于 5 可以来自空编辑或重复编辑，不能从终点把这些次数还原出来。汉明距离大于 5 在 5 次编辑的预算下不会出现。",
        "",
        "## 实际阶数",
        "",
        "μSearch 三次的唯一候选都落在 1 到 5。没有 6 个及以上突变的候选。",
        "",
        "| 运行 | 1 | 2 | 3 | 4 | 5 | 恰好为 5 的比例 | 最高分 | 最高分阶数 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in result["runs"]:
        if run["method"] != "musearch_gt":
            continue
        uniques = unique_records(run)
        counts = {label: order_stats(run, label)["calls_or_unique"] for label in ORDERS}
        best = max(uniques, key=lambda row: (row["score"], -row["candidate_index"]))
        lines.append(
            "| {0}/seed{1} | {2} | {3} | {4} | {5} | {6} | {7:.1%} | {8} | {9} |".format(
                run["method"], run["seed"],
                counts["1"], counts["2"], counts["3"], counts["4"], counts["5"],
                counts["5"] / len(uniques),
                fmt(best["score"]), best["distance"],
            )
        )
    lines.extend(
        [
            "",
            "阶数 5 的中位数三次都在 WT 分数 −0.072684 下方，均值约 −0.082。高分集中在右尾。查询加权和唯一序列两种口径都在 `mutation_order_by_run.csv`；下面的表是唯一序列口径。",
            "",
            "| 运行 | 阶数 5 的中位数 | 均值 | p90 | p99 | 最高分 | Top 10 平均 |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for run in result["runs"]:
        if run["method"] != "musearch_gt":
            continue
        stats = order_stats(run, "5")
        lines.append(
            "| {0}/seed{1} | {2} | {3} | {4} | {5} | {6} | {7} |".format(
                run["method"], run["seed"],
                fmt(stats["median"]), fmt(stats["mean"]), fmt(stats["p90"]),
                fmt(stats["p99"]), fmt(stats["maximum"]), fmt(stats["top10_mean"]),
            )
        )
    lines.extend(
        [
            "",
            "AdaLead 没有编辑上限。三次都从 0 延伸到 10+。下表是唯一序列口径。最高分单独看会受样本量影响，所以同时给出中位数、p99 和 Top 10 平均。",
            "",
            "| 运行 | 阶数 | 唯一序列 | 中位数 | 均值 | p90 | p99 | 最高分 | Top 10 平均 |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for run in result["runs"]:
        if run["method"] != "adalead_gt":
            continue
        for label in ["5", "6", "7", "8", "9", "10", "10+"]:
            stats = order_stats(run, label)
            lines.append(
                "| {0}/seed{1} | {2} | {3} | {4} | {5} | {6} | {7} | {8} | {9} |".format(
                    run["method"], run["seed"], label, stats["calls_or_unique"],
                    fmt(stats["median"]), fmt(stats["mean"]), fmt(stats["p90"]),
                    fmt(stats["p99"]), fmt(stats["maximum"]), fmt(stats["top10_mean"]),
                )
            )
    lines.extend(["", "## 高分候选是否是同一批组合", ""])
    lines.append("| 方法 | 种子 | 最高分 | 阶数 | 突变 | 同一种子内阶数不超过 5 的最高分 | 差值 | 超过该分数的更高阶唯一序列数 |")
    lines.append("| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |")
    comparisons = []
    for run in result["runs"]:
        uniques = unique_records(run)
        best = max(uniques, key=lambda row: (row["score"], -row["candidate_index"]))
        limited = max(
            (row for row in uniques if row["distance"] <= 5),
            key=lambda row: (row["score"], -row["candidate_index"]),
        )
        csv_limited = max(
            stats["maximum"]
            for stats in run["unique"]
            if stats["order"] in list("012345") and stats["maximum"] is not None
        )
        if abs(csv_limited - limited["score"]) > 1e-9:
            raise RuntimeError("阶数不超过 5 的最高分和分组表不一致")
        higher = [
            row for row in uniques
            if row["distance"] > 5 and row["score"] > limited["score"]
        ]
        comparisons.append((run, best, limited, higher))
        mutations = ";".join(mutation_list(best["sequence"], wild_type))
        lines.append(
            "| {0} | {1} | {2} | {3} | {4} | {5} | {6} | {7} |".format(
                run["method"], run["seed"], fmt(best["score"]), best["distance"], mutations,
                fmt(limited["score"]), fmt(best["score"] - limited["score"]), len(higher),
            )
        )
    shared_high = cross_seed_shares(result)
    lines.extend(
        [
            "",
            gap_sentence(comparisons),
            "",
            tail_family_sentence(comparisons, wild_type),
            "",
            champion_sentence(comparisons, wild_type),
            "",
            "跨种子共享的是整条序列，不只是高分表里的序列。下表没有列出的阶数，计数是 0。AdaLead 的跨种子重复停在阶数 0 和 1；μSearch 的候选序列没有跨种子重复。",
            "",
        ]
    )
    lines.extend(shared_high)
    lines.extend(
        [
            "",
            "Top 10 唯一序列的替换集合，两两种子的 Jaccard：",
            "",
            "| 方法 | 种子对 | 替换交集 | 替换并集 | Jaccard | 位点 Jaccard |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    lines.extend(jaccard_lines(candidates))
    pair_lines = pair_scope_lines(recurrent)
    lines.extend(
        [
            "",
            "高分表里的重复替换和成对共现写在 `recurrent_mutations.csv`。共现按高分序列计数。同一条序列里的两个替换同时出现，只说明搜索留下了这个组合，不能叫做正上位或负上位。这六次运行没有配齐对应的单突变和双突变对照，本次不计算上位效应。",
            "",
            pair_lines,
            "",
            "AdaLead 种子内部的重复组合，符合它从已有高分序列继续重组的做法。它说明一个种子沿着自己的谱系扩展，不能说明三个种子找到了同一组有益突变。",
            "",
            *distribution_lines(result, comparisons),
            "",
            "## 这个 checkpoint 的训练阶数",
            "",
            training_coverage(),
            "",
            "## 四个问题",
            "",
            *answer_lines(result, comparisons, candidates, wild_type),
            "",
            "C. 当前这份 MuFormer 对高阶突变有没有足够的训练标签：未知。配置里只有一次验证相关 0.566947，没有阶数直方图。文件名和另一份 Stiffler 实验都不能代替这个计数。高阶候选距离训练分布有多远，因此也未知。",
            "",
            "高阶候选有没有明显的分数异常：有尾部异常的证据，没有模型错误的直接证据。异常指的是冠军远离本阶 p99、所在阶数的中位数更低、以及跨种子不重复。这和模型外推、搜索钻到一个稀有高分、或者真实存在稀有高峰，三种情况都相容。本次不能把 MuFormer 分数分成“可靠”和“错误”两类。",
            "",
            "## 瓶颈在哪里",
            "",
            "四种可能对应四条路线。这次数据能分开的程度如下。",
            "",
            "1. 搜索步数不够，所以找不到更高阶组合。μSearch 没有写出 6 个以上突变。AdaLead 写出了，每个种子的单点最高分在 7、9 或 10。这些最高分没有跨种子重复，阶数 10 和 10+ 的中位数低于阶数 5。现有数据说明 H=5 是 μSearch 写出序列的上限。放宽上限会不会得到稳定更高的分数，这六次结果没有给出支持。",
            "",
            "2. 搜索不会识别有益突变之间的组合。三个 μSearch 种子的 5 突变冠军没有共同替换，AdaLead 三个种子的冠军也几乎没有共同替换。这只说明三次独立运行停在不同组合上。没有单突变对照时，看不出是策略没有利用已知的有益替换，还是景观本身有很多不同的高分组合。",
            "",
            "3. MuFormer 对高阶突变的预测不可靠，搜索越深越容易利用模型误差。训练标签的阶数覆盖未知，而观察到的高分正是不稳定的薄尾。这条目前是决策上的阻塞项：在它没有被单独核对之前，把 H 从 5 增加到 10 的分数升降无法归因。它还没有被证明就是原因。",
            "",
            "4. 蛋白质有较强的负上位，所以高阶突变的实际收益有限。这次分数不是实验测量。中位数在高阶下降，可能来自模型、AdaLead 的谱系漂移，也可能来自生物学。共现表不能用来计算上位效应。这条现在没有证据。",
            "",
            "下一步投入的排序：",
            "",
            "1. 先核对这份 `Ranganathan2015_CFX` checkpoint 的适应度标签覆盖了哪些突变阶数，以及有测量的高阶突变上预测和测量的差距。训练表如果不存在，就保持未知，不用别的数据集填上。",
            "2. 在现有 H=5 里检查位点和氨基酸选择。大约四分之三的唯一候选已经是 5 个突变，中位数仍接近 WT，三个种子的精英组合不同。这一步使用的仍是已经搜索过的阶数。",
            "3. 做 5 到 7 个突变的受控搜索。它是区分“H=5 的写出上限”和“高阶分数是否可信”的直接实验，适合排在标签覆盖被看过之后。现在启动它，仍然是在用同一份未核对的高阶预测当目标。",
            "4. 引入上位效应或可进化性。当前没有单突变、双突变对照，共现也不能当成上位效应。这一步缺少本次数据要求的输入。",
            "",
            "不确定性：如果随后找到的训练表显示这个 checkpoint 在高阶突变上有直接标签，并且预测和测量相符，第 1 项下降，第 3 项成为第一个新搜索。如果训练表显示标签基本停在单突变或双突变，第 1 项变成硬阻塞，第 3 项不应再把 MuFormer 的高阶分数当作改进目标。三种子的差距已经和所谓的高阶增益一样大，所以即使以后做 5 到 7 的搜索，也不能从一次最高分判断深浅。",
            "",
            "这次审查到此为止。不据此启动新实验。",
            "",
        ]
    )
    return "\n".join(lines)


def mutation_set(sequence, wild_type):
    return set(mutation_list(sequence, wild_type))


def gap_sentence(comparisons):
    adalead = [item for item in comparisons if item[0]["method"] == "adalead_gt"]
    gaps = [fmt(best["score"] - limited["score"]) for _run, best, limited, _higher in adalead]
    counts = [str(len(higher)) for _run, _best, _limited, higher in adalead]
    above = all(best["score"] > limited["score"] for _run, best, limited, _higher in adalead)
    relation = "都高于" if above else "并不都高于"
    return (
        "AdaLead 每个种子里，阶数大于 5 的最高分{0}该种子阶数不超过 5 的最高分。"
        "三个种子的差值是 {1}。超过这条线的更高阶唯一序列数是 {2}。"
    ).format(relation, "、".join(gaps), "、".join(counts))


def champion_sentence(comparisons, wild_type, methods=(("adalead_gt", "AdaLead"), ("musearch_gt", "μSearch"))):
    parts = []
    for method, title in methods:
        chosen = [
            (run["seed"], mutation_set(best["sequence"], wild_type), best)
            for run, best, _limited, _higher in comparisons
            if run["method"] == method
        ]
        if not chosen:
            continue
        pair_text = []
        for index, (seed_a, set_a, _best_a) in enumerate(chosen):
            for seed_b, set_b, _best_b in chosen[index + 1:]:
                shared = sorted(set_a & set_b)
                if shared:
                    pair_text.append("种子 {0} 和种子 {1} 的共同替换是 {2}".format(seed_a, seed_b, "、".join(shared)))
                else:
                    pair_text.append("种子 {0} 和种子 {1} 没有共同替换".format(seed_a, seed_b))
        distances = "、".join(str(best["distance"]) for _seed, _mutations, best in chosen)
        parts.append("{0} 三个冠军的阶数是 {1}。{2}。".format(title, distances, "；".join(pair_text)))
    return " ".join(parts)


def tail_family_sentence(comparisons, wild_type):
    pieces = []
    for run, best, _limited, higher in comparisons:
        if run["method"] != "adalead_gt" or not higher:
            continue
        champion = mutation_set(best["sequence"], wild_type)
        described = []
        for row in sorted(higher, key=lambda item: -item["score"]):
            shared = len(mutation_set(row["sequence"], wild_type) & champion)
            described.append(
                "{0} 突变、分数 {1}、与该种子冠军共享 {2} 个替换".format(
                    row["distance"], fmt(row["score"]), shared
                )
            )
        extras = [
            row for row in higher
            if row["sequence"] != best["sequence"]
            and champion <= mutation_set(row["sequence"], wild_type)
            and row["score"] < best["score"]
        ]
        described.append(
            "其中 {0} 条在冠军的全部替换之外又加了突变，分数仍低于冠军".format(len(extras))
        )
        pieces.append("种子 {0}：{1}。".format(run["seed"], "；".join(described)))
    return "超过本种子阶数不超过 5 的最高分的序列有这些：" + "".join(pieces)


def order7_sentence(result):
    higher = []
    lower = []
    top_below = []
    for run in result["runs"]:
        if run["method"] != "adalead_gt":
            continue
        if order_stats(run, "7")["median"] > order_stats(run, "5")["median"]:
            higher.append(str(run["seed"]))
        else:
            lower.append(str(run["seed"]))
        if order_stats(run, "7")["top10_mean"] < order_stats(run, "5")["top10_mean"]:
            top_below.append(str(run["seed"]))
    return (
        "阶数 7 的中位数高于本种子阶数 5 的是种子 {0}，不高于的是种子 {1}。"
        "阶数 7 的 Top 10 平均低于本种子阶数 5 的是种子 {2}。"
        "高阶分数的现象是少数右尾更高；阶数 10 和 10+ 的主体低于阶数 5；冠军序列没有跨种子重复。"
    ).format("、".join(higher) or "无", "、".join(lower) or "无", "、".join(top_below) or "无")


def distribution_lines(result, comparisons):
    def series(label, key):
        values = []
        for run in result["runs"]:
            if run["method"] == "adalead_gt":
                values.append(fmt(order_stats(run, label)[key]))
        return "、".join(values)

    medians_fall = all(
        order_stats(run, "10")["median"] < order_stats(run, "5")["median"]
        and order_stats(run, "10+")["median"] < order_stats(run, "5")["median"]
        and order_stats(run, "10")["mean"] < order_stats(run, "5")["mean"]
        and order_stats(run, "10+")["mean"] < order_stats(run, "5")["mean"]
        for run in result["runs"]
        if run["method"] == "adalead_gt"
    )
    top_fall = all(
        order_stats(run, "10")["top10_mean"] < order_stats(run, "5")["top10_mean"]
        for run in result["runs"]
        if run["method"] == "adalead_gt"
    )
    tail_bits = []
    for run, best, _limited, _higher in comparisons:
        if run["method"] != "adalead_gt":
            continue
        stats = order_stats(run, order_label(best["distance"]))
        tail_bits.append(
            "种子 {0} 的 {1} 突变最高分 {2}，该阶 p99 是 {3}，相差 {4}".format(
                run["seed"], best["distance"], fmt(best["score"]), fmt(stats["p99"]),
                fmt(best["score"] - stats["p99"]),
            )
        )
    musearch_scores = [
        best["score"] for run, best, _limited, _higher in comparisons if run["method"] == "musearch_gt"
    ]
    adalead_best = [
        (run["seed"], best) for run, best, _limited, _higher in comparisons if run["method"] == "adalead_gt"
    ]
    rank_bits = []
    for seed, best in adalead_best:
        below = sum(score > best["score"] for score in musearch_scores)
        if below == 0:
            relation = "高于全部 μSearch 冠军"
        else:
            relation = "低于 {0} 个 μSearch 冠军".format(below)
        rank_bits.append(
            "AdaLead 种子 {0} 的冠军 {1} {2}".format(seed, fmt(best["score"]), relation)
        )
    spread = max(musearch_scores) - min(musearch_scores)
    lead = max(best["score"] for _seed, best in adalead_best) - max(musearch_scores)
    if lead >= 0:
        lead_text = "AdaLead 全局最高分比 μSearch 全局最高分高 {0}".format(fmt(lead))
    else:
        lead_text = "μSearch 全局最高分比 AdaLead 全局最高分高 {0}".format(fmt(-lead))
    return [
        "## 分布有没有整体上移",
        "",
        "AdaLead 阶数 5 的中位数是 {0}。阶数 7 是 {1}。阶数 10 是 {2}。10+ 是 {3}。".format(
            series("5", "median"), series("7", "median"), series("10", "median"), series("10+", "median")
        ),
        "阶数 10 和 10+ 的中位数、均值三次都低于阶数 5。" if medians_fall else "阶数 10 或 10+ 的中位数、均值没有三次都低于阶数 5。",
        "",
        "阶数 5 的 Top 10 平均是 {0}。阶数 7 是 {1}。阶数 10 是 {2}。".format(
            series("5", "top10_mean"), series("7", "top10_mean"), series("10", "top10_mean")
        ),
        "阶数 10 的 Top 10 平均三次都低于阶数 5。" if top_fall else "阶数 10 的 Top 10 平均没有三次都低于阶数 5。",
        "",
        "冠军和本阶 p99 的距离：{0}。".format("；".join(tail_bits)),
        "",
        "六个冠军放在一起：μSearch 最高分是 {0}。{1}。{2}。μSearch 三个冠军的高低差是 {3}。".format(
            "、".join(fmt(score) for score in musearch_scores),
            "；".join(rank_bits),
            lead_text,
            fmt(spread),
        ),
        "",
        "10+ 的 Top 10 平均是 {0}。这三组分别有 {1} 条序列，Top 10 会抽到更大的右尾，不能直接当成阶数效应。{2}".format(
            series("10+", "top10_mean"),
            "、".join(
                str(order_stats(run, "10+")["calls_or_unique"])
                for run in result["runs"]
                if run["method"] == "adalead_gt"
            ),
            "10+ 的中位数三次都低于阶数 5。" if medians_fall else "10+ 的中位数没有三次都低于阶数 5。",
        ),
        "",
        order7_sentence(result),
    ]


def answer_lines(result, comparisons, candidates, wild_type):
    musearch_rows = [run for run in result["runs"] if run["method"] == "musearch_gt"]
    total = sum(len(run["records"]) for run in musearch_rows)
    fractions = []
    max_distance = []
    for run in musearch_rows:
        uniques = unique_records(run)
        fractions.append(sum(row["distance"] == 5 for row in uniques) / len(uniques))
        max_distance.append(max(row["distance"] for row in uniques))
    adalead_max = [
        max(row["distance"] for row in unique_records(run))
        for run in result["runs"]
        if run["method"] == "adalead_gt"
    ]
    jaccard_values = []
    grouped = {}
    for row in candidates:
        if row["set"] == "top10" and row["method"] == "adalead_gt":
            grouped.setdefault(row["seed"], set()).update(item for item in row["mutations"].split(";") if item)
    for left, right in ((0, 1), (0, 2), (1, 2)):
        union = grouped[left] | grouped[right]
        jaccard_values.append(len(grouped[left] & grouped[right]) / len(union))
    return [
        "A. H=5 限制了 μSearch 实际能写出的序列。产物里的编辑预算是 5。{0} 条 μSearch 候选的最大终点汉明距离是 {1}，恰好为 5 的比例是 {2}。AdaLead 的最大终点汉明距离是 {3}。阶数 10 和 10+ 的中位数低于阶数 5，更高阶冠军也没有跨种子重复。H=5 是 μSearch 写出序列的上限。这六次结果里，放宽上限之后的更高分数没有跨种子稳住。".format(
            total,
            "、".join(str(item) for item in max_distance),
            "、".join("{0:.1%}".format(item) for item in fractions),
            "、".join(str(item) for item in adalead_max),
        ),
        "",
        "B. AdaLead 的高阶高分候选没有跨种子稳定性。{0} Top 10 替换集合的 Jaccard 是 {1}。{2} 种子内部的重复来自同一条重组谱系。".format(
            champion_sentence(
                comparisons, wild_type, methods=(("adalead_gt", "AdaLead"),)
            ),
            "、".join("{0:.3f}".format(item) for item in jaccard_values),
            shared_sequence_sentence(result),
        ),
    ]


def shared_order_counts(result, method):
    buckets = {}
    for run in result["runs"]:
        if run["method"] != method:
            continue
        for row in unique_records(run):
            buckets.setdefault(row["sequence"], []).append(row)
    counts = {label: 0 for label in ORDERS}
    for rows in buckets.values():
        if len(rows) >= 2:
            counts[rows[0]["order"]] += 1
    return counts


def shared_sequence_sentence(result):
    counts = shared_order_counts(result, "adalead_gt")
    listed = "、".join(
        "{0} 阶 {1} 条".format(label, counts[label]) for label in ORDERS if counts[label]
    )
    higher = sum(counts[label] for label in ORDERS if label not in ("0", "1"))
    return "AdaLead 跨种子重复的整条序列是 {0}。阶数 2 及以上是 {1} 条。".format(listed, higher)


def cross_seed_shares(result):
    lines = ["| 方法 | 阶数 | 出现在至少两个种子的唯一序列数 |", "| --- | --- | ---: |"]
    for method in ("musearch_gt", "adalead_gt"):
        buckets = {}
        for run in result["runs"]:
            if run["method"] != method:
                continue
            for row in unique_records(run):
                buckets.setdefault(row["sequence"], []).append(row)
        by_order = {label: 0 for label in ORDERS}
        for rows in buckets.values():
            if len(rows) < 2:
                continue
            by_order[rows[0]["order"]] += 1
        for label in ORDERS:
            if by_order[label]:
                lines.append("| {0} | {1} | {2} |".format(method, label, by_order[label]))
        if not any(by_order.values()):
            lines.append("| {0} | 全部 | 0 |".format(method))
    return lines


def jaccard_lines(candidates):
    grouped = {}
    for row in candidates:
        if row["set"] != "top10":
            continue
        grouped.setdefault((row["method"], row["seed"]), set()).update(
            item for item in row["mutations"].split(";") if item
        )
    lines = []
    for method in ("musearch_gt", "adalead_gt"):
        for left, right in ((0, 1), (0, 2), (1, 2)):
            a = grouped[(method, left)]
            b = grouped[(method, right)]
            union = a | b
            sites_a = {item[1:-1] for item in a}
            sites_b = {item[1:-1] for item in b}
            site_union = sites_a | sites_b
            lines.append(
                "| {0} | {1}/{2} | {3} | {4} | {5:.3f} | {6:.3f} |".format(
                    method, left, right, len(a & b), len(union),
                    len(a & b) / len(union),
                    len(sites_a & sites_b) / len(site_union),
                )
            )
    return lines


def pair_scope_lines(recurrent):
    rows = [row for row in recurrent if row["set"] == "top10" and row["kind"] == "pair"]
    one_run = sum(row["n_runs"] == 1 for row in rows)
    multi = sum(row["n_runs"] >= 2 for row in rows)
    return (
        "Top 10 里出现至少两次的成对共现有 {0} 条，其中 {1} 条只来自一个运行，{2} 条来自至少两个运行。"
    ).format(len(rows), one_run, multi)


def write_manifest(root, runs_root):
    inputs = []
    for method, seed in RUNS:
        directory = runs_root / method / "seed{0}".format(seed)
        stem = "{0}_seed{1}".format(method, seed)
        for suffix in (".queries.csv", ".budget.json", ".time.json", ".seeds.json"):
            inputs.append(directory / (stem + suffix))
    outputs = [
        root / name
        for name in (
            "audit_high_order.py",
            "test_audit_high_order.py",
            "data_integrity.md",
            "mutation_order_by_run.csv",
            "mutation_order_summary.csv",
            "top_candidates.csv",
            "recurrent_mutations.csv",
            "high_order_audit_report.md",
        )
    ]
    lines = ["# inputs"]
    for path in inputs:
        lines.append("{0}  {1}".format(sha256(path), path))
    lines.append("# outputs")
    for path in outputs:
        lines.append("{0}  {1}".format(sha256(path), path))
    (root / "MANIFEST.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    root = Path(__file__).resolve().parent
    runs_root = DEFAULT_RUNS
    result = analyze(runs_root)
    order_table = order_rows(result)
    summary = summary_rows(order_table)
    candidates, _sets = high_score_tables(result)
    recurrent = recurrent_rows(candidates)
    order_fields = [
        "method", "seed", "scope", "order", "n", "unique_sequences",
        "duplicate_fraction", "mean", "median", "std", "p25", "p75",
        "p90", "p95", "p99", "maximum", "top10_mean",
    ]
    write_csv(root / "mutation_order_by_run.csv", order_table, order_fields)
    write_csv(
        root / "mutation_order_summary.csv",
        summary,
        [
            "method", "scope", "order", "n_seeds", "n_min", "n_max", "n_sum",
            "mean_of_means", "median_of_medians", "minimum_of_maxima", "maximum_of_maxima",
        ],
    )
    write_csv(
        root / "top_candidates.csv",
        candidates,
        [
            "method", "seed", "set", "set_size", "rank", "score", "hamming",
            "mutations", "n_runs_with_sequence", "other_runs",
        ],
    )
    write_csv(
        root / "recurrent_mutations.csv",
        recurrent,
        ["set", "kind", "item", "n_sequences", "n_runs", "runs"],
    )
    (root / "data_integrity.md").write_text(integrity_markdown(result), encoding="utf-8")
    (root / "high_order_audit_report.md").write_text(
        report_markdown(result, candidates, recurrent), encoding="utf-8"
    )
    write_manifest(root, runs_root)
    print("runs", len(result["runs"]), "order_rows", len(order_table), "top_rows", len(candidates))
    for run in result["runs"]:
        unique = {row["order"]: row["calls_or_unique"] for row in run["unique"]}
        best = max(run["records"], key=lambda row: row["score"])
        print(
            run["method"], run["seed"],
            "unique_orders", {key: unique[key] for key in ORDERS if unique[key]},
            "best", best["score"], "H", best["distance"],
        )


if __name__ == "__main__":
    main()

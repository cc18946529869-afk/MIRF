"""Persist one designated run and resume it from the last finished stage.

``Trainer.train`` keeps ``save_checkpoint=False``. Fold models return at the
requested epoch and do not call ``Trainer.save_checkpoint``. This module
writes the weights after that return. It does not call the commented
``torch.load`` inside ``trainer.py``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import metrics

MARKER_NAME = "designated_run.json"
FULL_NAME = "full"
FOLD_COUNT = 5


def sequence_sha256(sequences) -> str:
    payload = "\n".join(str(item) for item in sequences).encode()
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if not np.isfinite(number) else number
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"the run record cannot store {type(value).__name__}")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(_json_ready(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def marker_path(output_dir: Path) -> Path:
    return Path(output_dir) / MARKER_NAME


def read_marker(output_dir: Path) -> dict:
    return _read_json(marker_path(output_dir))


def open_run(output_dir: Path, *, resume: bool, schedule_only: bool) -> dict:
    """Create the run marker, or continue an incomplete directory.

    A finished directory is refused. An incomplete directory is refused until
    ``resume=True``. ``resume=True`` does not create a new run.
    """
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = marker_path(directory)
    if not path.exists():
        if resume:
            raise RuntimeError("resume requires an incomplete designated run in this directory")
        record = {
            "scale": "n1000",
            "outer_seed": 1975576344,
            "status": "running",
            "stage": "starting",
            "completed_stages": [],
            "schedule_only": bool(schedule_only),
            "error": None,
        }
        _write_json(path, record)
        return record
    record = _read_json(path)
    if record.get("status") == "complete":
        raise RuntimeError("the designated single-seed run already finished in this directory")
    if not resume:
        raise RuntimeError(
            "this directory has an incomplete designated run; pass resume=True to continue from the saved stages"
        )
    if bool(record.get("schedule_only")) != bool(schedule_only):
        raise RuntimeError("resume does not switch between the schedule check and the complete run")
    record["status"] = "running"
    record["error"] = None
    _write_json(path, record)
    return record


def note_stage(output_dir: Path, stage: str, **fields) -> None:
    path = marker_path(output_dir)
    record = _read_json(path)
    completed = [str(item) for item in record.get("completed_stages", [])]
    if stage not in completed:
        completed.append(stage)
    record["completed_stages"] = completed
    record["stage"] = stage
    record["status"] = "running"
    record["error"] = None
    for key, value in fields.items():
        record[key] = _json_ready(value)
    _write_json(path, record)


def mark_failed(output_dir: Path, stage: str, error: BaseException) -> None:
    path = marker_path(output_dir)
    if not path.exists():
        return
    record = _read_json(path)
    if record.get("status") == "complete":
        return
    record["status"] = "failed"
    record["stage"] = stage
    record["error"] = f"{type(error).__name__}: {error}"
    _write_json(path, record)


def mark_complete(output_dir: Path) -> None:
    path = marker_path(output_dir)
    record = _read_json(path)
    record["status"] = "complete"
    record["stage"] = "report"
    record["error"] = None
    _write_json(path, record)


def weights_path(directory: Path, name: str) -> Path:
    return Path(directory) / "muformer" / f"{name}.pt"


def _weights_path(directory: Path, name: str) -> Path:
    return weights_path(directory, name)


def _record_path(directory: Path, name: str) -> Path:
    return directory / "muformer" / f"{name}.json"


def module_state(trainer) -> dict:
    models = getattr(trainer, "models", None)
    if not models:
        raise RuntimeError("the trained μFormer has no weights to save")
    bare = trainer.get_bare_model(models[0])
    return {key: value.detach().cpu().clone() for key, value in bare.state_dict().items()}


def save_weights(path: Path, state: dict) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)
    return file_sha256(path)


def load_weights(trainer, path: Path) -> None:
    state = torch.load(Path(path), map_location="cpu", weights_only=True)
    bare = trainer.get_bare_model(trainer.models[0])
    bare.load_state_dict(state)
    digest = file_sha256(path)
    record_path = Path(path).with_suffix(".json")
    if record_path.is_file():
        recorded = _read_json(record_path).get("weight_sha256")
        if recorded is not None and recorded != digest:
            raise RuntimeError("the saved μFormer weights do not match their recorded sha256")


def _vector(values) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("a saved score must be one vector")
    return [float(item) for item in array.tolist()]


def write_full(directory: Path, trainer, result: dict) -> None:
    directory = Path(directory)
    sequences = [str(item) for item in np.asarray(result["sequences"]).tolist()]
    digest = save_weights(_weights_path(directory, FULL_NAME), module_state(trainer))
    _write_json(_record_path(directory, FULL_NAME), {
        "e_s": int(result["e_s"]),
        "epochs_run": int(result["epochs_run"]),
        "restore_best": bool(result["restore_best"]),
        "replaced_with_earlier_best": bool(result["replaced_with_earlier_best"]),
        "split": str(result["split"]),
        "score": _vector(result["score"]),
        "sequences": sequences,
        "sequence_sha256": sequence_sha256(sequences),
        "weight_sha256": digest,
    })
    note_stage(directory, "full_model", e_s=int(result["e_s"]))


def read_full(directory: Path) -> dict | None:
    directory = Path(directory)
    weights = _weights_path(directory, FULL_NAME)
    record_path = _record_path(directory, FULL_NAME)
    if not weights.exists() and not record_path.exists():
        return None
    if not weights.is_file() or not record_path.is_file():
        raise RuntimeError("the full-model checkpoint is incomplete")
    record = _read_json(record_path)
    sequences = [str(item) for item in record["sequences"]]
    if sequence_sha256(sequences) != record["sequence_sha256"]:
        raise RuntimeError("the saved validation sequences do not match their sha256")
    if file_sha256(weights) != record["weight_sha256"]:
        raise RuntimeError("the saved full-model weights do not match their sha256")
    if record["split"] != "validation":
        raise RuntimeError("the saved full model was not scored on the validation split")
    return {
        "e_s": int(record["e_s"]),
        "epochs_run": int(record["epochs_run"]),
        "restore_best": bool(record["restore_best"]),
        "replaced_with_earlier_best": bool(record["replaced_with_earlier_best"]),
        "split": "validation",
        "score": np.asarray(record["score"], dtype=np.float64),
        "sequences": np.asarray(sequences),
    }


def write_fold(directory: Path, trainer, fold: int, produced: dict) -> None:
    directory = Path(directory)
    sequences = [str(item) for item in produced["held_out_sequence"]]
    name = f"fold_{int(fold)}"
    digest = save_weights(_weights_path(directory, name), module_state(trainer))
    _write_json(_record_path(directory, name), {
        "fold": int(fold),
        "epochs_run": int(produced["epochs_run"]),
        "restore_best": bool(produced["restore_best"]),
        "replaced_with_earlier_best": bool(produced["replaced_with_earlier_best"]),
        "held_out_prediction": _vector(produced["held_out_prediction"]),
        "held_out_sequence": sequences,
        "sequence_sha256": sequence_sha256(sequences),
        "weight_sha256": digest,
    })


def read_fold(directory: Path, fold: int) -> dict | None:
    directory = Path(directory)
    name = f"fold_{int(fold)}"
    weights = _weights_path(directory, name)
    record_path = _record_path(directory, name)
    if not weights.exists() and not record_path.exists():
        return None
    if not weights.is_file() or not record_path.is_file():
        raise RuntimeError(f"fold {int(fold)} checkpoint is incomplete")
    record = _read_json(record_path)
    sequences = [str(item) for item in record["held_out_sequence"]]
    if sequence_sha256(sequences) != record["sequence_sha256"]:
        raise RuntimeError(f"fold {int(fold)} sequences do not match their sha256")
    if file_sha256(weights) != record["weight_sha256"]:
        raise RuntimeError(f"fold {int(fold)} weights do not match their sha256")
    return {
        "epochs_run": int(record["epochs_run"]),
        "restore_best": bool(record["restore_best"]),
        "replaced_with_earlier_best": bool(record["replaced_with_earlier_best"]),
        "held_out_prediction": np.asarray(record["held_out_prediction"], dtype=np.float64),
        "held_out_sequence": sequences,
    }


def write_oof(directory: Path, predictions, sequences, e_s: int, *, schedule_only: bool, weights_saved: bool) -> None:
    directory = Path(directory)
    names = [str(item) for item in sequences]
    values = _vector(predictions)
    if len(values) != len(names):
        raise ValueError("OOF predictions do not match the training sequences")
    _write_json(directory / "oof.json", {
        "rows": len(names),
        "e_s": int(e_s),
        "predictions": values,
        "sequence_sha256": sequence_sha256(names),
        "schedule_only": bool(schedule_only),
        "weights_saved": bool(weights_saved),
    })
    note_stage(directory, "oof", e_s=int(e_s), oof_rows=len(names))


def read_oof(directory: Path, sequences) -> dict:
    record = _read_json(Path(directory) / "oof.json")
    names = [str(item) for item in sequences]
    if record["sequence_sha256"] != sequence_sha256(names) or int(record["rows"]) != len(names):
        raise RuntimeError("the saved OOF predictions are not this training sequence list")
    values = np.asarray(record["predictions"], dtype=np.float64)
    if values.shape != (len(names),):
        raise RuntimeError("the saved OOF predictions are not this training sequence list")
    return record


def write_embedding_identity(directory: Path, train_meta: dict, valid_meta: dict, test_meta: dict) -> None:
    _write_json(Path(directory) / "pmlm_identity.json", {
        "train": train_meta,
        "valid": valid_meta,
        "test": test_meta,
        "embeddings_stored": False,
    })
    note_stage(directory, "embeddings")


def point_metrics(labels, sequences, scores) -> dict:
    y = np.asarray(labels, dtype=np.float64)
    names = np.asarray(sequences)
    values = np.asarray(scores, dtype=np.float64)
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    k = metrics.top_fraction_k(int(y.shape[0]))
    return {
        name: metrics.metric_value(name, y, values, names, y_min, y_max, k)
        for name in metrics.METRIC_NAMES
    }


def write_arms(directory: Path, arm_result: dict) -> None:
    directory = Path(directory)
    stored = {}
    for arm in arm_result["arms"]:
        digest = save_weights(directory / "arms" / f"{arm}.pt", arm_result["state_dict"][arm])
        stored[arm] = {
            "eta": arm_result["etas"][arm],
            "best_epoch": arm_result["best_epochs"][arm],
            "n_bootstrap": arm_result["n_bootstrap"][arm],
            "formal_experiment": False,
            "ten_seed_conclusion": False,
            "spearman_passes": arm_result["spearman_passes"][arm],
            "metrics": arm_result["metrics"][arm],
            "point": arm_result["point"][arm],
            "weight_sha256": digest,
        }
    _write_json(directory / "arms.json", {
        "arms": list(arm_result["arms"]),
        "shared_init": bool(arm_result["shared_init"]),
        "shared_first_order": bool(arm_result["shared_first_order"]),
        "formal_experiment": False,
        "ten_seed_conclusion": False,
        "by_arm": stored,
    })
    note_stage(directory, "arms")


def arms_ready(directory: Path) -> bool:
    path = Path(directory) / "arms.json"
    if not path.is_file():
        return False
    record = _read_json(path)
    for arm in record["arms"]:
        weights = Path(directory) / "arms" / f"{arm}.pt"
        if not weights.is_file() or file_sha256(weights) != record["by_arm"][arm]["weight_sha256"]:
            raise RuntimeError(f"the saved {arm} residual weights do not match their sha256")
    return True


def read_arms(directory: Path) -> dict:
    if not arms_ready(directory):
        raise RuntimeError("the three residual arms are not saved")
    return _read_json(Path(directory) / "arms.json")


def report_ready(directory: Path) -> bool:
    return (Path(directory) / "single_seed_report.json").is_file()


def write_report(directory: Path, report: dict) -> dict:
    directory = Path(directory)
    ready = _json_ready(report)
    _write_json(directory / "single_seed_report.json", ready)
    lines = [
        "# 单种子实验结果",
        "",
        f"规模 `{ready['scale']}`，外层种子 `{ready['outer_seed']}`。这是一次运行，不是十种子结论。",
        f"`formal_experiment` 为 `{str(ready['formal_experiment']).lower()}`，`ten_seed_conclusion` 为 `{str(ready['ten_seed_conclusion']).lower()}`。",
        f"该种子的 \\(E_s\\) 是 {ready['e_s']}。OOF 行数是 {ready['oof_rows']}。",
        "",
        "Spearman 的通过规则仍要求 10 个外层种子。这一次只有一个种子，所以 `spearman_passes` 不表示十种子结论。",
        "",
        "改善幅度是融合分数减去 μFormer baseline。RMSE 的改善幅度是 baseline 减去融合分数。区间是 10000 次配对 Bootstrap 的 2.5% 和 97.5% 分位。",
        "",
    ]
    for arm, payload in ready["by_arm"].items():
        lines.append(f"## {arm}")
        lines.append("")
        lines.append(f"\\(\\eta\\) = {payload['eta']}，残差最佳轮 = {payload['best_epoch']}。")
        lines.append("")
        lines.append("| 指标 | baseline | 融合 | 改善幅度 | 区间低 | 区间高 |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for name in metrics.METRIC_NAMES:
            point = payload["point"]
            delta = payload["metrics"][name]
            lines.append(
                f"| {name} | {point['baseline'][name]} | {point['fused'][name]} | "
                f"{delta['mean_delta']} | {delta['ci_low']} | {delta['ci_high']} |"
            )
        lines.append("")
    (directory / "single_seed_report.md").write_text("\n".join(lines))
    note_stage(directory, "report")
    return ready


def read_report(directory: Path) -> dict:
    path = Path(directory) / "single_seed_report.json"
    if not path.is_file():
        raise RuntimeError("the designated report is not saved")
    return _read_json(path)

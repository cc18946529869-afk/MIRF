"""Persist one designated run and resume it from the last finished stage.

``Trainer.train`` keeps ``save_checkpoint=False``. Fold models return at the
requested epoch and do not call ``Trainer.save_checkpoint``. This module
writes the weights after that return. It does not call the commented
``torch.load`` inside ``trainer.py``.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import metrics

MARKER_NAME = "designated_run.json"
LOCK_NAME = ".run.lock"
EXPERIMENT_ID = "n1000-1975576344"
DESIGNATED_SCALE = "n1000"
DESIGNATED_SEED = 1975576344
_HELD_LOCKS: dict[str, object] = {}
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


def _inside_repository(path: Path) -> bool:
    current = path.resolve()
    for parent in (current, *current.parents):
        if (parent / ".git").exists():
            return True
    return False


def _acquire_lock(directory: Path):
    key = str(directory.resolve())
    held = _HELD_LOCKS.get(key)
    if held is not None and not held.closed:
        return held
    handle = (directory / LOCK_NAME).open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        raise RuntimeError("another process already holds this designated run directory")
    _HELD_LOCKS[key] = handle
    return handle


def _release_lock(directory: Path) -> None:
    key = str(directory.resolve())
    handle = _HELD_LOCKS.pop(key, None)
    if handle is not None and not handle.closed:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _require_identity(record: dict) -> None:
    if record.get("experiment_id") != EXPERIMENT_ID:
        raise RuntimeError("this directory belongs to a different experiment")
    if record.get("scale") != DESIGNATED_SCALE or int(record.get("outer_seed")) != DESIGNATED_SEED:
        raise RuntimeError("this directory belongs to a different scale or outer seed")


def _open_locked(directory: Path, *, resume: bool, schedule_only: bool) -> dict:
    path = marker_path(directory)
    entries = [item for item in directory.iterdir() if item.name != LOCK_NAME]
    if not path.exists():
        if resume:
            raise RuntimeError("resume requires an incomplete designated run in this directory")
        if entries:
            raise RuntimeError("a new designated run requires an empty directory")
        record = {
            "experiment_id": EXPERIMENT_ID,
            "scale": DESIGNATED_SCALE,
            "outer_seed": DESIGNATED_SEED,
            "status": "running",
            "stage": "starting",
            "completed_stages": [],
            "schedule_only": bool(schedule_only),
            "training_started": False,
            "muformer_train_called": False,
            "error": None,
        }
        _write_json(path, record)
        return record
    record = _read_json(path)
    _require_identity(record)
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


def open_run(output_dir: Path, *, resume: bool, schedule_only: bool) -> dict:
    """Create the run marker, or continue an incomplete directory.

    A new run must be an empty directory outside any git repository. A
    directory that already holds another experiment's files is refused, and
    the marker is not created. Resume checks the scale, seed, and experiment
    id. The process that enters keeps an exclusive lock until it fails to
    enter or the process exits.
    """
    directory = Path(output_dir).resolve()
    if _inside_repository(directory):
        raise RuntimeError("the designated run directory must be outside the repository")
    directory.mkdir(parents=True, exist_ok=True)
    _acquire_lock(directory)
    try:
        return _open_locked(directory, resume=resume, schedule_only=schedule_only)
    except Exception:
        _release_lock(directory)
        raise


def recorded_training(output_dir: Path) -> dict:
    """Training flags stored for the whole experiment, not the latest process."""
    record = read_marker(output_dir)
    return {
        "training_started": bool(record.get("training_started")),
        "muformer_train_called": bool(record.get("muformer_train_called")),
    }


def remember_muformer_trained(output_dir: Path) -> None:
    """Record that this experiment called μFormer ``train``. Later resumes keep it."""
    path = marker_path(output_dir)
    record = _read_json(path)
    record["training_started"] = True
    record["muformer_train_called"] = True
    _write_json(path, record)


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


def _discard_pair(weights: Path, record_path: Path) -> None:
    for path in (
        weights,
        record_path,
        weights.with_name(weights.name + ".partial"),
        record_path.with_name(record_path.name + ".partial"),
        weights.with_name(weights.name + ".tmp"),
        record_path.with_name(record_path.name + ".tmp"),
    ):
        if path.is_file():
            path.unlink()


def _commit_pair(weights: Path, record_path: Path, state: dict, payload: dict) -> None:
    """Write the weight file and its JSON, then publish the pair.

    A crash before both names are in place leaves a partial pair. ``read``
    removes that pair so the model can be trained again.
    """
    weights.parent.mkdir(parents=True, exist_ok=True)
    partial_weights = weights.with_name(weights.name + ".partial")
    partial_record = record_path.with_name(record_path.name + ".partial")
    torch.save(state, partial_weights)
    body = dict(payload)
    body["weight_sha256"] = file_sha256(partial_weights)
    _write_json(partial_record, body)
    partial_weights.replace(weights)
    partial_record.replace(record_path)


def _read_committed(weights: Path, record_path: Path) -> dict | None:
    both = weights.is_file() and record_path.is_file()
    debris = any(
        path.exists()
        for path in (
            weights,
            record_path,
            weights.with_name(weights.name + ".partial"),
            record_path.with_name(record_path.name + ".partial"),
        )
    )
    if both:
        return _read_json(record_path)
    if not debris:
        return None
    _discard_pair(weights, record_path)
    return None


def write_full(directory: Path, trainer, result: dict) -> None:
    directory = Path(directory)
    sequences = [str(item) for item in np.asarray(result["sequences"]).tolist()]
    _commit_pair(
        _weights_path(directory, FULL_NAME),
        _record_path(directory, FULL_NAME),
        module_state(trainer),
        {
            "e_s": int(result["e_s"]),
            "epochs_run": int(result["epochs_run"]),
            "restore_best": bool(result["restore_best"]),
            "replaced_with_earlier_best": bool(result["replaced_with_earlier_best"]),
            "split": str(result["split"]),
            "score": _vector(result["score"]),
            "sequences": sequences,
            "sequence_sha256": sequence_sha256(sequences),
        },
    )
    remember_muformer_trained(directory)
    note_stage(directory, "full_model", e_s=int(result["e_s"]))


def read_full(directory: Path) -> dict | None:
    directory = Path(directory)
    weights = _weights_path(directory, FULL_NAME)
    record_path = _record_path(directory, FULL_NAME)
    record = _read_committed(weights, record_path)
    if record is None:
        return None
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
    _commit_pair(
        _weights_path(directory, name),
        _record_path(directory, name),
        module_state(trainer),
        {
            "fold": int(fold),
            "epochs_run": int(produced["epochs_run"]),
            "restore_best": bool(produced["restore_best"]),
            "replaced_with_earlier_best": bool(produced["replaced_with_earlier_best"]),
            "held_out_prediction": _vector(produced["held_out_prediction"]),
            "held_out_sequence": sequences,
            "sequence_sha256": sequence_sha256(sequences),
        },
    )
    remember_muformer_trained(directory)


def read_fold(directory: Path, fold: int) -> dict | None:
    directory = Path(directory)
    name = f"fold_{int(fold)}"
    weights = _weights_path(directory, name)
    record_path = _record_path(directory, name)
    record = _read_committed(weights, record_path)
    if record is None:
        return None
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


def _report_document_complete(directory: Path, report: dict) -> bool:
    if report.get("scale") != DESIGNATED_SCALE or int(report.get("outer_seed", -1)) != DESIGNATED_SEED:
        return False
    if report.get("formal_experiment") or report.get("ten_seed_conclusion"):
        return False
    by_arm = report.get("by_arm")
    if not isinstance(by_arm, dict):
        return False
    for arm in ("full", "sum", "no_potts"):
        payload = by_arm.get(arm)
        if not isinstance(payload, dict):
            return False
        scored = payload.get("metrics")
        if not isinstance(scored, dict):
            return False
        for name in metrics.METRIC_NAMES:
            item = scored.get(name)
            if not isinstance(item, dict):
                return False
            if "mean_delta" not in item or "ci_low" not in item or "ci_high" not in item:
                return False
    history = recorded_training(directory)
    if bool(report.get("training_started")) != history["training_started"]:
        return False
    if bool(report.get("muformer_train_called")) != history["muformer_train_called"]:
        return False
    text = (directory / "single_seed_report.md").read_text()
    if str(DESIGNATED_SEED) not in text or DESIGNATED_SCALE not in text:
        return False
    return True


def report_ready(directory: Path) -> bool:
    """True only when the report matches this experiment and the saved artifacts."""
    directory = Path(directory)
    json_path = directory / "single_seed_report.json"
    markdown_path = directory / "single_seed_report.md"
    if not json_path.is_file() or not markdown_path.is_file() or markdown_path.stat().st_size == 0:
        return False
    if not marker_path(directory).is_file():
        return False
    try:
        report = _read_json(json_path)
        _require_identity(read_marker(directory))
    except (OSError, ValueError, RuntimeError, KeyError, TypeError):
        return False
    if not _report_document_complete(directory, report):
        return False
    if not (directory / "oof.json").is_file():
        return False
    weights = _weights_path(directory, FULL_NAME)
    record_path = _record_path(directory, FULL_NAME)
    if not weights.is_file() or not record_path.is_file():
        return False
    try:
        return arms_ready(directory)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError):
        return False


def write_report(directory: Path, report: dict) -> dict:
    directory = Path(directory)
    ready = _json_ready(report)
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
    json_path = directory / "single_seed_report.json"
    markdown_path = directory / "single_seed_report.md"
    json_partial = json_path.with_name(json_path.name + ".partial")
    markdown_partial = markdown_path.with_name(markdown_path.name + ".partial")
    _write_json(json_partial, ready)
    markdown_partial.write_text("\n".join(lines))
    json_partial.replace(json_path)
    markdown_partial.replace(markdown_path)
    note_stage(directory, "report")
    return ready


def read_report(directory: Path) -> dict:
    path = Path(directory) / "single_seed_report.json"
    if not path.is_file():
        raise RuntimeError("the designated report is not saved")
    return _read_json(path)

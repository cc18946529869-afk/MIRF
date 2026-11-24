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
DESIGNATED_OOF_ROWS = 1000
REQUIRED_ARMS = ("full", "sum", "no_potts")
ARM_RESULT_FIELDS = ("eta", "best_epoch", "metrics", "point", "n_bootstrap", "spearman_passes")
_HELD_LOCKS: dict[str, object] = {}
FULL_NAME = "full"
FOLD_COUNT = 5
PMLM_HIDDEN = 1280
PMLM_COORDINATE_MAP = "blat_ref_pos_is_token_after_bos_v1"
_CHECKPOINT_CACHE: dict = {}


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


def _finite_vector(values) -> np.ndarray:
    """One little-endian float64 vector. Non-finite values are refused."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("a saved score must be one vector")
    if not bool(np.isfinite(array).all()):
        raise ValueError("a saved score contains a non-finite value")
    return np.ascontiguousarray(array, dtype="<f8")


def vector_sha256(values) -> str:
    """SHA256 of the vector length and its canonical float64 bytes."""
    array = _finite_vector(values)
    digest = hashlib.sha256()
    digest.update(np.asarray([array.shape[0]], dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _vector(values) -> list[float]:
    array = _finite_vector(values)
    stored = [float(item) for item in array.tolist()]
    restored = np.asarray(json.loads(json.dumps(stored)), dtype=np.float64)
    if restored.shape != array.shape or not np.array_equal(restored, array):
        raise ValueError("a saved score does not round-trip as float64")
    return stored


def _checked_vector(values) -> tuple[list[float], str]:
    stored = _vector(values)
    return stored, vector_sha256(stored)


def _require_saved_vector(values, recorded, label: str) -> np.ndarray:
    if not isinstance(recorded, str) or vector_sha256(values) != recorded:
        raise RuntimeError(f"the saved {label} does not match its sha256")
    return _finite_vector(values)


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


def _strict_positive_int(value, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise RuntimeError(f"the saved full model {label} is not a positive integer")
    return value


def _require_saved_full_record(record: dict) -> None:
    """The saved full model restored the best epoch inside the completed run.

    This is the same relation ``outer_train._require_full_record`` requires
    while training: ``restore_best`` is true, and replacing an earlier epoch
    happens exactly when ``E_s`` is before the last completed epoch.
    """
    e_s = _strict_positive_int(record.get("e_s"), "E_s")
    epochs_run = _strict_positive_int(record.get("epochs_run"), "epochs_run")
    if e_s > epochs_run:
        raise RuntimeError("the saved full model E_s is outside the completed epochs")
    if record.get("restore_best") is not True:
        raise RuntimeError("the saved full model did not restore its best weights")
    if record.get("replaced_with_earlier_best") is not (e_s < epochs_run):
        raise RuntimeError(
            "whether earlier weights were restored does not match E_s and the completed epochs"
        )


def write_full(directory: Path, trainer, result: dict) -> None:
    directory = Path(directory)
    _require_saved_full_record({
        "e_s": result["e_s"],
        "epochs_run": result["epochs_run"],
        "restore_best": result["restore_best"],
        "replaced_with_earlier_best": result["replaced_with_earlier_best"],
    })
    sequences = [str(item) for item in np.asarray(result["sequences"]).tolist()]
    score, score_sha256 = _checked_vector(result["score"])
    if len(score) != len(sequences):
        raise ValueError("the validation score does not match the validation sequences")
    payload = {
        "e_s": result["e_s"],
        "epochs_run": result["epochs_run"],
        "restore_best": result["restore_best"],
        "replaced_with_earlier_best": result["replaced_with_earlier_best"],
        "split": str(result["split"]),
        "score": score,
        "score_sha256": score_sha256,
        "sequences": sequences,
        "sequence_sha256": sequence_sha256(sequences),
    }
    _commit_pair(
        _weights_path(directory, FULL_NAME),
        _record_path(directory, FULL_NAME),
        module_state(trainer),
        payload,
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
    score = _require_saved_vector(record["score"], record.get("score_sha256"), "validation score")
    if int(score.shape[0]) != len(sequences):
        raise RuntimeError("the saved validation score does not match the validation sequences")
    _require_saved_full_record(record)
    return {
        "e_s": int(record["e_s"]),
        "epochs_run": int(record["epochs_run"]),
        "restore_best": bool(record["restore_best"]),
        "replaced_with_earlier_best": bool(record["replaced_with_earlier_best"]),
        "split": "validation",
        "score": score,
        "sequences": np.asarray(sequences),
    }


def write_fold(directory: Path, trainer, fold: int, produced: dict) -> None:
    directory = Path(directory)
    sequences = [str(item) for item in produced["held_out_sequence"]]
    name = f"fold_{int(fold)}"
    prediction, prediction_sha256 = _checked_vector(produced["held_out_prediction"])
    _commit_pair(
        _weights_path(directory, name),
        _record_path(directory, name),
        module_state(trainer),
        {
            "fold": int(fold),
            "epochs_run": int(produced["epochs_run"]),
            "restore_best": bool(produced["restore_best"]),
            "replaced_with_earlier_best": bool(produced["replaced_with_earlier_best"]),
            "held_out_prediction": prediction,
            "held_out_prediction_sha256": prediction_sha256,
            "held_out_sequence": sequences,
            "sequence_sha256": sequence_sha256(sequences),
        },
    )
    remember_muformer_trained(directory)


def _saved_full_epoch(directory: Path) -> int | None:
    path = _record_path(directory, FULL_NAME)
    if not path.is_file():
        return None
    recorded = _read_json(path).get("e_s")
    if isinstance(recorded, bool) or not isinstance(recorded, int):
        raise RuntimeError("the saved full model does not record E_s")
    return int(recorded)


def read_fold(directory: Path, fold: int) -> dict | None:
    directory = Path(directory)
    name = f"fold_{int(fold)}"
    weights = _weights_path(directory, name)
    record_path = _record_path(directory, name)
    record = _read_committed(weights, record_path)
    if record is None:
        return None
    if int(record.get("fold", -1)) != int(fold):
        raise RuntimeError(f"fold {int(fold)} record is not this fold")
    if record.get("restore_best") is not False or record.get("replaced_with_earlier_best") is not False:
        raise RuntimeError(f"fold {int(fold)} did not keep the weights from epoch E_s")
    epochs = record.get("epochs_run")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or int(epochs) < 1:
        raise RuntimeError(f"fold {int(fold)} does not record its training epoch")
    full_epoch = _saved_full_epoch(directory)
    if full_epoch is not None and int(epochs) != full_epoch:
        raise RuntimeError(f"fold {int(fold)} was not trained for E_s")
    sequences = [str(item) for item in record["held_out_sequence"]]
    if sequence_sha256(sequences) != record["sequence_sha256"]:
        raise RuntimeError(f"fold {int(fold)} sequences do not match their sha256")
    if file_sha256(weights) != record["weight_sha256"]:
        raise RuntimeError(f"fold {int(fold)} weights do not match their sha256")
    prediction = _require_saved_vector(
        record["held_out_prediction"],
        record.get("held_out_prediction_sha256"),
        f"fold {int(fold)} prediction",
    )
    return {
        "epochs_run": int(record["epochs_run"]),
        "restore_best": bool(record["restore_best"]),
        "replaced_with_earlier_best": bool(record["replaced_with_earlier_best"]),
        "held_out_prediction": prediction,
        "held_out_sequence": sequences,
    }


def write_oof(directory: Path, predictions, sequences, e_s: int, *, schedule_only: bool, weights_saved: bool) -> None:
    directory = Path(directory)
    names = [str(item) for item in sequences]
    values, values_sha256 = _checked_vector(predictions)
    if len(values) != len(names):
        raise ValueError("OOF predictions do not match the training sequences")
    _write_json(directory / "oof.json", {
        "rows": len(names),
        "e_s": int(e_s),
        "predictions": values,
        "predictions_sha256": values_sha256,
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
    values = _require_saved_vector(record["predictions"], record.get("predictions_sha256"), "OOF predictions")
    if values.shape != (len(names),):
        raise RuntimeError("the saved OOF predictions are not this training sequence list")
    return record


def _require_folds_match_oof(directory: Path, sequences, e_s: int) -> None:
    """The five fold predictions are the frozen partition and the saved OOF."""
    import oof

    names = [str(item) for item in sequences]
    if len(names) != DESIGNATED_OOF_ROWS:
        raise RuntimeError("the training sequences are not the designated 1000 rows")
    loaded = oof.load_scale(Path(__file__).resolve().parent / "precheck", DESIGNATED_SCALE)
    if int(loaded["n"]) != DESIGNATED_OOF_ROWS or len(loaded["folds"]) != FOLD_COUNT:
        raise RuntimeError("the frozen OOF partition is not five folds of 1000 rows")
    assembled = np.full(DESIGNATED_OOF_ROWS, np.nan, dtype=np.float64)
    occupied = np.zeros(DESIGNATED_OOF_ROWS, dtype=bool)
    for spec in loaded["folds"]:
        fold = int(spec["fold"])
        saved = read_fold(directory, fold)
        if saved is None:
            raise RuntimeError(f"fold {fold} is not saved")
        if int(saved["epochs_run"]) != int(e_s):
            raise RuntimeError(f"fold {fold} was not trained for E_s")
        val_idx = np.asarray(spec["val_idx"], dtype=np.int64)
        if val_idx.ndim != 1 or int(val_idx.size) == 0:
            raise RuntimeError(f"fold {fold} has no frozen validation index")
        if bool(np.any(val_idx < 0)) or bool(np.any(val_idx >= DESIGNATED_OOF_ROWS)):
            raise RuntimeError(f"fold {fold} validation index is outside the training rows")
        if len(set(int(item) for item in val_idx.tolist())) != int(val_idx.size):
            raise RuntimeError(f"fold {fold} validation index repeats a row")
        prediction = np.asarray(saved["held_out_prediction"], dtype=np.float64)
        if prediction.shape != val_idx.shape:
            raise RuntimeError(f"fold {fold} prediction does not match the frozen validation index")
        expected = [names[int(item)] for item in val_idx.tolist()]
        if list(saved["held_out_sequence"]) != expected:
            raise RuntimeError(f"fold {fold} sequences are not the frozen validation rows")
        if bool(occupied[val_idx].any()):
            raise RuntimeError("the five folds overlap")
        occupied[val_idx] = True
        assembled[val_idx] = prediction
    if not bool(occupied.all()):
        raise RuntimeError("the five folds do not cover every training row")
    oof_record = read_oof(directory, names)
    saved_oof = _require_saved_vector(
        oof_record["predictions"],
        oof_record.get("predictions_sha256"),
        "OOF predictions",
    )
    if not np.array_equal(assembled, saved_oof):
        raise RuntimeError("the saved OOF predictions are not the assembled fold predictions")


def _checkpoint_path() -> Path:
    return Path(__file__).resolve().parents[3] / "Mu-Protein" / "pretrained" / "pmlm_650m.pt"


def sha256_cached(path: Path, slot: dict) -> str:
    """SHA256 of one file. The saved digest is reused only for the same inode.

    Size and mtime are not enough: a replacement can keep both. Device, inode,
    and ctime have to match as well.
    """
    resolved = Path(path).resolve()
    stat = resolved.stat()
    key = (
        str(resolved),
        int(stat.st_dev),
        int(stat.st_ino),
        int(stat.st_size),
        int(stat.st_mtime_ns),
        int(stat.st_ctime_ns),
    )
    saved = slot.get("entry")
    if isinstance(saved, tuple) and len(saved) == 2 and saved[0] == key:
        return saved[1]
    digest = file_sha256(resolved)
    slot["entry"] = (key, digest)
    return digest


_PMLM_STAT_FIELDS = ("device", "inode", "size", "mtime_ns", "ctime_ns")


def pmlm_file_stat(path: Path) -> dict:
    """Device, inode, size, mtime, and ctime. This does not hash the file."""
    stat = Path(path).resolve().stat()
    return {
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "ctime_ns": int(stat.st_ctime_ns),
    }


def _sealed_pmlm(record: dict) -> dict | None:
    saved = record.get("pmlm_checkpoint")
    if saved is None:
        return None
    if not isinstance(saved, dict):
        raise RuntimeError("the sealed PMLM identity is not a record")
    digest = saved.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise RuntimeError("the sealed PMLM identity has no sha256")
    for key in _PMLM_STAT_FIELDS:
        value = saved.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise RuntimeError("the sealed PMLM identity is incomplete")
    return saved


def _muformer_files_present(directory: Path) -> bool:
    folder = Path(directory) / "muformer"
    if not folder.is_dir():
        return False
    return any(path.is_file() for path in folder.iterdir())


def _require_same_pmlm(saved: dict, checkpoint: Path) -> None:
    """The file must still be the sealed one. A different file is not hashed into the marker."""
    before = pmlm_file_stat(checkpoint)
    for key in _PMLM_STAT_FIELDS:
        if saved[key] != before[key]:
            raise RuntimeError("the PMLM checkpoint does not match the identity sealed for this experiment")
    digest = sha256_cached(checkpoint, _CHECKPOINT_CACHE)
    after = pmlm_file_stat(checkpoint)
    if before != after or digest != saved["sha256"]:
        raise RuntimeError("the PMLM checkpoint does not match the identity sealed for this experiment")


def seal_pmlm_checkpoint(directory: Path, checkpoint: Path) -> dict:
    """Write the checkpoint identity once. Later calls only compare it.

    A mismatch raises and leaves the saved identity in place. Weights that
    already exist cannot receive a newly computed identity.
    """
    path = marker_path(directory)
    record = _read_json(path)
    saved = _sealed_pmlm(record)
    if saved is None:
        if _muformer_files_present(directory):
            raise RuntimeError("μFormer weights exist without a sealed PMLM identity")
        before = pmlm_file_stat(checkpoint)
        digest = sha256_cached(checkpoint, _CHECKPOINT_CACHE)
        after = pmlm_file_stat(checkpoint)
        if before != after:
            raise RuntimeError("the PMLM checkpoint changed while its identity was being read")
        sealed = dict(before)
        sealed["sha256"] = digest
        record["pmlm_checkpoint"] = sealed
        _write_json(path, record)
        return sealed
    _require_same_pmlm(saved, checkpoint)
    return saved


def pmlm_split_record(sequences, checkpoint_sha256: str) -> dict:
    """Identity fields written for one embedded split. Vectors are not included."""
    if not isinstance(checkpoint_sha256, str) or len(checkpoint_sha256) != 64:
        raise RuntimeError("the sealed PMLM sha256 is missing")
    names = [str(item) for item in sequences]
    return {
        "checkpoint_sha256": checkpoint_sha256,
        "sequence_sha256": sequence_sha256(names),
        "rows": len(names),
        "hidden": PMLM_HIDDEN,
        "coordinate_map_version": PMLM_COORDINATE_MAP,
        "embeddings_stored": False,
        "lm_head_called": False,
        "fc_reduce_called": False,
    }


def _designated_split_sequences() -> tuple[list[str], list[str]]:
    import feature_align
    import potts_features

    valid_tsv = potts_features.MAIN_SPLITS / "BLAT_ECOLX_Stiffler_2015_valid.tsv"
    test_tsv = potts_features.MAIN_SPLITS / "BLAT_ECOLX_Stiffler_2015_test.tsv"
    valid_sequences, _valid_scores = feature_align.read_scored_tsv(valid_tsv)
    test_sequences, _test_scores = feature_align.read_scored_tsv(test_tsv)
    return valid_sequences, test_sequences


def write_checked_pmlm_identity(directory: Path, train_sequences) -> None:
    """Write the three split identities for the sequences this run embeds."""
    sealed = seal_pmlm_checkpoint(directory, _checkpoint_path())
    digest = sealed["sha256"]
    valid_sequences, test_sequences = _designated_split_sequences()
    write_embedding_identity(
        directory,
        pmlm_split_record(train_sequences, digest),
        pmlm_split_record(valid_sequences, digest),
        pmlm_split_record(test_sequences, digest),
    )


def _require_pmlm_identity(directory: Path, train_sequences) -> None:
    path = Path(directory) / "pmlm_identity.json"
    if not path.is_file():
        raise RuntimeError("the PMLM identity record is missing")
    saved = _read_json(path)
    if saved.get("embeddings_stored") is not False:
        raise RuntimeError("the PMLM identity record stores embeddings")
    sealed = seal_pmlm_checkpoint(directory, _checkpoint_path())
    digest = sealed["sha256"]
    valid_sequences, test_sequences = _designated_split_sequences()
    expected = {
        "train": pmlm_split_record(train_sequences, digest),
        "valid": pmlm_split_record(valid_sequences, digest),
        "test": pmlm_split_record(test_sequences, digest),
    }
    for split, wanted in expected.items():
        if not _json_equal(saved.get(split), wanted):
            raise RuntimeError(f"the PMLM {split} identity does not match this experiment")


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
    """True only when the saved arms are exactly full, sum, and no_potts."""
    path = Path(directory) / "arms.json"
    if not path.is_file():
        return False
    try:
        record = _read_json(path)
    except (OSError, ValueError, RuntimeError):
        return False
    listed = record.get("arms")
    if not isinstance(listed, list):
        return False
    names = [str(item) for item in listed]
    if len(names) != len(REQUIRED_ARMS) or set(names) != set(REQUIRED_ARMS):
        return False
    by_arm = record.get("by_arm")
    if not isinstance(by_arm, dict) or set(by_arm) != set(REQUIRED_ARMS):
        return False
    for arm in REQUIRED_ARMS:
        payload = by_arm.get(arm)
        weights = Path(directory) / "arms" / f"{arm}.pt"
        if not isinstance(payload, dict) or not weights.is_file():
            return False
        recorded = payload.get("weight_sha256")
        if not isinstance(recorded, str) or file_sha256(weights) != recorded:
            return False
    return True


def _json_equal(left, right) -> bool:
    return json.dumps(_json_ready(left), sort_keys=True) == json.dumps(_json_ready(right), sort_keys=True)


def report_matches_arms(report: dict, arms_record: dict) -> bool:
    """The report's three arm results are the saved arm record, field by field."""
    report_arms = report.get("by_arm")
    saved = arms_record.get("by_arm")
    if not isinstance(report_arms, dict) or set(report_arms) != set(REQUIRED_ARMS):
        return False
    if not isinstance(saved, dict):
        return False
    for arm in REQUIRED_ARMS:
        left = report_arms.get(arm)
        right = saved.get(arm)
        if not isinstance(left, dict) or not isinstance(right, dict):
            return False
        for field in ARM_RESULT_FIELDS:
            if field not in left or field not in right or not _json_equal(left[field], right[field]):
                return False
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
    return True


def _canonical_report(report: dict) -> dict:
    return json.loads(json.dumps(_json_ready(report), ensure_ascii=False, sort_keys=True))


def _report_markdown(ready: dict) -> str:
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
    return "\n".join(lines)


def report_files_present(directory: Path) -> bool:
    directory = Path(directory)
    return (directory / "single_seed_report.json").is_file() and (directory / "single_seed_report.md").is_file()


def report_ready(directory: Path, sequences) -> bool:
    """True only when the report, weights, OOF, and the three arms agree."""
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
        if markdown_path.read_text() != _report_markdown(report):
            return False
        if not _report_document_complete(directory, report):
            return False
        names = [str(item) for item in sequences]
        if len(names) != DESIGNATED_OOF_ROWS:
            return False
        full = read_full(directory)
        if full is None:
            return False
        oof_record = read_oof(directory, names)
        if int(oof_record["rows"]) != DESIGNATED_OOF_ROWS:
            return False
        if bool(oof_record.get("schedule_only")) or not bool(oof_record.get("weights_saved")):
            return False
        if int(report["e_s"]) != int(full["e_s"]) or int(oof_record["e_s"]) != int(full["e_s"]):
            return False
        if int(report.get("oof_rows", -1)) != DESIGNATED_OOF_ROWS or int(report["oof_rows"]) != int(oof_record["rows"]):
            return False
        _require_folds_match_oof(directory, names, int(full["e_s"]))
        _require_pmlm_identity(directory, names)
        if not arms_ready(directory):
            return False
        return report_matches_arms(report, read_arms(directory))
    except (OSError, ValueError, RuntimeError, KeyError, TypeError):
        return False


def require_consistent_report(directory: Path, sequences) -> None:
    """Refuse to treat this directory as finished when any artifact disagrees."""
    if not report_ready(directory, sequences):
        raise RuntimeError("the saved report does not match the experiment artifacts")


def write_report(directory: Path, report: dict) -> dict:
    directory = Path(directory)
    ready = _canonical_report(report)
    text = _report_markdown(ready)
    json_path = directory / "single_seed_report.json"
    markdown_path = directory / "single_seed_report.md"
    json_partial = json_path.with_name(json_path.name + ".partial")
    markdown_partial = markdown_path.with_name(markdown_path.name + ".partial")
    _write_json(json_partial, ready)
    markdown_partial.write_text(text)
    json_partial.replace(json_path)
    markdown_partial.replace(markdown_path)
    note_stage(directory, "report")
    return ready


def read_report(directory: Path) -> dict:
    path = Path(directory) / "single_seed_report.json"
    if not path.is_file():
        raise RuntimeError("the designated report is not saved")
    return _read_json(path)

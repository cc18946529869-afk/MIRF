"""Read-only inventory of the local Potts bootstrap inputs.

The worktree does not contain the untracked bootstrap directory. This script
reads the files where they already are and writes text into the worktree. It
does not refit, resample, or copy ``.npy`` arrays into the commit list.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np

import identity_audit
import bootstrap_fit
import potts_features

LOCKED_R_J = 0.4581354189599478
LOCKED_ACTION = "parameter_unstable"
REFERENCE_GAUGE = ("fields.npy", "directional.npy", "h_hat.npy", "j_star.npy")
EXPECTED_ARRAYS = {
    "h_hat.npy": ((263, 21), np.dtype("<f8")),
    "j_star.npy": ((263, 263, 21, 21), np.dtype("<f8")),
    "q.npy": ((263, 21), np.dtype("<f8")),
}
MAIN_BOOTSTRAP = Path("/home/cc233/data/cc233_data/work_dir/MIRF/research/mirf-muformer/dcr/bootstrap")
COMMIT_FILES = (
    "Mu-Protein/mu-former/src/trainer.py",
    "research/mirf-muformer/dcr/lobes.py",
    "research/mirf-muformer/dcr/geometry.py",
    "research/mirf-muformer/dcr/frozen_pmlm.py",
    "research/mirf-muformer/dcr/residual.py",
    "research/mirf-muformer/dcr/predict.py",
    "research/mirf-muformer/dcr/oof.py",
    "research/mirf-muformer/dcr/bootstrap_inventory.py",
    "research/mirf-muformer/dcr/test_best_weights.py",
    "research/mirf-muformer/dcr/test_residual_protocol.py",
    "research/mirf-muformer/dcr/bootstrap_input_inventory.tsv",
    "research/mirf-muformer/dcr/bootstrap_verification_report.json",
    "research/mirf-muformer/dcr/commit_file_list.txt",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def npy_meta(path: Path) -> tuple[tuple[int, ...], str]:
    """Read shape and dtype from a ``.npy`` header without loading the array."""
    with Path(path).open("rb") as handle:
        magic = handle.read(6)
        if magic != b"\x93NUMPY":
            raise ValueError(f"{path} is not an npy file")
        major, _minor = struct.unpack("BB", handle.read(2))
        if major == 1:
            (header_len,) = struct.unpack("<H", handle.read(2))
        else:
            (header_len,) = struct.unpack("<I", handle.read(4))
        header = ast.literal_eval(handle.read(header_len).decode("latin1").strip())
    dtype = np.dtype(header["descr"])
    shape = tuple(int(item) for item in header["shape"])
    return shape, dtype.str


def describe(path: Path) -> dict[str, object]:
    record = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
        "shape": "",
        "dtype": "",
    }
    if path.suffix == ".npy":
        shape, dtype = npy_meta(path)
        record["shape"] = " ".join(str(item) for item in shape)
        record["dtype"] = dtype
    return record


def resolve_bootstrap(explicit: str | None) -> tuple[Path, str]:
    if explicit:
        path = Path(explicit)
        if not path.is_dir():
            raise FileNotFoundError(path)
        return path, "explicit"
    local = Path(__file__).resolve().parent / "bootstrap"
    if local.is_dir():
        return local, "worktree"
    if MAIN_BOOTSTRAP.is_dir():
        return MAIN_BOOTSTRAP, "main_checkout_readonly"
    raise FileNotFoundError("the local bootstrap directory is not in the worktree or the main checkout")


def _require_array(record: dict[str, object], name: str) -> None:
    if name not in EXPECTED_ARRAYS:
        return
    shape, dtype = EXPECTED_ARRAYS[name]
    if record["shape"] != " ".join(str(item) for item in shape) or record["dtype"] != dtype.str:
        raise ValueError(f"{record['path']} has shape {record['shape']} dtype {record['dtype']}")


def inventory(bootstrap: Path) -> tuple[list[dict[str, object]], dict]:
    """Hash the reference arrays, the formal lock, and replicates 1–100."""
    failures: list[str] = []
    rows: list[dict[str, object]] = []
    reference = bootstrap / "reference"
    formal = bootstrap / "formal"
    manifest = json.loads((reference / "manifest.json").read_text())
    if manifest.get("refit") is not False:
        failures.append("reference manifest refit is not false")
    if manifest.get("accepted_point_estimate") is not True or manifest.get("n_passed") != 263:
        failures.append("reference manifest is not the accepted 263-site point estimate")
    for name in ("manifest.json", *REFERENCE_GAUGE):
        record = describe(reference / name)
        rows.append(record)
        _require_array(record, name)
        if name in manifest.get("gauge_sha256", {}):
            if record["sha256"] != manifest["gauge_sha256"][name]:
                failures.append(f"reference {name} does not match manifest gauge_sha256")
        if name in potts_features.GAUGE_SHA256 and record["sha256"] != potts_features.GAUGE_SHA256[name]:
            failures.append(f"reference {name} does not match the locked gauge constant")
    stage_path = formal / "stage_100.json"
    stage_text = stage_path.read_text()
    if "0.4581354189599478" not in stage_text:
        failures.append("stage_100.json no longer contains the locked R_J digits")
    stage = json.loads(stage_text)
    if stage.get("r_j") != LOCKED_R_J or stage.get("action") != LOCKED_ACTION or stage.get("passed") is not False:
        failures.append("stage_100.json does not still say parameter_unstable with the locked R_J")
    seed = json.loads((formal / "seed.json").read_text())
    rows.append(describe(formal / "seed.json"))
    rows.append(describe(stage_path))
    present = []
    for replicate_id in range(1, 101):
        directory = formal / f"replicate_{replicate_id:03d}"
        present.append(directory.is_dir())
        if not directory.is_dir():
            failures.append(f"replicate {replicate_id} is missing")
            continue
        try:
            identity_audit._require_replicate(directory, replicate_id, seed)
        except ValueError as exc:
            failures.append(str(exc))
        manifest_path = directory / "manifest.json"
        replicate_manifest = json.loads(manifest_path.read_text())
        names = ["manifest.json"] + sorted(entry.name for entry in os.scandir(directory) if entry.name.endswith(".npy"))
        for required in ("h_hat.npy", "j_star.npy", "q.npy"):
            if required not in names:
                failures.append(f"replicate {replicate_id} is missing {required}")
        site_npz = sum(1 for entry in os.scandir(directory) if entry.name.startswith("site_") and entry.name.endswith(".npz"))
        if site_npz != 263:
            failures.append(f"replicate {replicate_id} does not have 263 site npz files")
        for name in names:
            record = describe(directory / name)
            rows.append(record)
            _require_array(record, name)
            if name == "indices.npy":
                payload = bootstrap_fit.indices_sha256(np.load(directory / name, mmap_mode="r"))
                if record["shape"] != str(replicate_manifest.get("draw_count")):
                    failures.append(f"replicate {replicate_id} indices length is not draw_count")
                if payload != replicate_manifest.get("indices_sha256"):
                    failures.append(f"replicate {replicate_id} indices payload hash does not match its manifest")
        if replicate_id % 10 == 0:
            print(f"inventory replicate {replicate_id}", flush=True)
    if not all(present) or len(present) != 100:
        failures.append("replicates 1 through 100 are not all present")
    report = {
        "read_only": True,
        "bootstrap_dir": str(bootstrap),
        "npy_copied_into_worktree": False,
        "refit": False,
        "bootstrap_rerun": False,
        "formal_training_run": False,
        "formal_metrics_computed": False,
        "phylogenetic_correction_completed": False,
        "r_j": stage.get("r_j"),
        "r_j_digits": "0.4581354189599478",
        "r_j_unchanged": stage.get("r_j") == LOCKED_R_J and "0.4581354189599478" in stage_text,
        "action": stage.get("action"),
        "action_unchanged": stage.get("action") == LOCKED_ACTION,
        "passed": stage.get("passed"),
        "parameter_unstable_retained": stage.get("action") == LOCKED_ACTION and stage.get("passed") is False,
        "replicates_present": int(sum(present)),
        "site_npz_hashed": False,
        "site_npz_per_replicate": 263,
        "inventory_rows": len(rows),
        "failures": failures,
        "passed_check": not failures,
    }
    return rows, report


def commit_list() -> list[str]:
    selected = []
    for relative in COMMIT_FILES:
        if relative.endswith(".npy") or relative.endswith(".npz"):
            raise RuntimeError("an array file was placed on the commit list")
        selected.append(relative)
    return selected


def write_outputs(rows: list[dict[str, object]], report: dict, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    inventory_path = destination / "bootstrap_input_inventory.tsv"
    with inventory_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "bytes", "sha256", "shape", "dtype"), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    report["commit_list"] = commit_list()
    report["npy_in_commit_list"] = False
    report["summary_zh"] = (
        "本机 Bootstrap 输入已只读重算哈希。"
        f" R_J 仍是 {LOCKED_R_J}，action 仍是 parameter_unstable，passed 仍为假。"
        "这不是系统发育校正已经完成。清单里的 SHA256 是文件哈希；"
        "抽样索引清单里的 indices_sha256 是 int64 内容哈希，两者分开核对。"
        " npy 不在将要提交的清单里，也没有重跑 Bootstrap。"
    )
    (destination / "bootstrap_verification_report.json").write_text(json.dumps(report, indent=2) + "\n")
    (destination / "commit_file_list.txt").write_text("\n".join(report["commit_list"]) + "\n")


def main(argv: list[str] | None = None) -> None:
    argument = argv[0] if argv else None
    bootstrap, source = resolve_bootstrap(argument)
    rows, report = inventory(bootstrap)
    report["resolved_from"] = source
    report["worktree_bootstrap_absent"] = not (Path(__file__).resolve().parent / "bootstrap").is_dir()
    write_outputs(rows, report, Path(__file__).resolve().parent)
    print(json.dumps({key: report[key] for key in ("passed_check", "replicates_present", "r_j", "action", "inventory_rows", "resolved_from")}))
    if not report["passed_check"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main(sys.argv[1:])

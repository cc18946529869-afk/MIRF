"""Build the sealed 14 Potts features for one substitution table.

Means come from the saved full-fit gauge. Standard deviations are the sample
standard deviation across formal replicates 1–100, with divisor 99. Each
replicate uses its own q.npy. This module does not refit Potts or train.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

import identity_audit
import potts_fit
import solver_v2

FEATURE_NAMES = (
    "r_mean",
    "delta_e",
    "std_r_mean",
    "std_delta_e",
    "r_source",
    "std_r_source",
    "r_other",
    "r_contact",
    "r_remote",
    "r_unknown",
    "std_r_other",
    "std_r_contact",
    "std_r_remote",
    "std_r_unknown",
)
MEAN_NAMES = ("r_mean", "delta_e", "r_source", "r_other", "r_contact", "r_remote", "r_unknown")
COMPONENT_NAMES = ("r_mean", "delta_e", "r_source", "r_other", "r_contact", "r_remote", "r_unknown")
SPLIT_SHA256 = {
    "train": "615e47bd09ca1a413e77449dbf286036d227d588cfc12034feff70c7a9ffbe67",
    "valid": "803af89ec564da0f84b857e3c6efbb96a9718e4db45d7ab853294b87f7406fc6",
    "test": "66e8ce3eec1e603e435ff54111bc95804b64104a4f5da90fb1cf45a569f9854d",
}
SPLIT_ROWS = {"train": 3497, "valid": 500, "test": 999}
GAUGE_SHA256 = {
    "h_hat.npy": "59f3bc61ed99a6448c75119af9c54cf7995f094047ead32784d2da2c8de3cbec",
    "j_star.npy": "eaedeba815dfea038a1847d92febb0c7bd6109cefe008e4ad5fba0a7aa5c472d",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_std(replicates: np.ndarray) -> np.ndarray:
    """Sample standard deviation across the first axis. The divisor is count - 1."""
    if replicates.shape[0] < 2:
        raise ValueError("a sample standard deviation needs at least two replicates")
    return np.std(replicates, axis=0, ddof=1)


def describe_substitution(wild: str, sequence: str) -> dict | None:
    """Derive one mature WiM from a full sequence. Anything else is unobserved."""
    if len(sequence) != len(wild):
        return None
    differences = [index for index, (left, right) in enumerate(zip(wild, sequence)) if left != right]
    if len(differences) != 1:
        return None
    position = differences[0]
    wild_aa = wild[position]
    mutant_aa = sequence[position]
    prep = Path(__file__).resolve().parents[1] / "prep"
    if str(prep) not in sys.path:
        sys.path.insert(0, str(prep))
    import msa_prepare

    if wild_aa not in msa_prepare.CODE or mutant_aa not in msa_prepare.CODE:
        return None
    ref_pos = position + 1
    if ref_pos < msa_prepare.MATURE_START or ref_pos > len(wild):
        return None
    return {
        "ref_pos": ref_pos,
        "mature_pos": ref_pos - msa_prepare.MATURE_START + 1,
        "mature_index": ref_pos - msa_prepare.MATURE_START,
        "wt_aa": wild_aa,
        "mut_aa": mutant_aa,
        "mutation": f"{wild_aa}{ref_pos}{mutant_aa}",
        "state": msa_prepare.CODE[mutant_aa],
    }


def pack_features(means: dict[str, float], deviations: dict[str, float]) -> np.ndarray:
    """Place the seven means and seven standard deviations in the sealed order."""
    return np.array(
        [
            means["r_mean"],
            means["delta_e"],
            deviations["r_mean"],
            deviations["delta_e"],
            means["r_source"],
            deviations["r_source"],
            means["r_other"],
            means["r_contact"],
            means["r_remote"],
            means["r_unknown"],
            deviations["r_other"],
            deviations["r_contact"],
            deviations["r_remote"],
            deviations["r_unknown"],
        ],
        dtype=np.float64,
    )


def all_components(
    sequence: np.ndarray,
    fields: np.ndarray,
    couplings: np.ndarray,
    background: np.ndarray,
    masks: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Return seven arrays shaped (sites, states) for one parameter set."""
    identity_audit.require_saved_parameters(fields, couplings, length=int(sequence.shape[0]))
    if background.shape != (sequence.shape[0], identity_audit.STATES):
        raise ValueError("background q does not match the sites and 21 states")
    if not np.isfinite(background).all() or np.any(background < 0):
        raise ValueError("background q must be finite and nonnegative")
    if float(np.max(np.abs(background.sum(axis=1) - 1.0))) > identity_audit.ACCEPTANCE:
        raise ValueError("background q is not normalized within the locked bound")
    stacked = {name: np.empty((sequence.shape[0], identity_audit.STATES), dtype=np.float64) for name in COMPONENT_NAMES}
    index = np.arange(sequence.shape[0])
    old = np.asarray(couplings, dtype=np.float64)[index, :, sequence, :]
    for state in range(identity_audit.STATES):
        parts = identity_audit.component_state(sequence, fields, couplings, background, masks, state, old)
        for name in COMPONENT_NAMES:
            stacked[name][:, state] = parts[name]
    return stacked


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _load_locked_parameters(directory: Path, names: tuple[str, str] = ("h_hat.npy", "j_star.npy")) -> tuple[np.ndarray, np.ndarray]:
    fields = np.load(directory / names[0])
    couplings = np.load(directory / names[1])
    identity_audit.require_saved_parameters(fields, couplings, length=identity_audit.LENGTH)
    return fields, couplings


def write_stiffler_table(root: Path) -> dict:
    """Write the 4996-row feature table and return its audit summary."""
    prep = root / "prep"
    dcr = root / "dcr"
    if str(prep) not in sys.path:
        sys.path.insert(0, str(prep))
    import msa_prepare

    wild = msa_prepare.TARGET_SEQUENCE
    mature = np.load(prep / "mature_codes.npy")
    decoded = "".join(msa_prepare.AMINO_ACIDS[int(code) - 1] for code in mature[0])
    if decoded != wild[msa_prepare.MATURE_START - 1 :]:
        raise ValueError("the mature query does not match the target sequence")
    sequence = mature[0].astype(np.int64)
    masks = identity_audit.geometry(prep / "domain_map.tsv", prep / "1BTL.pdb")
    alignment, weights, total = potts_fit.load_official_alignment(prep)
    if not np.array_equal(alignment[0], sequence):
        raise ValueError("the official alignment query is not the mature query")
    background = solver_v2._background(alignment, weights, total, identity_audit.STATES)
    reference = dcr / "bootstrap" / "reference"
    manifest = json.loads((reference / "manifest.json").read_text())
    for name, digest in GAUGE_SHA256.items():
        if manifest["gauge_sha256"][name] != digest or file_sha256(reference / name) != digest:
            raise ValueError(f"{name} does not match the locked gauge hash")
    fields, couplings = _load_locked_parameters(reference)
    full = all_components(sequence, fields, couplings, background, masks)
    full_audit = identity_audit.audit_parameters(sequence, fields, couplings, background, masks)
    if full_audit["normalized_energy"] > identity_audit.ACCEPTANCE or full_audit["normalized_geometry"] > identity_audit.ACCEPTANCE:
        raise ValueError("the full-fit decomposition exceeds the locked bound")
    replicate_values = {name: [] for name in COMPONENT_NAMES}
    replicate_maxima = []
    lock = json.loads((dcr / "bootstrap" / "formal" / "seed.json").read_text())
    for replicate_id in range(1, 101):
        directory = dcr / "bootstrap" / "formal" / f"replicate_{replicate_id:03d}"
        identity_audit._require_replicate(directory, replicate_id, lock)
        replicate_fields, replicate_couplings = _load_locked_parameters(directory)
        replicate_q = np.load(directory / "q.npy")
        parts = all_components(sequence, replicate_fields, replicate_couplings, replicate_q, masks)
        checked = identity_audit.audit_parameters(sequence, replicate_fields, replicate_couplings, replicate_q, masks)
        if checked["normalized_energy"] > identity_audit.ACCEPTANCE or checked["normalized_geometry"] > identity_audit.ACCEPTANCE:
            raise ValueError(f"replicate {replicate_id} exceeds the locked bound")
        replicate_maxima.append(checked)
        for name in COMPONENT_NAMES:
            replicate_values[name].append(parts[name])
        print(f"features {replicate_id}", flush=True)
    deviations = {name: sample_std(np.stack(replicate_values[name], axis=0)) for name in COMPONENT_NAMES}
    if deviations["r_mean"].shape != (identity_audit.LENGTH, identity_audit.STATES):
        raise ValueError("the bootstrap standard deviation has the wrong shape")
    repo = root.parents[1]
    split_dir = repo / "proteingym_v0.1_split" / "subs"
    records = []
    seen = set()
    for split, count in SPLIT_ROWS.items():
        path = split_dir / f"BLAT_ECOLX_Stiffler_2015_{split}.tsv"
        if file_sha256(path) != SPLIT_SHA256[split]:
            raise ValueError(f"the {split} split hash does not match the reviewed v0.1 file")
        rows = _load_rows(path)
        if len(rows) != count or any(set(row) != {"sequence", "score", "score_bin"} for row in rows):
            raise ValueError(f"the {split} split does not have the reviewed rows and columns")
        for row in rows:
            if row["sequence"] in seen:
                raise ValueError("a sequence occurs in more than one accepted row")
            seen.add(row["sequence"])
            described = describe_substitution(wild, row["sequence"])
            if described is None:
                values = np.zeros(14, dtype=np.float64)
                observed = np.zeros(14, dtype=bool)
                described = {"mutation": "", "ref_pos": "", "mature_pos": "", "wt_aa": "", "mut_aa": ""}
            else:
                site = described["mature_index"]
                state = described["state"]
                values = pack_features(
                    {name: float(full[name][site, state]) for name in MEAN_NAMES},
                    {name: float(deviations[name][site, state]) for name in COMPONENT_NAMES},
                )
                observed = np.ones(14, dtype=bool)
                if not np.isfinite(values).all():
                    raise ValueError(f"{described['mutation']} has a non-finite feature")
            records.append((split, row["sequence"], described, values, observed))
    if len(records) != sum(SPLIT_ROWS.values()):
        raise ValueError("the feature table does not contain the 4996 reviewed sequences")
    destination = dcr / "features"
    destination.mkdir(exist_ok=True)
    table_path = destination / "blat_ecolx_stiffler_2015_v0.1.tsv"
    with table_path.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["split", "sequence", "mutation", "ref_pos", "mature_pos", "wt_aa", "mut_aa", *FEATURE_NAMES, *[f"mask_{index:02d}" for index in range(1, 15)]])
        for split, sequence_text, described, values, observed in records:
            writer.writerow(
                [
                    split,
                    sequence_text,
                    described["mutation"],
                    described["ref_pos"],
                    described["mature_pos"],
                    described["wt_aa"],
                    described["mut_aa"],
                    *[format(float(value), ".17g") for value in values],
                    *[int(flag) for flag in observed],
                ]
            )
    n1000_path = root.parents[2] / "Muformer_DPFunc" / "new-Mu-Protein-main" / "data" / "dms" / "BLAT_ECOLX_Stiffler_2015_n1000" / "BLAT_ECOLX_Stiffler_2015_n1000_train.tsv"
    reloaded = _load_rows(table_path)
    if [row["sequence"] for row in reloaded] != [item[1] for item in records]:
        raise ValueError("the written table does not preserve sequence order")
    max_reload_energy = 0.0
    max_reload_geometry = 0.0
    for item, saved in zip(records, reloaded):
        if not bool(item[4][0]):
            if any(float(saved[name]) != 0.0 for name in FEATURE_NAMES) or any(saved[f"mask_{index:02d}"] != "0" for index in range(1, 15)):
                raise ValueError("a masked row was not stored as zeros")
            continue
        if any(saved[f"mask_{index:02d}"] != "1" for index in range(1, 15)):
            raise ValueError("an observed row does not have all 14 masks")
        values = [float(saved[name]) for name in FEATURE_NAMES]
        if not np.isfinite(values).all():
            raise ValueError("a reloaded feature is not finite")
        delta_e, r_mean, r_source, r_other = (float(saved[name]) for name in ("delta_e", "r_mean", "r_source", "r_other"))
        contact, remote, unknown = (float(saved[name]) for name in ("r_contact", "r_remote", "r_unknown"))
        max_reload_energy = max(max_reload_energy, abs(delta_e - (r_mean + r_source + r_other)) / max(1.0, abs(delta_e) + abs(r_mean) + abs(r_source) + abs(r_other)))
        max_reload_geometry = max(max_reload_geometry, abs(r_other - (contact + remote + unknown)) / max(1.0, abs(r_other) + abs(contact) + abs(remote) + abs(unknown)))
    if max_reload_energy > identity_audit.ACCEPTANCE or max_reload_geometry > identity_audit.ACCEPTANCE:
        raise ValueError("the written table exceeds the locked identity bound")
    n1000 = _load_rows(n1000_path)
    train_sequences = {item[1] for item in records if item[0] == "train"}
    observed_rows = [item for item in records if bool(item[4][0])]
    summary = {
        "rows": len(records),
        "observed_rows": len(observed_rows),
        "masked_rows": len(records) - len(observed_rows),
        "splits": SPLIT_ROWS,
        "split_sha256": SPLIT_SHA256,
        "feature_names": FEATURE_NAMES,
        "score_not_used_as_feature": True,
        "bootstrap_replicates": 100,
        "std_divisor": 99,
        "acceptance": identity_audit.ACCEPTANCE,
        "full_fit": full_audit,
        "bootstrap_max_normalized_energy": max(item["normalized_energy"] for item in replicate_maxima),
        "bootstrap_max_normalized_geometry": max(item["normalized_geometry"] for item in replicate_maxima),
        "reloaded_max_normalized_energy": max_reload_energy,
        "reloaded_max_normalized_geometry": max_reload_geometry,
        "n1000_train_rows_found_in_train_split": sum(row["sequence"] in train_sequences for row in n1000),
        "table_sha256": file_sha256(table_path),
        "j_std_not_used": True,
    }
    (destination / "audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    summary = write_stiffler_table(root)
    print(json.dumps({key: summary[key] for key in ("rows", "observed_rows", "masked_rows", "bootstrap_max_normalized_energy", "bootstrap_max_normalized_geometry", "n1000_train_rows_found_in_train_split")}))


if __name__ == "__main__":
    main()

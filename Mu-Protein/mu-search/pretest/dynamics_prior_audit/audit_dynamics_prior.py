"""WT-structure ANM features and the cefotaxime-label gate.

The gate refuses an ampicillin assay. Feature ranks are properties of one
crystal structure. They are not fitness labels.
"""

import csv
from pathlib import Path

import numpy as np


THREE_LETTER = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}

# Ambler numbers of the catalytic set named for this audit.
CATALYTIC_AMBLER = (70, 130, 132, 166, 234)
SEQUENCE_START = 24


def read_ca_atoms(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("ATOM") or line[12:16].strip() != "CA" or line[21] != "A":
            continue
        rows.append(
            {
                "pdb_number": int(line[22:26]),
                "residue": THREE_LETTER[line[17:20].strip()],
                "coordinate": np.array(
                    [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                    dtype=np.float64,
                ),
            }
        )
    return rows


def map_to_sequence(atoms, wild_type):
    mapped = []
    for index, atom in enumerate(atoms):
        sequence_position = SEQUENCE_START + index
        expected = wild_type[sequence_position - 1]
        mapped.append(
            {
                "ca_index": index,
                "pdb_number": atom["pdb_number"],
                "sequence_position": sequence_position,
                "pdb_residue": atom["residue"],
                "sequence_residue": expected,
                "residue_agrees": atom["residue"] == expected,
                "coordinate": atom["coordinate"],
                "catalytic": int(atom["pdb_number"] in CATALYTIC_AMBLER),
            }
        )
    return mapped


def anm_hessian(coordinates, cutoff):
    count = coordinates.shape[0]
    hessian = np.zeros((3 * count, 3 * count), dtype=np.float64)
    for i in range(count):
        for j in range(i + 1, count):
            difference = coordinates[i] - coordinates[j]
            distance = float(np.linalg.norm(difference))
            if distance == 0 or distance >= cutoff:
                continue
            block = np.outer(difference, difference) / (distance * distance)
            hessian[3 * i:3 * i + 3, 3 * i:3 * i + 3] += block
            hessian[3 * j:3 * j + 3, 3 * j:3 * j + 3] += block
            hessian[3 * i:3 * i + 3, 3 * j:3 * j + 3] -= block
            hessian[3 * j:3 * j + 3, 3 * i:3 * i + 3] -= block
    return hessian


def mode_covariance(hessian, n_modes):
    values, vectors = np.linalg.eigh(hessian)
    order = np.argsort(values)
    values = values[order]
    vectors = vectors[:, order]
    start = 6
    stop = start + n_modes
    if stop > len(values):
        raise ValueError("not enough modes")
    used = values[start:stop]
    if np.any(used <= 1e-8):
        raise ValueError("a selected mode is not a positive vibration")
    covariance = np.zeros_like(hessian)
    for value, vector in zip(used, vectors[:, start:stop].T):
        covariance += np.outer(vector, vector) / value
    return covariance, values


def response_magnitudes(covariance):
    count = covariance.shape[0] // 3
    magnitudes = np.zeros((count, count), dtype=np.float64)
    for source in range(count):
        total = np.zeros(count, dtype=np.float64)
        for axis in range(3):
            force = np.zeros(3 * count, dtype=np.float64)
            force[3 * source + axis] = 1.0
            displacement = covariance @ force
            total += np.linalg.norm(displacement.reshape(count, 3), axis=1)
        magnitudes[:, source] = total / 3.0
    return magnitudes


def flexibility_and_coupling(magnitudes, catalytic_indices):
    if not catalytic_indices:
        raise ValueError("catalytic set is empty")
    total = magnitudes.sum()
    if total <= 0:
        raise ValueError("response magnitudes are empty")
    flexibility = magnitudes.sum(axis=1) / total
    functional = magnitudes[:, list(catalytic_indices)].mean(axis=1)
    background = magnitudes.mean(axis=1)
    coupling = functional / background
    return flexibility, coupling


def rank_values(values):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        stop = start
        while stop + 1 < len(order) and values[order[stop + 1]] == values[order[start]]:
            stop += 1
        average = (start + stop) / 2.0 + 1.0
        ranks[order[start:stop + 1]] = average
        start = stop + 1
    return ranks


def spearman(left, right):
    left_rank = rank_values(np.asarray(left, dtype=np.float64))
    right_rank = rank_values(np.asarray(right, dtype=np.float64))
    left_center = left_rank - left_rank.mean()
    right_center = right_rank - right_rank.mean()
    scale = np.sqrt(np.sum(left_center ** 2) * np.sum(right_center ** 2))
    if scale == 0:
        return None
    return float(np.sum(left_center * right_center) / scale)


def require_cefotaxime_assay(assay):
    text = assay.lower()
    if "ampicillin" in text or "cefotaxime" not in text:
        raise ValueError("this table is not a cefotaxime single-mutant assay")
    return assay


def contact_counts(coordinates, radius):
    count = coordinates.shape[0]
    counts = np.zeros(count, dtype=np.int64)
    for i in range(count):
        for j in range(i + 1, count):
            if np.linalg.norm(coordinates[i] - coordinates[j]) <= radius:
                counts[i] += 1
                counts[j] += 1
    return counts


def write_mapping(mapped, path):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "ca_index", "pdb_number", "sequence_position", "pdb_residue",
                "sequence_residue", "residue_agrees", "catalytic",
            ],
        )
        writer.writeheader()
        for row in mapped:
            writer.writerow({key: row[key] for key in writer.fieldnames})


def write_features(mapped, flexibility, coupling, contacts, path):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sequence_position", "pdb_number", "pdb_residue", "sequence_residue",
                "residue_agrees", "catalytic", "dfi_like", "dci_like", "contact_count_8A",
            ],
        )
        writer.writeheader()
        for row, flex, couple, contact in zip(mapped, flexibility, coupling, contacts):
            writer.writerow(
                {
                    "sequence_position": row["sequence_position"],
                    "pdb_number": row["pdb_number"],
                    "pdb_residue": row["pdb_residue"],
                    "sequence_residue": row["sequence_residue"],
                    "residue_agrees": int(row["residue_agrees"]),
                    "catalytic": row["catalytic"],
                    "dfi_like": "{0:.8g}".format(float(flex)),
                    "dci_like": "{0:.8g}".format(float(couple)),
                    "contact_count_8A": int(contact),
                }
            )


def feature_matrix(coordinates, catalytic_indices, cutoff, n_modes):
    hessian = anm_hessian(coordinates, cutoff)
    covariance, _values = mode_covariance(hessian, n_modes)
    magnitudes = response_magnitudes(covariance)
    return flexibility_and_coupling(magnitudes, catalytic_indices)


def main():
    root = Path(__file__).resolve().parents[4]
    pdb_path = root / "research/mirf-muformer/prep/1BTL.pdb"
    wild_type = (
        "MSIQHFRVALIPFFAAFCLPVFAHPETLVKVKDAEDQLGARVGYIELDLNSGKILESFRPEERFPMMSTF"
        "KVLLCGAVLSRVDAGQEQLGRRIHYSQNDLVEYSPVTEKHLTDGMTVRELCSAAITMSDNTAANLLLTTIG"
        "GPKELTAFLHNMGDHVTRLDRWEPELNEAIPNDERDTTMPAAMATTLRKLLTGELLTLASRQQLIDWMEAD"
        "KVAGPLLRSALPAGWFIADKSGAGERGSRGIIAALGPDGKPSRIVVIYTTGSQATMDERNRQIAEIGASLIKHW"
    )
    atoms = read_ca_atoms(pdb_path)
    mapped = map_to_sequence(atoms, wild_type)
    if len(mapped) != 263 or mapped[-1]["sequence_position"] != 286:
        raise RuntimeError("1BTL did not cover sequence positions 24 through 286")
    coordinates = np.vstack([row["coordinate"] for row in mapped])
    catalytic = [row["ca_index"] for row in mapped if row["catalytic"]]
    if len(catalytic) != len(CATALYTIC_AMBLER):
        raise RuntimeError("catalytic residues were not all present")
    out = Path(__file__).resolve().parent
    write_mapping(mapped, out / "residue_mapping.csv")
    flexibility, coupling = feature_matrix(coordinates, catalytic, 13.0, 20)
    contacts = contact_counts(coordinates, 8.0)
    write_features(mapped, flexibility, coupling, contacts, out / "dci_dfi_features.csv")
    rows = []
    reference_coupling = coupling
    for cutoff in (10.0, 13.0, 15.0):
        for n_modes in (10, 20):
            _flex, other = feature_matrix(coordinates, catalytic, cutoff, n_modes)
            rows.append(
                {
                    "cutoff_A": cutoff,
                    "n_modes": n_modes,
                    "spearman_with_cutoff13_modes20": "{0:.6f}".format(
                        spearman(reference_coupling, other)
                    ),
                }
            )
    with (out / "feature_stability.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["cutoff_A", "n_modes", "spearman_with_cutoff13_modes20"])
        writer.writeheader()
        writer.writerows(rows)
    disagreements = [row["sequence_position"] for row in mapped if not row["residue_agrees"]]
    print("sites", len(mapped), "disagreements", disagreements)
    print("catalytic_sequence_positions", [mapped[index]["sequence_position"] for index in catalytic])


if __name__ == "__main__":
    main()

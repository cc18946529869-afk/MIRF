"""Directional ANM features for 1BTL and 1ZG4.

R[i, j] is the response at residue i when residue j is perturbed.
This script does not read fitness labels.
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
CATALYTIC_AMBLER = (70, 130, 132, 166, 234)
SEQUENCE_START = 24
WILD_TYPE = (
    "MSIQHFRVALIPFFAAFCLPVFAHPETLVKVKDAEDQLGARVGYIELDLNSGKILESFRPEERFPMMSTF"
    "KVLLCGAVLSRVDAGQEQLGRRIHYSQNDLVEYSPVTEKHLTDGMTVRELCSAAITMSDNTAANLLLTTIG"
    "GPKELTAFLHNMGDHVTRLDRWEPELNEAIPNDERDTTMPAAMATTLRKLLTGELLTLASRQQLIDWMEAD"
    "KVAGPLLRSALPAGWFIADKSGAGERGSRGIIAALGPDGKPSRIVVIYTTGSQATMDERNRQIAEIGASLIKHW"
)


def read_chain(path):
    seqres = []
    atoms = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("SEQRES") and len(line) > 11 and line[11] == "A":
            seqres.extend(THREE_LETTER[token] for token in line[19:].split())
        if line.startswith("ATOM") and line[12:16].strip() == "CA" and line[21] == "A":
            atoms.append(
                {
                    "pdb_number": int(line[22:26]),
                    "residue": THREE_LETTER[line[17:20].strip()],
                    "coordinate": np.array(
                        [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                        dtype=np.float64,
                    ),
                }
            )
    return "".join(seqres), atoms


def map_atoms(atoms, wild_type):
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
                "catalytic": int(atom["pdb_number"] in CATALYTIC_AMBLER),
                "coordinate": atom["coordinate"],
            }
        )
    return mapped


def anm_hessian(coordinates, cutoff):
    count = coordinates.shape[0]
    hessian = np.zeros((3 * count, 3 * count), dtype=np.float64)
    for left in range(count):
        for right in range(left + 1, count):
            difference = coordinates[left] - coordinates[right]
            distance = float(np.linalg.norm(difference))
            if distance == 0 or distance >= cutoff:
                continue
            block = np.outer(difference, difference) / (distance * distance)
            hessian[3 * left:3 * left + 3, 3 * left:3 * left + 3] += block
            hessian[3 * right:3 * right + 3, 3 * right:3 * right + 3] += block
            hessian[3 * left:3 * left + 3, 3 * right:3 * right + 3] -= block
            hessian[3 * right:3 * right + 3, 3 * left:3 * left + 3] -= block
    return hessian


def vibration_basis(hessian):
    values, vectors = np.linalg.eigh(hessian)
    order = np.argsort(values)
    values = values[order]
    vectors = vectors[:, order]
    if np.max(np.abs(values[:6])) > 1e-6 or values[6] <= 1e-6:
        raise ValueError("the six rigid modes are not separated from the rest")
    return values, vectors


def covariance_from_modes(values, vectors, n_modes):
    available = len(values) - 6
    if n_modes is None:
        n_modes = available
    if n_modes < 1 or n_modes > available:
        raise ValueError("mode count is outside the non-rigid spectrum")
    covariance = np.zeros((values.shape[0], values.shape[0]), dtype=np.float64)
    for index in range(6, 6 + n_modes):
        vector = vectors[:, index]
        covariance += np.outer(vector, vector) / values[index]
    return covariance, n_modes


def response_matrix(covariance):
    """R[i, j] is the mean coordinate response at i when j is perturbed."""
    count = covariance.shape[0] // 3
    responses = np.zeros((count, count), dtype=np.float64)
    for source in range(count):
        total = np.zeros(count, dtype=np.float64)
        for axis in range(3):
            force = np.zeros(3 * count, dtype=np.float64)
            force[3 * source + axis] = 1.0
            displacement = covariance @ force
            total += np.linalg.norm(displacement.reshape(count, 3), axis=1)
        responses[:, source] = total / 3.0
    return responses


def flexibility_index(responses):
    total = responses.sum()
    if total <= 0:
        raise ValueError("responses are empty")
    return responses.sum(axis=1) / total


def directional_coupling(responses, active_indices):
    active = np.asarray(active_indices, dtype=int)
    if active.size == 0:
        raise ValueError("catalytic set is empty")
    row_background = responses.mean(axis=1)
    column_background = responses.mean(axis=0)
    active_to_residue = responses[:, active].mean(axis=1) / row_background
    residue_to_active = responses[active, :].mean(axis=0) / column_background
    return active_to_residue, residue_to_active, active_to_residue - residue_to_active


def percentile_ranks(values):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        stop = start
        while stop + 1 < len(order) and values[order[stop + 1]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop + 1]] = (start + stop) / 2.0 + 1.0
        start = stop + 1
    return 100.0 * (ranks - 0.5) / len(values)


def spearman(left, right):
    left_rank = percentile_ranks(np.asarray(left, dtype=np.float64))
    right_rank = percentile_ranks(np.asarray(right, dtype=np.float64))
    left_center = left_rank - left_rank.mean()
    right_center = right_rank - right_rank.mean()
    scale = np.sqrt(np.sum(left_center ** 2) * np.sum(right_center ** 2))
    if scale == 0:
        return None
    return float(np.sum(left_center * right_center) / scale)


def contact_counts(coordinates, radius):
    count = coordinates.shape[0]
    counts = np.zeros(count, dtype=np.int64)
    for left in range(count):
        for right in range(left + 1, count):
            if np.linalg.norm(coordinates[left] - coordinates[right]) <= radius:
                counts[left] += 1
                counts[right] += 1
    return counts


def kabsch_rmsd(left, right):
    left_center = left - left.mean(axis=0)
    right_center = right - right.mean(axis=0)
    rotation, _, vh = np.linalg.svd(left_center.T @ right_center)
    if np.linalg.det(vh.T @ rotation.T) < 0:
        vh[-1] *= -1
    aligned = right_center @ (vh.T @ rotation.T)
    return float(np.sqrt(np.mean(np.sum((left_center - aligned) ** 2, axis=1))))


def write_mapping(mapped, seqres, path):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "ca_index", "pdb_number", "sequence_position", "pdb_residue",
                "sequence_residue", "seqres_residue", "residue_agrees", "catalytic",
            ],
        )
        writer.writeheader()
        for row in mapped:
            position = row["sequence_position"]
            writer.writerow(
                {
                    "ca_index": row["ca_index"],
                    "pdb_number": row["pdb_number"],
                    "sequence_position": position,
                    "pdb_residue": row["pdb_residue"],
                    "sequence_residue": row["sequence_residue"],
                    "seqres_residue": seqres[position - 1] if seqres else "",
                    "residue_agrees": int(row["residue_agrees"]),
                    "catalytic": row["catalytic"],
                }
            )


def feature_rows(structure, mapped, responses, flexibility, directions, contacts, cutoff, n_modes, label):
    active_to, residue_to, asymmetry = directions
    flex_rank = percentile_ranks(flexibility)
    rows = []
    for index, row in enumerate(mapped):
        rows.append(
            {
                "structure": structure,
                "cutoff_A": cutoff,
                "mode_set": label,
                "n_modes": n_modes,
                "sequence_position": row["sequence_position"],
                "pdb_number": row["pdb_number"],
                "pdb_residue": row["pdb_residue"],
                "sequence_residue": row["sequence_residue"],
                "residue_agrees": int(row["residue_agrees"]),
                "catalytic": row["catalytic"],
                "dfi": float(flexibility[index]),
                "dfi_percentile": float(flex_rank[index]),
                "dci_active_to_residue": float(active_to[index]),
                "dci_residue_to_active": float(residue_to[index]),
                "dci_asymmetry": float(asymmetry[index]),
                "contact_count_8A": int(contacts[index]),
            }
        )
    return rows


def compute_structure(structure, path, cutoffs, mode_requests):
    seqres, atoms = read_chain(path)
    mapped = map_atoms(atoms, WILD_TYPE)
    coordinates = np.vstack([row["coordinate"] for row in mapped])
    active = [row["ca_index"] for row in mapped if row["catalytic"]]
    contacts = contact_counts(coordinates, 8.0)
    stored = {}
    rows = []
    for cutoff in cutoffs:
        hessian = anm_hessian(coordinates, cutoff)
        values, vectors = vibration_basis(hessian)
        for request in mode_requests:
            covariance, used = covariance_from_modes(values, vectors, request)
            responses = response_matrix(covariance)
            flexibility = flexibility_index(responses)
            directions = directional_coupling(responses, active)
            label = "all_nonrigid" if request is None else "lowest_{0}".format(used)
            rows.extend(
                feature_rows(
                    structure, mapped, responses, flexibility, directions,
                    contacts, cutoff, used, label,
                )
            )
            stored[(cutoff, label)] = {
                "active_to": directions[0],
                "residue_to": directions[1],
                "dfi": flexibility,
            }
    return {
        "seqres": seqres,
        "mapped": mapped,
        "coordinates": coordinates,
        "rows": rows,
        "stored": stored,
        "nonrigid": 3 * len(mapped) - 6,
    }


def write_csv(path, rows, fieldnames):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            output = {}
            for key in fieldnames:
                value = row[key]
                output[key] = "{0:.8g}".format(value) if isinstance(value, float) else value
            writer.writerow(output)


def main():
    out = Path(__file__).resolve().parent
    root = Path(__file__).resolve().parents[4]
    structures = {
        "1BTL": root / "research/mirf-muformer/prep/1BTL.pdb",
        "1ZG4": out / "structures/1ZG4.pdb",
    }
    cutoffs = (10.0, 13.0, 15.0)
    mode_requests = (10, 20, 50, None)
    computed = {}
    all_rows = []
    for name, path in structures.items():
        computed[name] = compute_structure(name, path, cutoffs, mode_requests)
        all_rows.extend(computed[name]["rows"])
        if name == "1ZG4":
            write_mapping(computed[name]["mapped"], computed[name]["seqres"], out / "residue_mapping_1zg4.csv")
    reference = [
        row for row in all_rows
        if row["structure"] == "1ZG4" and row["cutoff_A"] == 13.0 and row["mode_set"] == "all_nonrigid"
    ]
    write_csv(
        out / "anm_prs_features.csv",
        reference,
        [
            "structure", "cutoff_A", "mode_set", "n_modes", "sequence_position", "pdb_number",
            "pdb_residue", "sequence_residue", "residue_agrees", "catalytic",
            "dfi", "dfi_percentile", "contact_count_8A",
        ],
    )
    write_csv(
        out / "dci_directional_features.csv",
        reference,
        [
            "structure", "cutoff_A", "mode_set", "n_modes", "sequence_position", "pdb_number",
            "catalytic", "dci_active_to_residue", "dci_residue_to_active", "dci_asymmetry",
        ],
    )
    sensitivity = []
    for name, result in computed.items():
        stored = result["stored"]
        reference = stored[(13.0, "all_nonrigid")]
        for label in ("lowest_10", "lowest_20", "lowest_50"):
            sensitivity.append(
                {
                    "comparison": "{0} cutoff 13, all_nonrigid vs {1}, active_to_residue".format(name, label),
                    "value": spearman(reference["active_to"], stored[(13.0, label)]["active_to"]),
                }
            )
            sensitivity.append(
                {
                    "comparison": "{0} cutoff 13, all_nonrigid vs {1}, dfi".format(name, label),
                    "value": spearman(reference["dfi"], stored[(13.0, label)]["dfi"]),
                }
            )
        for cutoff in (10.0, 15.0):
            sensitivity.append(
                {
                    "comparison": "{0} all_nonrigid, cutoff 13 vs {1}, active_to_residue".format(name, cutoff),
                    "value": spearman(reference["active_to"], stored[(cutoff, "all_nonrigid")]["active_to"]),
                }
            )
        sensitivity.append(
            {
                "comparison": "{0} cutoff 13 all_nonrigid, active_to_residue vs residue_to_active".format(name),
                "value": spearman(reference["active_to"], reference["residue_to"]),
            }
        )
    for label in ("all_nonrigid", "lowest_20"):
        sensitivity.append(
            {
                "comparison": "1ZG4 vs 1BTL, cutoff 13 {0}, active_to_residue".format(label),
                "value": spearman(
                    computed["1ZG4"]["stored"][(13.0, label)]["active_to"],
                    computed["1BTL"]["stored"][(13.0, label)]["active_to"],
                ),
            }
        )
    sensitivity.append(
        {
            "comparison": "1ZG4 vs 1BTL C-alpha RMSD after superposition, angstrom",
            "value": kabsch_rmsd(
                computed["1ZG4"]["coordinates"],
                computed["1BTL"]["coordinates"],
            ),
        }
    )
    write_csv(out / "parameter_sensitivity.csv", sensitivity, ["comparison", "value"])
    print("nonrigid", computed["1ZG4"]["nonrigid"])
    print("seqres_equals_search", computed["1ZG4"]["seqres"] == WILD_TYPE)
    print("rmsd_row", sensitivity[-1])
    print("direction_row", [row for row in sensitivity if "residue_to_active" in row["comparison"]])


if __name__ == "__main__":
    main()

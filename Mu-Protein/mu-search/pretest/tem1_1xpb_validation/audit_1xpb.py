"""Map 1XPB, 1ZG4, and 1BTL by sequence alignment and recompute ANM features.

R[i, j] is the mean coordinate response at residue i when residue j is perturbed.
DCI follows the displayed ratio in Modi and Ozkan, Int. J. Mol. Sci. 2018, equation 6:
the average response at i over functional perturbations, divided by the average
response at i over all perturbations. The reverse index uses that same per-residue
normalization at the functional residues. Sulfate and water are not ANM nodes.
"""

import csv
import json
from pathlib import Path

import numpy as np


THREE_LETTER = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}
WILD_TYPE = (
    "MSIQHFRVALIPFFAAFCLPVFAHPETLVKVKDAEDQLGARVGYIELDLNSGKILESFRPEERFPMMSTF"
    "KVLLCGAVLSRVDAGQEQLGRRIHYSQNDLVEYSPVTEKHLTDGMTVRELCSAAITMSDNTAANLLLTTIG"
    "GPKELTAFLHNMGDHVTRLDRWEPELNEAIPNDERDTTMPAAMATTLRKLLTGELLTLASRQQLIDWMEAD"
    "KVAGPLLRSALPAGWFIADKSGAGERGSRGIIAALGPDGKPSRIVVIYTTGSQATMDERNRQIAEIGASLIKHW"
)
FUNCTIONAL_SEARCH_POSITIONS = (68, 128, 130, 164, 232)
FUNCTIONAL_RESIDUES = "SSNEK"


def align_sequences(query, target):
    """Global alignment. Positions are 1-based. None marks a gap."""
    match_score, mismatch_score, gap_score = 2, -1, -2
    rows = len(query) + 1
    cols = len(target) + 1
    scores = np.zeros((rows, cols), dtype=np.int32)
    for i in range(1, rows):
        scores[i, 0] = i * gap_score
    for j in range(1, cols):
        scores[0, j] = j * gap_score
    for i in range(1, rows):
        for j in range(1, cols):
            diagonal = scores[i - 1, j - 1] + (
                match_score if query[i - 1] == target[j - 1] else mismatch_score
            )
            scores[i, j] = max(diagonal, scores[i - 1, j] + gap_score, scores[i, j - 1] + gap_score)
    pairs = []
    i, j = len(query), len(target)
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            step = match_score if query[i - 1] == target[j - 1] else mismatch_score
            if scores[i, j] == scores[i - 1, j - 1] + step:
                pairs.append((i, j))
                i -= 1
                j -= 1
                continue
        if i > 0 and scores[i, j] == scores[i - 1, j] + gap_score:
            pairs.append((i, None))
            i -= 1
            continue
        pairs.append((None, j))
        j -= 1
    pairs.reverse()
    return pairs


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
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if len(left) != len(right) or len(left) < 3:
        raise ValueError("spearman needs two equal columns")
    left_rank = percentile_ranks(left)
    right_rank = percentile_ranks(right)
    left_center = left_rank - left_rank.mean()
    right_center = right_rank - right_rank.mean()
    scale = np.sqrt(np.sum(left_center ** 2) * np.sum(right_center ** 2))
    if scale == 0:
        return None
    return float(np.sum(left_center * right_center) / scale)


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


def dci_functional_to_residue(responses, functional):
    """Equation 6: perturb functional residues, read the response at each residue."""
    functional = np.asarray(functional, dtype=int)
    numerator = responses[:, functional].sum(axis=1) / functional.size
    denominator = responses.sum(axis=1) / responses.shape[1]
    return numerator / denominator


def dci_residue_to_functional(responses, functional):
    """Average, over functional residues k, of the response at k when i is perturbed.

    Each term is divided by residue k's own average response. This is the reverse
    of equation 6. It is not the ratio of two column means.
    """
    functional = np.asarray(functional, dtype=int)
    baseline = responses.mean(axis=1)
    ratios = responses[functional, :] / baseline[functional, None]
    return ratios.mean(axis=0)


def column_mean_ratio(responses, functional):
    """The previous reverse index. Kept only so the correction can be measured."""
    functional = np.asarray(functional, dtype=int)
    return responses[functional, :].mean(axis=0) / responses.mean(axis=0)


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
    if np.linalg.det(rotation @ vh) < 0:
        vh = vh.copy()
        vh[-1] *= -1
    aligned = right_center @ (rotation @ vh).T
    return float(np.sqrt(np.mean(np.sum((left_center - aligned) ** 2, axis=1))))


def read_pdb_ca(path):
    atoms = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("ATOM") or line[12:16].strip() != "CA" or line[21] != "A":
            continue
        alt = line[16].strip()
        if alt not in ("", "A"):
            continue
        atoms.append(
            {
                "residue": THREE_LETTER[line[17:20].strip()],
                "label_seq_id": len(atoms) + 1,
                "auth_seq_id": int(line[22:26]),
                "ins_code": line[26].strip(),
                "coordinate": np.array(
                    [float(line[30:38]), float(line[38:46]), float(line[46:54])],
                    dtype=np.float64,
                ),
            }
        )
    return atoms


def cif_value(token):
    if token in (".", "?"):
        return ""
    return token


def read_mmcif(path):
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    entity = []
    capture = False
    for line in lines:
        if line.startswith("_entity_poly.pdbx_seq_one_letter_code"):
            capture = True
            continue
        if capture:
            text = line.strip()
            if text == ";":
                if entity:
                    break
                continue
            if text.startswith(";"):
                text = text[1:]
            entity.append(text)
    atoms = []
    sulfate = []
    in_atoms = False
    for line in lines:
        if line.startswith("_atom_site.group_PDB"):
            in_atoms = True
            continue
        if in_atoms and line.startswith("#"):
            break
        if not in_atoms or not (line.startswith("ATOM") or line.startswith("HETATM")):
            continue
        fields = line.split()
        if fields[6] != "A" and fields[5] != "SO4":
            continue
        if fields[3] == "CA" and fields[6] == "A" and cif_value(fields[4]) in ("", "A"):
            atoms.append(
                {
                    "residue": THREE_LETTER[fields[5]],
                    "label_seq_id": int(fields[8]),
                    "auth_seq_id": int(fields[16]),
                    "ins_code": "" if fields[9] in (".", "?") else fields[9],
                    "coordinate": np.array([float(fields[10]), float(fields[11]), float(fields[12])], dtype=np.float64),
                }
            )
        if fields[5] == "SO4" and fields[3] == "S":
            sulfate.append(
                {
                    "auth_seq_id": int(fields[16]),
                    "auth_asym_id": fields[18],
                    "coordinate": np.array([float(fields[10]), float(fields[11]), float(fields[12])], dtype=np.float64),
                }
            )
    return "".join(entity), atoms, sulfate


def map_chain(atoms, wild_type):
    structure = "".join(atom["residue"] for atom in atoms)
    pairs = align_sequences(wild_type, structure)
    by_structure = {}
    for query_pos, target_pos in pairs:
        if target_pos is not None and query_pos is not None:
            by_structure[target_pos - 1] = query_pos
    mapped = []
    for index, atom in enumerate(atoms):
        search_position = by_structure.get(index)
        expected = wild_type[search_position - 1] if search_position else ""
        mapped.append(
            {
                "structure_index": index,
                "label_seq_id": atom["label_seq_id"],
                "auth_seq_id": atom["auth_seq_id"],
                "ins_code": atom["ins_code"],
                "pdb_residue": atom["residue"],
                "search_position": search_position if search_position else "",
                "search_residue": expected,
                "residue_agrees": int(bool(search_position) and atom["residue"] == expected),
                "coordinate": atom["coordinate"],
            }
        )
    return structure, pairs, mapped


def functional_indices(mapped):
    wanted = {
        position: residue
        for position, residue in zip(FUNCTIONAL_SEARCH_POSITIONS, FUNCTIONAL_RESIDUES)
    }
    found = {}
    for row in mapped:
        if row["search_position"] in wanted:
            found[row["search_position"]] = row
    missing = [position for position in wanted if position not in found]
    if missing:
        raise ValueError("functional search positions are not aligned: {0}".format(missing))
    for position, residue in wanted.items():
        row = found[position]
        if row["search_residue"] != residue or row["pdb_residue"] != residue:
            raise ValueError("functional residue changed at search position {0}".format(position))
    return [found[position]["structure_index"] for position in FUNCTIONAL_SEARCH_POSITIONS]


def compute_profiles(coordinates, functional, cutoffs, mode_requests):
    stored = {}
    for cutoff in cutoffs:
        values, vectors = vibration_basis(anm_hessian(coordinates, cutoff))
        for request in mode_requests:
            covariance, used = covariance_from_modes(values, vectors, request)
            responses = response_matrix(covariance)
            label = "all_nonrigid" if request is None else "lowest_{0}".format(used)
            stored[(cutoff, label)] = {
                "n_modes": used,
                "dfi": flexibility_index(responses),
                "dci_functional_to_i": dci_functional_to_residue(responses, functional),
                "dci_i_to_functional": dci_residue_to_functional(responses, functional),
                "previous_column_ratio": column_mean_ratio(responses, functional),
            }
    return stored, 3 * len(coordinates) - 6


def write_csv(path, rows, fieldnames):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: "{0:.8g}".format(row[key]) if isinstance(row[key], float) else row[key]
                    for key in fieldnames
                }
            )


def feature_table(structure, mapped, profile, contacts, cutoff):
    dfi_rank = percentile_ranks(profile["dfi"])
    forward_rank = percentile_ranks(profile["dci_functional_to_i"])
    rows = []
    for index, row in enumerate(mapped):
        if row["search_position"] == "":
            continue
        forward = profile["dci_functional_to_i"][index]
        reverse = profile["dci_i_to_functional"][index]
        rows.append(
            {
                "structure": structure,
                "cutoff_A": cutoff,
                "mode_set": "all_nonrigid",
                "n_modes": profile["n_modes"],
                "search_position": row["search_position"],
                "search_residue": row["search_residue"],
                "label_seq_id": row["label_seq_id"],
                "auth_seq_id": row["auth_seq_id"],
                "pdb_residue": row["pdb_residue"],
                "residue_agrees": row["residue_agrees"],
                "functional": int(row["search_position"] in FUNCTIONAL_SEARCH_POSITIONS),
                "ca_x": float(row["coordinate"][0]),
                "ca_y": float(row["coordinate"][1]),
                "ca_z": float(row["coordinate"][2]),
                "dfi": float(profile["dfi"][index]),
                "dfi_percentile": float(dfi_rank[index]),
                "dci_functional_to_i": float(forward),
                "dci_functional_to_i_percentile": float(forward_rank[index]),
                "dci_i_to_functional": float(reverse),
                "dci_asymmetry": float(forward - reverse),
                "contact_count_8A": int(contacts[index]),
            }
        )
    return rows


def values_by_position(rows, key):
    return {row["search_position"]: row[key] for row in rows}


def main():
    if len(WILD_TYPE) != 286:
        raise ValueError("search sequence length changed")
    observed = "".join(WILD_TYPE[position - 1] for position in FUNCTIONAL_SEARCH_POSITIONS)
    if observed != FUNCTIONAL_RESIDUES:
        raise ValueError("functional residues do not match the search sequence")
    out = Path(__file__).resolve().parent
    root = Path(__file__).resolve().parents[4]
    sources = {
        "1XPB": out / "structures/1XPB.cif",
        "1ZG4": root / "Mu-Protein/mu-search/pretest/tem1_ctx_structure_validation/structures/1ZG4.pdb",
        "1BTL": root / "research/mirf-muformer/prep/1BTL.pdb",
    }
    entity, atoms, sulfate = read_mmcif(sources["1XPB"])
    chains = {
        "1XPB": atoms,
        "1ZG4": read_pdb_ca(sources["1ZG4"]),
        "1BTL": read_pdb_ca(sources["1BTL"]),
    }
    cutoffs = (10.0, 13.0, 15.0)
    mode_requests = (10, 20, 50, None)
    computed = {}
    tables = {}
    for name, chain in chains.items():
        structure_sequence, pairs, mapped = map_chain(chain, WILD_TYPE)
        functional = functional_indices(mapped)
        coordinates = np.vstack([row["coordinate"] for row in mapped])
        profiles, nonrigid = compute_profiles(coordinates, functional, cutoffs, mode_requests)
        contacts = contact_counts(coordinates, 8.0)
        reference = profiles[(13.0, "all_nonrigid")]
        tables[name] = feature_table(name, mapped, reference, contacts, 13.0)
        computed[name] = {
            "sequence": structure_sequence,
            "pairs": pairs,
            "mapped": mapped,
            "profiles": profiles,
            "nonrigid": nonrigid,
            "functional": functional,
        }
    reference_rows = tables["1XPB"]
    mapping_rows = []
    for row in computed["1XPB"]["mapped"]:
        mapping_rows.append(
            {
                "search_position": row["search_position"],
                "search_residue": row["search_residue"],
                "label_seq_id": row["label_seq_id"],
                "auth_seq_id": row["auth_seq_id"],
                "ins_code": row["ins_code"],
                "pdb_residue": row["pdb_residue"],
                "residue_agrees": row["residue_agrees"],
                "functional": int(row["search_position"] in FUNCTIONAL_SEARCH_POSITIONS),
                "ca_x": float(row["coordinate"][0]),
                "ca_y": float(row["coordinate"][1]),
                "ca_z": float(row["coordinate"][2]),
            }
        )
    write_csv(
        out / "residue_mapping_1xpb.csv",
        mapping_rows,
        [
            "search_position", "search_residue", "label_seq_id", "auth_seq_id", "ins_code",
            "pdb_residue", "residue_agrees", "functional", "ca_x", "ca_y", "ca_z",
        ],
    )
    write_csv(
        out / "anm_prs_features.csv",
        reference_rows,
        [
            "structure", "cutoff_A", "mode_set", "n_modes", "search_position", "auth_seq_id",
            "pdb_residue", "residue_agrees", "functional", "dfi", "dfi_percentile", "contact_count_8A",
        ],
    )
    write_csv(
        out / "dci_directional_features.csv",
        reference_rows,
        [
            "structure", "cutoff_A", "mode_set", "n_modes", "search_position", "functional",
            "dci_functional_to_i", "dci_functional_to_i_percentile", "dci_i_to_functional", "dci_asymmetry",
        ],
    )
    sensitivity = []
    profile = computed["1XPB"]["profiles"]
    base = profile[(13.0, "all_nonrigid")]
    for label in ("lowest_10", "lowest_20", "lowest_50"):
        other = profile[(13.0, label)]
        sensitivity.append({"comparison": "1XPB cutoff 13, all_nonrigid vs {0}, dci_functional_to_i".format(label), "value": spearman(base["dci_functional_to_i"], other["dci_functional_to_i"])})
        sensitivity.append({"comparison": "1XPB cutoff 13, all_nonrigid vs {0}, dfi".format(label), "value": spearman(base["dfi"], other["dfi"])})
    for cutoff in (10.0, 15.0):
        sensitivity.append({"comparison": "1XPB all_nonrigid, cutoff 13 vs {0}, dci_functional_to_i".format(cutoff), "value": spearman(base["dci_functional_to_i"], profile[(cutoff, "all_nonrigid")]["dci_functional_to_i"])})
    sensitivity.append({"comparison": "1XPB cutoff 13 all_nonrigid, functional_to_i vs i_to_functional", "value": spearman(base["dci_functional_to_i"], base["dci_i_to_functional"])})
    sensitivity.append({"comparison": "1XPB cutoff 13 all_nonrigid, corrected reverse vs previous column-mean ratio", "value": spearman(base["dci_i_to_functional"], base["previous_column_ratio"])})
    sensitivity.append({"comparison": "1XPB cutoff 13 all_nonrigid, dfi vs contact_count_8A", "value": spearman(base["dfi"], [row["contact_count_8A"] for row in reference_rows])})
    sensitivity.append({"comparison": "1XPB cutoff 13 all_nonrigid, dci_functional_to_i vs contact_count_8A", "value": spearman(base["dci_functional_to_i"], [row["contact_count_8A"] for row in reference_rows])})
    shared = set(values_by_position(tables["1XPB"], "dfi"))
    for name in ("1ZG4", "1BTL"):
        shared &= set(values_by_position(tables[name], "dfi"))
    shared = sorted(shared)
    for name in ("1ZG4", "1BTL"):
        for key in ("dci_functional_to_i", "dfi"):
            sensitivity.append(
                {
                    "comparison": "1XPB vs {0}, shared search positions, {1}".format(name, key),
                    "value": spearman(
                        [values_by_position(tables["1XPB"], key)[position] for position in shared],
                        [values_by_position(tables[name], key)[position] for position in shared],
                    ),
                }
            )
    coordinates = {}
    for name in ("1XPB", "1ZG4", "1BTL"):
        coordinates[name] = np.vstack(
            [
                next(row["coordinate"] for row in computed[name]["mapped"] if row["search_position"] == position)
                for position in shared
            ]
        )
    for left, right in (("1XPB", "1ZG4"), ("1XPB", "1BTL"), ("1ZG4", "1BTL")):
        sensitivity.append(
            {
                "comparison": "{0} vs {1} C-alpha RMSD on shared search positions, angstrom".format(left, right),
                "value": kabsch_rmsd(coordinates[left], coordinates[right]),
            }
        )
    write_csv(out / "parameter_sensitivity.csv", sensitivity, ["comparison", "value"])
    mismatches = []
    for name, result in computed.items():
        for row in result["mapped"]:
            if row["search_position"] != "" and not row["residue_agrees"]:
                mismatches.append(
                    {
                        "structure": name,
                        "search_position": row["search_position"],
                        "search_residue": row["search_residue"],
                        "auth_seq_id": row["auth_seq_id"],
                        "pdb_residue": row["pdb_residue"],
                    }
                )
    unmapped_query = []
    for query_pos, target_pos in computed["1XPB"]["pairs"]:
        if target_pos is None:
            unmapped_query.append(query_pos)
    sulfate_rows = []
    for item in sulfate:
        distances = []
        for row in computed["1XPB"]["mapped"]:
            if row["search_position"] in FUNCTIONAL_SEARCH_POSITIONS:
                distances.append(
                    {
                        "search_position": row["search_position"],
                        "auth_seq_id": row["auth_seq_id"],
                        "distance_A": float(np.linalg.norm(item["coordinate"] - row["coordinate"])),
                    }
                )
        sulfate_rows.append({"auth_seq_id": item["auth_seq_id"], "auth_asym_id": item["auth_asym_id"], "distances": distances})
    summary = {
        "entity_length": len(entity),
        "entity_equals_ca_sequence": entity == computed["1XPB"]["sequence"],
        "entity_equals_mature_slice_check": None,
        "ca_count": len(computed["1XPB"]["mapped"]),
        "mapped_count": len(reference_rows),
        "identity_counts": {
            name: int(sum(row["residue_agrees"] for row in result["mapped"] if row["search_position"] != ""))
            for name, result in computed.items()
        },
        "mapped_counts": {
            name: int(sum(row["search_position"] != "" for row in result["mapped"]))
            for name, result in computed.items()
        },
        "mismatches": mismatches,
        "unmapped_search_positions": unmapped_query,
        "nonrigid_modes": {name: result["nonrigid"] for name, result in computed.items()},
        "shared_positions": len(shared),
        "sulfate": sulfate_rows,
        "max_abs_asymmetry": max(abs(row["dci_asymmetry"]) for row in reference_rows),
        "median_abs_asymmetry": float(np.median([abs(row["dci_asymmetry"]) for row in reference_rows])),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("ca_count", "identity_counts", "mismatches", "unmapped_search_positions", "shared_positions")}, indent=2))


if __name__ == "__main__":
    main()

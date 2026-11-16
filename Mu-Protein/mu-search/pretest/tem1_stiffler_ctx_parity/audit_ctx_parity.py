"""Compare the local ANM with the public DFI-DCI formulas and summarize CTX calls.

The reference formulas are the functions in
https://github.com/SBOZKAN/DFI-DCI at c89c0b3d401d5640b94395a754fe087d9aaf1eef.
This script does not replace the features already written for 1XPB.
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "tem1_1xpb_validation"))
import audit_1xpb as prior


AMBLER_OMITTED = (239, 253)
SEVEN_DIRECTIONS = np.array(
    [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]],
    dtype=np.float64,
)
MSA_PATH = Path("/data/cc233_data/work_dir/MIRF/research/mirf-muformer/prep/BLAT_ECOLX_full_11-26-2021_b02.a2m")
MSA_SHA256 = "219506a958d1196c84cbcbe983d2ddbb159b519d8facebd38046c9a719229fa1"


def unit_directions(directions):
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    return directions / norms


def official_hessian(coordinates):
    """Distance-weighted ANM Hessian used by dfi_calc.calchessian.

    The repository names a squared distance `r` and sets every spring to
    100**3 / r**3. No cutoff is applied in calc_covariance.
    """
    count = coordinates.shape[0]
    hessian = np.zeros((3 * count, 3 * count), dtype=np.float64)
    gamma = 100.0
    for left in range(count):
        for right in range(count):
            if left == right:
                continue
            difference = coordinates[left] - coordinates[right]
            squared = float(np.dot(difference, difference))
            spring = (gamma ** 3) / (squared ** 3)
            block = spring * np.outer(difference, difference) / squared
            hessian[3 * left:3 * left + 3, 3 * left:3 * left + 3] += block
            hessian[3 * left:3 * left + 3, 3 * right:3 * right + 3] -= block
    return hessian


def official_covariance(hessian):
    left, values, right = np.linalg.svd(hessian, full_matrices=False)
    singular = values < 1e-6
    inverse = np.zeros_like(values)
    inverse[~singular] = 1.0 / values[~singular]
    covariance = (left * inverse) @ right
    return covariance, int(np.sum(singular))


def response_matrix(covariance, directions):
    count = covariance.shape[0] // 3
    total = np.zeros((count, count), dtype=np.float64)
    for direction in directions:
        forces = np.zeros((3 * count, count), dtype=np.float64)
        for source in range(count):
            forces[3 * source:3 * source + 3, source] = direction
        displacement = covariance @ forces
        total += np.linalg.norm(displacement.reshape(count, 3, count), axis=1)
    return total / len(directions)


def official_profiles(responses, functional):
    normalized = responses / responses.sum()
    flexibility = normalized.sum(axis=1)
    numerator = normalized[:, functional].sum(axis=1) / len(functional)
    denominator = normalized.sum(axis=1) / normalized.shape[0]
    forward = numerator / denominator
    baseline = normalized.mean(axis=1)
    reverse = (normalized[functional, :] / baseline[functional, None]).mean(axis=0)
    return flexibility, forward, reverse


def ambler_to_search(number):
    if number < 26 or number > 290 or number in AMBLER_OMITTED:
        raise ValueError("Ambler number {0} is not a mature TEM-1 position".format(number))
    index = sum(1 for value in range(26, number) if value not in AMBLER_OMITTED)
    return 24 + index


def read_calls(path):
    grouped = defaultdict(set)
    skipped = 0
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            name = row["geneNames"]
            if name == "NA" or len(name) < 3 or not name[1:-1].isdigit():
                skipped += 1
                continue
            source = name[0]
            target = name[-1]
            number = int(name[1:-1])
            position = ambler_to_search(number)
            expected = prior.WILD_TYPE[position - 1]
            if source != expected:
                raise ValueError("{0} does not match the search sequence".format(name))
            grouped[position].add(target)
    return grouped, skipped


def column_identity(path, width):
    frequencies = np.zeros(width, dtype=np.float64)
    occupied = np.zeros(width, dtype=np.float64)
    query = None
    current = []
    sequences = 0

    def finish(chars):
        nonlocal query, sequences
        text = "".join(chars)
        if len(text) != width:
            raise ValueError("MSA row width is {0}".format(len(text)))
        letters = text.upper()
        if query is None:
            query = letters
            if query != prior.WILD_TYPE:
                raise ValueError("MSA query is not the search sequence")
        for index, char in enumerate(letters):
            if char == "-":
                continue
            occupied[index] += 1
            if char == query[index]:
                frequencies[index] += 1
        sequences += 1

    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(">"):
                if current:
                    finish(current)
                current = []
            else:
                current.append(line.strip())
        if current:
            finish(current)
    return frequencies / occupied, sequences


def benjamini_hochberg(pvalues):
    count = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(count, dtype=np.float64)
    running = 1.0
    for rank in range(count, 0, -1):
        index = order[rank - 1]
        running = min(running, pvalues[index] * count / rank)
        adjusted[index] = running
    return adjusted


def permutation_p(observed, null):
    return float((1 + np.sum(np.abs(null) >= abs(observed))) / (1 + len(null)))


def bootstrap_interval(values, outcome, draws, rng):
    count = len(values)
    samples = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        chosen = rng.integers(0, count, count)
        samples[index] = prior.spearman(values[chosen], outcome[chosen])
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def association_row(name, values, outcome, draws, rng):
    observed = prior.spearman(values, outcome)
    null = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        null[index] = prior.spearman(values, rng.permutation(outcome))
    low, high = bootstrap_interval(values, outcome, draws, rng)
    return {
        "test": name,
        "estimate": observed,
        "ci95_low": low,
        "ci95_high": high,
        "permutation_p": permutation_p(observed, null),
    }


def quadratic_coefficient(predictor, outcome):
    design = np.column_stack([np.ones(len(predictor)), predictor, predictor ** 2])
    coefficient, _, _, _ = np.linalg.lstsq(design, outcome, rcond=None)
    return float(coefficient[2])


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


def read_previous(path, key):
    values = {}
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            values[int(row["search_position"])] = float(row[key])
    return values


def main():
    out = Path(__file__).resolve().parent
    root = Path(__file__).resolve().parents[4]
    calls, skipped = read_calls(out / "data/elife73250_fig1_data2.csv")
    _, atoms, _ = prior.read_mmcif(
        root / "Mu-Protein/mu-search/pretest/tem1_1xpb_validation/structures/1XPB.cif"
    )
    _, _, mapped = prior.map_chain(atoms, prior.WILD_TYPE)
    functional = prior.functional_indices(mapped)
    coordinates = np.vstack([row["coordinate"] for row in mapped])
    covariance, singular = official_covariance(official_hessian(coordinates))
    if singular != 6:
        raise ValueError("official pseudoinverse removed {0} modes".format(singular))
    responses = response_matrix(covariance, unit_directions(SEVEN_DIRECTIONS))
    flexibility, forward, reverse = official_profiles(responses, functional)
    contacts = prior.contact_counts(coordinates, 8.0)
    distances = []
    for index, point in enumerate(coordinates):
        others = [site for site in functional if site != index]
        distances.append(float(np.linalg.norm(coordinates[others] - point, axis=1).min()))
    distances = np.array(distances, dtype=np.float64)
    identity, msa_count = column_identity(MSA_PATH, len(prior.WILD_TYPE))
    previous_forward = read_previous(
        root / "Mu-Protein/mu-search/pretest/tem1_1xpb_validation/dci_directional_features.csv",
        "dci_functional_to_i",
    )
    previous_reverse = read_previous(
        root / "Mu-Protein/mu-search/pretest/tem1_1xpb_validation/dci_directional_features.csv",
        "dci_i_to_functional",
    )
    previous_dfi = read_previous(
        root / "Mu-Protein/mu-search/pretest/tem1_1xpb_validation/anm_prs_features.csv",
        "dfi",
    )
    sites = []
    for row, flex, dci_forward, dci_reverse, contact, distance in zip(
        mapped, flexibility, forward, reverse, contacts, distances
    ):
        position = row["search_position"]
        positive = calls.get(position, set())
        sites.append(
            {
                "search_position": position,
                "search_residue": row["search_residue"],
                "auth_seq_id": row["auth_seq_id"],
                "positive_substitutions": len(positive),
                "positive_fraction": len(positive) / 19.0,
                "dfi_reference": float(flex),
                "dci_functional_to_i_reference": float(dci_forward),
                "dci_i_to_functional_reference": float(dci_reverse),
                "contact_count_8A": int(contact),
                "min_distance_to_functional_A": float(distance),
                "msa_identity": float(identity[position - 1]),
                "previous_dfi": previous_dfi[position],
                "previous_dci_functional_to_i": previous_forward[position],
                "previous_dci_i_to_functional": previous_reverse[position],
            }
        )
    write_csv(
        out / "site_features.csv",
        sites,
        list(sites[0].keys()),
    )
    outcome = np.array([row["positive_fraction"] for row in sites], dtype=np.float64)
    rng = np.random.default_rng(20261116)
    draws = 5000
    tests = [
        association_row("reference dci_functional_to_i vs positive fraction", np.array([row["dci_functional_to_i_reference"] for row in sites]), outcome, draws, rng),
        association_row("reference dci_i_to_functional vs positive fraction", np.array([row["dci_i_to_functional_reference"] for row in sites]), outcome, draws, rng),
        association_row("reference dfi vs positive fraction", np.array([row["dfi_reference"] for row in sites]), outcome, draws, rng),
        association_row("contact count vs positive fraction", np.array([row["contact_count_8A"] for row in sites], dtype=float), outcome, draws, rng),
        association_row("minimum functional distance vs positive fraction", np.array([row["min_distance_to_functional_A"] for row in sites]), outcome, draws, rng),
        association_row("MSA identity vs positive fraction", np.array([row["msa_identity"] for row in sites]), outcome, draws, rng),
        association_row("previous uniform-spring dci_functional_to_i vs positive fraction", np.array([row["previous_dci_functional_to_i"] for row in sites]), outcome, draws, rng),
    ]
    primary = tests[:3]
    adjusted = benjamini_hochberg([row["permutation_p"] for row in primary])
    for row, value in zip(primary, adjusted):
        row["bh_q_primary"] = float(value)
    for row in tests[3:]:
        row["bh_q_primary"] = ""
    predictor = np.array([row["dfi_reference"] for row in sites])
    observed_quadratic = quadratic_coefficient(predictor, outcome)
    null_quadratic = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        null_quadratic[index] = quadratic_coefficient(predictor, rng.permutation(outcome))
    dci_values = np.array([row["dci_functional_to_i_reference"] for row in sites])
    high = dci_values >= np.quantile(dci_values, 2.0 / 3.0)
    order = np.argsort(predictor)
    middle = np.zeros(len(sites), dtype=bool)
    middle[order[len(sites) // 3: 2 * len(sites) // 3]] = True
    selected = high & middle
    observed_diff = float(outcome[selected].mean() - outcome[~selected].mean())
    null_diff = np.empty(draws, dtype=np.float64)
    for index in range(draws):
        shuffled = rng.permutation(outcome)
        null_diff[index] = shuffled[selected].mean() - shuffled[~selected].mean()
    tests.append(
        {
            "test": "top-third reference DCI and middle-third reference DFI, mean fraction minus complement",
            "estimate": observed_diff,
            "ci95_low": "",
            "ci95_high": "",
            "permutation_p": permutation_p(observed_diff, null_diff),
            "bh_q_primary": "",
        }
    )
    tests.append(
        {
            "test": "quadratic coefficient of reference DFI",
            "estimate": observed_quadratic,
            "ci95_low": "",
            "ci95_high": "",
            "permutation_p": permutation_p(observed_quadratic, null_quadratic),
            "bh_q_primary": "",
        }
    )
    write_csv(out / "association_stats.csv", tests, ["test", "estimate", "ci95_low", "ci95_high", "permutation_p", "bh_q_primary"])
    summary = {
        "call_rows_skipped": skipped,
        "positive_substitutions": int(sum(len(value) for value in calls.values())),
        "sites_with_positive_call": len(calls),
        "sites_without_positive_call": 263 - len(calls),
        "msa_sequences": msa_count,
        "singular_values_removed": singular,
        "parity_forward": prior.spearman(forward, [previous_forward[row["search_position"]] for row in mapped]),
        "parity_reverse": prior.spearman(reverse, [previous_reverse[row["search_position"]] for row in mapped]),
        "parity_dfi": prior.spearman(flexibility, [previous_dfi[row["search_position"]] for row in mapped]),
        "reference_direction_spearman": prior.spearman(forward, reverse),
        "selected_sites": int(np.sum(selected)),
    }
    (out / "parity_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

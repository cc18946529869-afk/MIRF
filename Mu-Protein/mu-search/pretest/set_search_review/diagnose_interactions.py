"""CPU check of the six finished logs and a pre-specified ridge comparison.

Reads the pretest logs. Does not call MuFormer and does not train a policy.
Amino-acid pair terms are counted. They are not given a saturated fit: seed 0
has no substitution pair that appears five times. The regression uses only
features whose training-fold count meets the stated threshold.
"""

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, cg

RUNS = Path(
    "/data/cc233_data/work_dir/MIRF-musearch-pretest/Mu-Protein/mu-search/pretest/runs/musearch_gt"
)
MAPPING = Path(
    "/data/cc233_data/tmp/MIRF-tem1-1xpb-structure-validation-2026-11-16/"
    "Mu-Protein/mu-search/pretest/tem1_1xpb_validation/residue_mapping_1xpb.csv"
)
OUT = Path(__file__).resolve().parent / "outputs"
LAMBDAS = (10.0, 100.0)
FOLD_SEED = 20261117
REWIRE_SEED = 20261117
N_FOLDS = 5
SUB_MIN = 5
SITE_PAIR_MIN = 5
CONTACT_PAIR_MIN = 2
AA_PAIR_MIN = 2
CG_TOL = 1e-8
CG_MAXITER = 2000


def load_seed(seed):
    path = RUNS / f"seed{seed}" / f"musearch_gt_seed{seed}.queries.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    counts = Counter(row["purpose"] for row in rows)
    wt = next(row["sequence"] for row in rows if row["purpose"] == "initialization")
    reward = [row for row in rows if row["purpose"] == "reward"]
    sequences = [row["sequence"] for row in reward]
    scores = np.array([float(row["score"]) for row in reward], dtype=np.float64)
    return counts, wt, sequences, scores


def mutations(wt, sequence):
    return tuple((index + 1, sequence[index]) for index, (left, right) in enumerate(zip(wt, sequence)) if left != right)


def contact_edges(path):
    positions = []
    coords = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            positions.append(int(row["search_position"]))
            coords.append((float(row["ca_x"]), float(row["ca_y"]), float(row["ca_z"])))
    coordinates = np.asarray(coords, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.int64)
    delta = coordinates[:, None, :] - coordinates[None, :, :]
    distance = np.linalg.norm(delta, axis=2)
    edges = []
    for left in range(len(positions)):
        for right in range(left + 1, len(positions)):
            gap = abs(int(positions[left]) - int(positions[right]))
            if distance[left, right] < 8.0 and gap != 1:
                edges.append((int(positions[left]), int(positions[right])))
    return edges


def rewire(edges, seed):
    rng = np.random.default_rng(seed)
    current = {frozenset(edge) for edge in edges}
    order = [tuple(sorted(edge)) for edge in current]
    attempts = 20 * len(order)
    for _ in range(attempts):
        first, second = rng.choice(len(order), size=2, replace=False)
        a, b = order[first]
        c, d = order[second]
        if rng.random() < 0.5:
            proposed = ((a, c), (b, d))
        else:
            proposed = ((a, d), (b, c))
        accepted = []
        for left, right in proposed:
            if left == right or abs(left - right) == 1:
                accepted = []
                break
            key = frozenset((left, right))
            if key in current:
                accepted = []
                break
            accepted.append(tuple(sorted((left, right))))
        if len(accepted) == 2 and accepted[0] != accepted[1]:
            current.remove(frozenset(order[first]))
            current.remove(frozenset(order[second]))
            current.add(frozenset(accepted[0]))
            current.add(frozenset(accepted[1]))
            order[first] = accepted[0]
            order[second] = accepted[1]
    return order


def degree_map(edges):
    degree = Counter()
    for left, right in edges:
        degree[left] += 1
        degree[right] += 1
    return degree


def design(records, columns):
    rows = []
    cols = []
    index = {feature: number for number, feature in enumerate(columns)}
    for row_number, features in enumerate(records):
        for feature in features:
            column = index.get(feature)
            if column is not None:
                rows.append(row_number)
                cols.append(column)
    matrix = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float64), (rows, cols)),
        shape=(len(records), len(columns)),
    )
    return matrix


def add_intercept(matrix):
    ones = sparse.csr_matrix(np.ones((matrix.shape[0], 1), dtype=np.float64))
    return sparse.hstack([ones, matrix], format="csr")


def fit_predict(train_x, train_y, test_x, lam):
    train_x = add_intercept(train_x)
    test_x = add_intercept(test_x)
    penalty = np.full(train_x.shape[1], lam, dtype=np.float64)
    penalty[0] = 0.0
    normal = (train_x.T @ train_x).tocsc() + sparse.diags(penalty, format="csc")
    target = np.asarray(train_x.T @ train_y).ravel()
    operator = LinearOperator(
        normal.shape, matvec=lambda vector: normal @ vector, dtype=np.float64
    )
    coef, info = cg(operator, target, rtol=CG_TOL, maxiter=CG_MAXITER)
    if info != 0:
        raise RuntimeError(f"conjugate gradient did not converge: {info}")
    return np.asarray(test_x @ coef).ravel(), coef


def r2(y_true, y_pred):
    center = y_true - y_true.mean()
    residual = y_true - y_pred
    denom = float(center @ center)
    if denom == 0.0:
        return None
    return 1.0 - float(residual @ residual) / denom


def selected_pairs(train_muts, minimum, allowed):
    counts = Counter()
    for muts in train_muts:
        sites = sorted({site for site, _aa in muts})
        for left_i, left in enumerate(sites):
            for right in sites[left_i + 1 :]:
                pair = (left, right)
                if allowed is None or pair in allowed:
                    counts[pair] += 1
    return [pair for pair, count in counts.items() if count >= minimum]


def selected_subs(train_muts):
    counts = Counter(item for muts in train_muts for item in muts)
    return [("sub", site, aa) for (site, aa), count in counts.items() if count >= SUB_MIN]


def evaluate_seed(seed, sequences, scores, muts, contact, random_edges):
    rng = np.random.default_rng(FOLD_SEED + seed)
    order = rng.permutation(len(sequences))
    folds = np.array_split(order, N_FOLDS)
    contact_set = {tuple(sorted(edge)) for edge in contact}
    random_set = {tuple(sorted(edge)) for edge in random_edges}
    specs = {
        "main": (SUB_MIN, None, None),
        "site_pairs": (SITE_PAIR_MIN, None, "site"),
        "contact_site_pairs": (CONTACT_PAIR_MIN, contact_set, "site"),
        "rewired_site_pairs": (CONTACT_PAIR_MIN, random_set, "site"),
    }
    rows = []
    for lam in LAMBDAS:
        for model, (minimum, allowed, kind) in specs.items():
            for fold_id in range(N_FOLDS):
                test_index = folds[fold_id]
                train_index = np.concatenate([folds[i] for i in range(N_FOLDS) if i != fold_id])
                train_muts = [muts[i] for i in train_index]
                columns = selected_subs(train_muts)
                n_sub = len(columns)
                n_pair = 0
                if kind == "site":
                    pairs = selected_pairs(train_muts, minimum, allowed)
                    columns.extend(("sitepair", left, right) for left, right in pairs)
                    n_pair = len(pairs)
                pair_columns = [item[1:] for item in columns if item[0] == "sitepair"]
                train_records = []
                for i in train_index:
                    train_records.append(
                        [("sub", site, aa) for site, aa in muts[i]]
                        + [("sitepair", left, right) for left, right in pair_columns if left in {s for s, _ in muts[i]} and right in {s for s, _ in muts[i]}]
                    )
                test_records = []
                for i in test_index:
                    sites = {s for s, _ in muts[i]}
                    test_records.append(
                        [("sub", site, aa) for site, aa in muts[i]]
                        + [("sitepair", left, right) for left, right in pair_columns if left in sites and right in sites]
                    )
                train_x = design(train_records, columns)
                test_x = design(test_records, columns)
                train_y = scores[train_index] - scores[train_index].mean()
                prediction, _coef = fit_predict(train_x, train_y, test_x, lam)
                prediction = prediction + scores[train_index].mean()
                rows.append(
                    {
                        "seed": seed,
                        "model": model,
                        "lambda": lam,
                        "fold": fold_id,
                        "n_train": int(len(train_index)),
                        "n_test": int(len(test_index)),
                        "n_sub": n_sub,
                        "n_pair": n_pair,
                        "r2": r2(scores[test_index], prediction),
                    }
                )
    return rows


def support(muts):
    sub_counts = Counter(item for one in muts for item in one)
    aa_pairs = Counter()
    site_pairs = Counter()
    for one in muts:
        ordered = tuple(sorted(one))
        sites = tuple(sorted({site for site, _aa in one}))
        for left_i, left in enumerate(ordered):
            for right in ordered[left_i + 1 :]:
                aa_pairs[(left, right)] += 1
        for left_i, left in enumerate(sites):
            for right in sites[left_i + 1 :]:
                site_pairs[(left, right)] += 1
    def summarize(counter):
        values = list(counter.values())
        return {
            "unique": len(counter),
            "n_ge_2": sum(value >= 2 for value in values),
            "n_ge_5": sum(value >= 5 for value in values),
            "max": max(values) if values else 0,
        }
    return {"substitutions": summarize(sub_counts), "aa_pairs": summarize(aa_pairs), "site_pairs": summarize(site_pairs)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    loaded = [load_seed(seed) for seed in range(3)]
    wt = loaded[0][1]
    if any(item[1] != wt for item in loaded):
        raise RuntimeError("wild-type sequence differs across seeds")
    edges = contact_edges(MAPPING)
    random_edges = rewire(edges, REWIRE_SEED)
    if degree_map(edges) != degree_map(random_edges):
        raise RuntimeError("rewiring did not preserve degree")
    diagnosis = {
        "contact_edges": len(edges),
        "rewired_edges": len(random_edges),
        "rewire_seed": REWIRE_SEED,
        "fold_seed": FOLD_SEED,
        "lambdas": list(LAMBDAS),
        "thresholds": {
            "substitution": SUB_MIN,
            "unrestricted_site_pair": SITE_PAIR_MIN,
            "contact_site_pair": CONTACT_PAIR_MIN,
        },
        "seeds": [],
    }
    all_rows = []
    parsed = []
    for seed, (counts, _wt, sequences, scores) in enumerate(loaded):
        muts = [mutations(wt, sequence) for sequence in sequences]
        parsed.append((sequences, scores, muts))
        hamming = Counter(len(one) for one in muts)
        diagnosis["seeds"].append(
            {
                "seed": seed,
                "purpose_counts": dict(counts),
                "reward_n": len(sequences),
                "unique_sequences": len(set(sequences)),
                "score_min": float(scores.min()),
                "score_median": float(np.median(scores)),
                "score_max": float(scores.max()),
                "hamming": {str(key): hamming[key] for key in range(0, 6)},
                "support": support(muts),
            }
        )
    shared = set(parsed[0][0]) & set(parsed[1][0]) & set(parsed[2][0])
    diagnosis["sequences_shared_by_three_seeds"] = len(shared)
    for seed, (_sequences, scores, muts) in enumerate(parsed):
        print(f"fitting seed {seed}", flush=True)
        all_rows.extend(evaluate_seed(seed, parsed[seed][0], scores, muts, edges, random_edges))
    diagnosis_path = OUT / "diagnosis.json"
    diagnosis_path.write_text(json.dumps(diagnosis, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (OUT / "cv_folds.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    summary = []
    grouped = {}
    for row in all_rows:
        grouped.setdefault((row["seed"], row["model"], row["lambda"]), []).append(row["r2"])
    for key, values in sorted(grouped.items()):
        summary.append(
            {
                "seed": key[0],
                "model": key[1],
                "lambda": key[2],
                "mean_r2": float(np.mean(values)),
                "min_r2": float(np.min(values)),
                "max_r2": float(np.max(values)),
            }
        )
    (OUT / "cv_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

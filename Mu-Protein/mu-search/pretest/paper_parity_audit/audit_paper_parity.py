"""Protocol checks and the 16-sequence MIC scoring path.

Scoring loads safetensors only. It does not unpickle the original checkpoint.
"""

import csv
import math
from pathlib import Path


WILD_TYPE = (
    "MSIQHFRVALIPFFAAFCLPVFAHPETLVKVKDAEDQLGARVGYIELDLNSGKILESFRPEERFPMMSTF"
    "KVLLCGAVLSRVDAGQEQLGRRIHYSQNDLVEYSPVTEKHLTDGMTVRELCSAAITMSDNTAANLLLTTIG"
    "GPKELTAFLHNMGDHVTRLDRWEPELNEAIPNDERDTTMPAAMATTLRKLLTGELLTLASRQQLIDWMEAD"
    "KVAGPLLRSALPAGWFIADKSGAGERGSRGIIAALGPDGKPSRIVVIYTTGSQATMDERNRQIAEIGASLIKHW"
)

# Weinreich et al., Science 2006, Table 1, column without the g4205a promoter
# mutation. Names use the paper's numbering, which is two less than Ambler.
WEINREICH_MIC = {
    frozenset(): 0.088,
    frozenset({"G236S"}): 1.4,
    frozenset({"M180T"}): 0.063,
    frozenset({"M180T", "G236S"}): 32.0,
    frozenset({"E102K"}): 0.13,
    frozenset({"E102K", "G236S"}): 360.0,
    frozenset({"E102K", "M180T"}): 0.18,
    frozenset({"E102K", "M180T", "G236S"}): 360.0,
    frozenset({"A40G"}): 0.088,
    frozenset({"A40G", "G236S"}): 23.0,
    frozenset({"A40G", "M180T"}): 1.4,
    frozenset({"A40G", "M180T", "G236S"}): 360.0,
    frozenset({"A40G", "E102K"}): 1.4,
    frozenset({"A40G", "E102K", "G236S"}): 2100.0,
    frozenset({"A40G", "E102K", "M180T"}): 0.8,
    frozenset({"A40G", "E102K", "M180T", "G236S"}): 2900.0,
}


def parse_mutations(text):
    if text == "WT":
        return []
    return text.split(";")


def mutation_key(mutations):
    return frozenset(mutations)


def apply_mutations(wild_type, mutations):
    sequence = list(wild_type)
    for mutation in mutations:
        old = mutation[0]
        new = mutation[-1]
        position = int(mutation[1:-1])
        index = position - 1
        if sequence[index] != old:
            raise ValueError("position {0} is {1}, not {2}".format(position, sequence[index], old))
        sequence[index] = new
    return "".join(sequence)


def hamming(sequence, wild_type):
    return sum(left != right for left, right in zip(sequence, wild_type))


def rank_descending(values):
    order = sorted(range(len(values)), key=lambda index: (-values[index], index))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start
        while stop + 1 < len(order) and values[order[stop + 1]] == values[order[start]]:
            stop += 1
        average = (start + 1 + stop + 1) / 2
        for position in range(start, stop + 1):
            ranks[order[position]] = average
        start = stop + 1
    return ranks


def spearman(left, right):
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("spearman needs two equal columns")
    left_rank = rank_descending(left)
    right_rank = rank_descending(right)
    return pearson(left_rank, right_rank)


def pearson(left, right):
    count = len(left)
    mean_left = sum(left) / count
    mean_right = sum(right) / count
    numerator = sum((a - mean_left) * (b - mean_right) for a, b in zip(left, right))
    left_scale = math.sqrt(sum((a - mean_left) ** 2 for a in left))
    right_scale = math.sqrt(sum((b - mean_right) ** 2 for b in right))
    if left_scale == 0 or right_scale == 0:
        return None
    return numerator / (left_scale * right_scale)


def read_mic_table(path):
    rows = []
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for record in csv.DictReader(handle, delimiter="\t"):
            mutations = parse_mutations(record["mutation"].strip())
            key = mutation_key(mutations)
            published = WEINREICH_MIC.get(key)
            local_mic = float(record["score"])
            rows.append(
                {
                    "mutation": record["mutation"].strip(),
                    "mutations": mutations,
                    "order": len(mutations),
                    "mic": local_mic,
                    "weinreich_mic": published,
                    "mic_matches_weinreich": published is not None and abs(published - local_mic) < 1e-8,
                }
            )
    return rows


def weinreich_match_report(rows):
    keys = {mutation_key(row["mutations"]) for row in rows}
    missing = [name for name in WEINREICH_MIC if name not in keys]
    extra = [row["mutation"] for row in rows if row["weinreich_mic"] is None]
    mismatched = [row["mutation"] for row in rows if not row["mic_matches_weinreich"]]
    return {
        "n": len(rows),
        "missing": len(missing),
        "extra": extra,
        "mismatched": mismatched,
        "complete": len(rows) == 16 and not missing and not extra and not mismatched,
    }


def load_and_score(sequences):
    import importlib.util
    import os
    import sys

    import torch
    from fairseq.data import Dictionary
    from safetensors.torch import load_file

    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "src/flexs/flexs/landscapes/src"))
    from protein.models.pmlm import ProtPairwiseModel, prot_pair_1b_architecture

    model_path = root / "src/flexs/flexs/landscapes/landscape/muformer/muformer_landscape/src/model.py"
    spec = importlib.util.spec_from_file_location("muformer_module", model_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    dictionary = Dictionary.load(str(root / "src/flexs/flexs/landscapes/src/protein/dict.txt"))
    dictionary.add_symbol("<mask>")
    args = type("Args", (), {})()
    for key, value in {
        "arch": "prot_pmlm_1b",
        "pretrained_file": None,
        "tokens_per_sample": 512,
        "max_positions": 512,
        "dropout": 0.1,
        "attention_dropout": 0.1,
        "act_dropout": 0.0,
        "encoder_embed_dim": 1280,
        "encoder_layers": 34,
        "encoder_attention_heads": 20,
        "encoder_ffn_embed_dim": 5120,
        "share_encoder_input_output_embed": False,
        "encoder_learned_pos": False,
        "no_token_positional_embeddings": False,
        "num_segment": 2,
        "sentence_class_num": 2,
        "sent_loss": False,
        "apply_bert_init": False,
        "activation_fn": "relu",
        "pooler_activation_fn": "tanh",
        "encoder_normalize_before": False,
    }.items():
        setattr(args, key, value)
    prot_pair_1b_architecture(args)
    task = type("Task", (), {"dictionary": dictionary})()
    encoder = ProtPairwiseModel.build_model(args, task)
    weights_dir = Path("/data/cc233_data/tmp/musearch-convert/out")
    encoder.load_state_dict(load_file(str(weights_dir / "ur50.safetensors")), strict=True)
    muformer = module.Muformer(
        encoder=encoder,
        vocab_size=len(dictionary),
        encoder_dim=1280,
        hidden_size=256,
        num_heads=20,
        dropout=0.1,
        decoder_name="mono",
        joint_mode_enabled=True,
        joint_method="average",
    )
    muformer.load_state_dict(load_file(str(weights_dir / "muformer.safetensors")), strict=True)
    muformer.eval()
    device = "cpu"
    muformer.to(device)

    def encode(sequence):
        return dictionary.encode_line("<s> " + " ".join(sequence)).long()

    encoded = [encode(sequence) for sequence in sequences]
    width = max(item.numel() for item in encoded)
    tokens = torch.zeros(len(encoded), width, dtype=torch.long)
    mask = torch.zeros(len(encoded), width, dtype=torch.long)
    for row, item in enumerate(encoded):
        tokens[row, : item.numel()] = item
        mask[row, : item.numel()] = 1
    with torch.no_grad():
        output, _embedding = muformer(tokens, None, x1_mask=mask)
    return [float(item) for item in output.detach().float().cpu().view(-1)]


def write_validation(rows, predictions, path):
    mic_ranks = rank_descending([row["weinreich_mic"] for row in rows])
    pred_ranks = rank_descending(predictions)
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "mutation", "order", "mic", "weinreich_mic", "prediction",
                "mic_rank", "prediction_rank", "purpose",
            ],
        )
        writer.writeheader()
        for row, prediction, mic_rank, pred_rank in zip(rows, predictions, mic_ranks, pred_ranks):
            writer.writerow(
                {
                    "mutation": row["mutation"],
                    "order": row["order"],
                    "mic": "{0:.8g}".format(row["mic"]),
                    "weinreich_mic": "{0:.8g}".format(row["weinreich_mic"]),
                    "prediction": "{0:.8g}".format(prediction),
                    "mic_rank": "{0:.4g}".format(mic_rank),
                    "prediction_rank": "{0:.4g}".format(pred_rank),
                    "purpose": "validation_inference",
                }
            )


def main():
    root = Path(__file__).resolve().parent
    source = (
        Path(__file__).resolve().parents[2]
        / "src/flexs/flexs/landscapes/landscape/muformer/muformer_landscape/CFX_test_from_pan.tsv"
    )
    rows = read_mic_table(source)
    report = weinreich_match_report(rows)
    if not report["complete"]:
        print("mic_match_failed", report)
        return 2
    sequences = [apply_mutations(WILD_TYPE, row["mutations"]) for row in rows]
    if sequences[0] != WILD_TYPE:
        raise RuntimeError("WT row did not rebuild the reference sequence")
    predictions = load_and_score(sequences)
    wild_type_score = predictions[0]
    print("wt_score", wild_type_score)
    if abs(wild_type_score + 0.0726844072341919) > 1e-3:
        raise RuntimeError("wild-type score does not match the recorded landscape score")
    write_validation(rows, predictions, root / "mic16_validation.csv")
    print("spearman", spearman([row["mic"] for row in rows], predictions))
    print("n", len(predictions))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

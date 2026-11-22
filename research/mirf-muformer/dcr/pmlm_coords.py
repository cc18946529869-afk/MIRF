"""Map BLAT reference positions to PMLM token indices for every saved row.

All 4996 sequences are length-286 substitutions, so a reference position has
one token index. That index is checked from the dictionary, not assumed.
This module does not store hidden states and does not train.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import feature_align
import lobes

DICTIONARY = Path(__file__).resolve().parents[3] / "Mu-Protein" / "mu-former" / "src" / "protein" / "dict.txt"
DOMAIN_MAP = Path(__file__).resolve().parents[1] / "prep" / "domain_map.tsv"
CHECKED_POSITIONS = (24, 82, 182, 286)
FULL_LENGTH = 286
MATURE_LENGTH = 263


def encode_sequence(dictionary, sequence: str):
    from pmlm_check import _encode

    return _encode(dictionary, sequence)


def load_dictionary():
    from fairseq.data import Dictionary

    return Dictionary.load(str(DICTIONARY))


def token_index_for_reference(dictionary, sequence: str, ref_pos: int) -> int:
    """Return the token index of one 1-based reference position, or refuse."""
    if ref_pos < 1 or ref_pos > len(sequence):
        raise ValueError(f"reference position {ref_pos} is outside the variant sequence")
    tokens = encode_sequence(dictionary, sequence)
    if int(tokens[0]) != int(dictionary.index("<s>")):
        raise ValueError("token 0 is not <s>")
    amino = sequence[ref_pos - 1]
    if int(tokens[ref_pos]) != int(dictionary.index(amino)):
        raise ValueError(f"token index {ref_pos} is not {amino} at reference position {ref_pos}")
    return int(ref_pos)


def per_row_lobe_indices(domain_map: Path, ref_pos: int) -> tuple[np.ndarray, np.ndarray]:
    """Source and other token indices for this mutation's own lobe."""
    source, other = lobes.source_and_other_tokens(domain_map, ref_pos)
    end_token = FULL_LENGTH + 1
    for name, index in (("source", source), ("other", other)):
        if int(index.min()) < lobes.MATURE_START or int(index.max()) > lobes.MATURE_END:
            raise ValueError(f"{name} includes the signal peptide, <s>, or the end token")
        if 0 in set(index.tolist()) or end_token in set(index.tolist()):
            raise ValueError(f"{name} includes a special token")
    return source, other


def audit_table(index: feature_align.FeatureIndex, destination: Path) -> dict:
    """Check every saved sequence. The report stores counts, not embeddings."""
    dictionary = load_dictionary()
    site_indices = []
    source_lengths = []
    source_keys = []
    failures = []
    for sequence, row in index.rows.items():
        ref_pos = row["ref_pos"]
        try:
            if len(sequence) != FULL_LENGTH:
                raise ValueError("the variant is not the 286-residue BLAT sequence")
            if row["mut_aa"] != sequence[ref_pos - 1]:
                raise ValueError("the variant residue does not match the mutation")
            expected_mature = int(ref_pos) - lobes.MATURE_START + 1
            if row["mature_pos"] != expected_mature:
                raise ValueError("mature_pos is not ref_pos minus the 23-residue signal peptide")
            if expected_mature < 1 or expected_mature > MATURE_LENGTH:
                raise ValueError("the mutation is outside the mature protein")
            site = token_index_for_reference(dictionary, sequence, ref_pos)
            for position in CHECKED_POSITIONS:
                token_index_for_reference(dictionary, sequence, position)
            source, _other = per_row_lobe_indices(DOMAIN_MAP, ref_pos)
            site_indices.append(site)
            source_lengths.append(int(source.shape[0]))
            source_keys.append(tuple(int(item) for item in source.tolist()))
        except (ValueError, KeyError) as exc:
            failures.append({"ref_pos": ref_pos, "error": str(exc)})
            if len(failures) >= 5:
                break
    unique_source_lengths = sorted(set(source_lengths))
    distinct_sources = len(set(source_keys))
    payload = {
        "passed": not failures and len(site_indices) == len(index.rows),
        "rows": len(index.rows),
        "checked_rows": len(site_indices),
        "full_length": FULL_LENGTH,
        "mature_length": MATURE_LENGTH,
        "mature_start": lobes.MATURE_START,
        "mature_pos_equals_ref_pos_minus_23": not failures and len(site_indices) == len(index.rows),
        "checked_positions": list(CHECKED_POSITIONS),
        "token_index_equals_reference_position": not failures and len(site_indices) == len(index.rows),
        "special_token_offset": "<s> occupies index 0; residue r is token r",
        "source_lengths": unique_source_lengths,
        "distinct_source_index_sets": distinct_sources,
        "source_and_other_shared_across_rows": distinct_sources == 1,
        "unmappable_policy": "a length other than 286, a mismatched residue, or a non-mature site is refused",
        "failures": failures,
        "embeddings_stored": False,
    }
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n")
    return payload

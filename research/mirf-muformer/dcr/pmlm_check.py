"""Load the existing 650M PMLM and check the frozen residual readout.

The checkpoint is the local file. This module does not download weights, does
not train, and does not read a fine-tuned μFormer or ``fc_reduce``. A failed
load is written as a failure.
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path

import torch

import frozen_pmlm
import lobes

CHECKPOINT = Path("/home/cc233/data/cc233_data/work_dir/MIRF/Mu-Protein/pretrained/pmlm_650m.pt")
DICTIONARY = Path(__file__).resolve().parents[3] / "Mu-Protein" / "mu-former" / "src" / "protein" / "dict.txt"


def _write(path: Path, payload: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def _encode(dictionary, sequence: str) -> torch.Tensor:
    line = "<s> " + " ".join(sequence)
    encoded = dictionary.encode_line(line, append_eos=True, add_if_not_exist=False)
    return encoded.long()


def verify_frozen_pmlm(checkpoint: Path, domain_map: Path, destination: Path, sequence: str) -> dict:
    """Run one forward pass and record the frozen-readout checks."""
    checkpoint = Path(checkpoint)
    destination = Path(destination)
    if not checkpoint.is_file():
        return _write(
            destination,
            {
                "passed": False,
                "checkpoint": str(checkpoint),
                "error": f"checkpoint file is missing: {checkpoint}",
                "downloaded": False,
                "trained": False,
            },
        )
    try:
        from fairseq.data import Dictionary

        model = frozen_pmlm.load_independent_pretrained(str(checkpoint))
        dictionary = Dictionary.load(str(DICTIONARY))
        prob_calls = {"count": 0}
        encoder_prob = getattr(model.encoder, "prob", None)

        def _count_prob(*args, **kwargs):
            prob_calls["count"] += 1
            return encoder_prob(*args, **kwargs)

        if encoder_prob is not None:
            model.encoder.prob = _count_prob
        model.train(True)
        tokens = _encode(dictionary, sequence).unsqueeze(0)
        hidden = model(tokens)
        if hidden.dim() != 3:
            raise RuntimeError("frozen output does not have axes time, batch, channel")
        embed_dim = int(model.encoder.args.encoder_embed_dim)
        if int(hidden.shape[0]) != int(tokens.shape[1]) or int(hidden.shape[1]) != 1 or int(hidden.shape[2]) != embed_dim:
            raise RuntimeError(
                f"output shape {tuple(hidden.shape)} does not match tokens and encoder_embed_dim {embed_dim}"
            )
        if int(tokens[0, 0]) != int(dictionary.index("<s>")):
            raise RuntimeError("token 0 is not <s>")
        checked_ref_pos = (24, 82, 182, 286)
        for ref_pos in checked_ref_pos:
            amino = sequence[ref_pos - 1]
            if int(tokens[0, ref_pos]) != int(dictionary.index(amino)):
                raise RuntimeError(f"token index {ref_pos} is not the amino acid at that 1-based reference position")
        requires_grad = any(bool(parameter.requires_grad) for parameter in model.parameters())
        if requires_grad or model.training or model.encoder.training:
            raise RuntimeError("the loaded PMLM is not frozen in eval mode")
        site_matches = []
        lobe_min = []
        for ref_pos in (24, 82):
            source, other = lobes.source_and_other_tokens(domain_map, ref_pos)
            if int(source.min()) < lobes.MATURE_START or int(other.min()) < lobes.MATURE_START:
                raise RuntimeError("a lobe average includes <s> or the signal peptide")
            if 0 in set(source.tolist()) or 0 in set(other.tolist()):
                raise RuntimeError("a lobe average includes token 0")
            site, source_mean, other_mean = model.representations(
                tokens,
                ref_pos,
                torch.as_tensor(source),
                torch.as_tensor(other),
            )
            source_index = torch.as_tensor(source, dtype=torch.long)
            other_index = torch.as_tensor(other, dtype=torch.long)
            if not torch.equal(site, hidden[ref_pos]):
                raise RuntimeError(f"the site vector is not hidden state index {ref_pos}")
            if not torch.allclose(source_mean, hidden.index_select(0, source_index).mean(dim=0)):
                raise RuntimeError("the source lobe mean is not the mean of its reference-position tokens")
            if not torch.allclose(other_mean, hidden.index_select(0, other_index).mean(dim=0)):
                raise RuntimeError("the other lobe mean is not the mean of its reference-position tokens")
            site_matches.append(ref_pos)
            lobe_min.append(int(min(int(source.min()), int(other.min()))))
        if prob_calls["count"] != 0:
            raise RuntimeError("the frozen forward called the language-model head")
        if hasattr(model, "fc_reduce"):
            raise RuntimeError("the frozen module exposes fc_reduce")
        payload = {
            "passed": True,
            "checkpoint": str(checkpoint),
            "downloaded": False,
            "trained": False,
            "read_finetuned_hidden": False,
            "read_fc_reduce": False,
            "prob_calls": prob_calls["count"],
            "training": bool(model.training),
            "requires_grad": requires_grad,
            "embed_dim": embed_dim,
            "output_shape": [int(hidden.shape[0]), int(hidden.shape[1]), int(hidden.shape[2])],
            "token_index_checked_ref_pos": list(checked_ref_pos),
            "site_token_index_equals_ref_pos": site_matches,
            "lobe_min_token_index": lobe_min,
            "signal_peptide_and_bos_excluded": True,
            "error": None,
        }
        return _write(destination, payload)
    except Exception as exc:
        payload = {
            "passed": False,
            "checkpoint": str(checkpoint),
            "downloaded": False,
            "trained": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        return _write(destination, payload)

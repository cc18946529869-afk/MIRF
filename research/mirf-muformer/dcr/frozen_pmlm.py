"""A second pretrained PMLM used only by the residual branch.

It is loaded by the same fairseq call as ``Trainer``, then set to ``eval``
with gradients disabled. It does not read a fine-tuned μFormer encoder and it
does not call the language-model head.
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn


def _per_row_indices(index, batch: int, device) -> list[torch.Tensor]:
    """One 1-d token index per batch row. Lengths may differ."""
    if isinstance(index, (list, tuple)):
        if len(index) != batch:
            raise ValueError("each sequence needs its own lobe token indices")
        return [torch.as_tensor(item, dtype=torch.long, device=device).view(-1) for item in index]
    tensor = torch.as_tensor(index, dtype=torch.long, device=device)
    if tensor.ndim != 2 or int(tensor.shape[0]) != batch:
        raise ValueError("each sequence needs its own lobe token indices")
    return [tensor[row] for row in range(batch)]


class FrozenPmlm(nn.Module):
    """Hold an independent encoder and read its last block output."""

    def __init__(self, encoder: nn.Module):
        super().__init__()
        if encoder is None:
            raise ValueError("the frozen PMLM needs its own encoder")
        self.encoder = encoder
        self.encoder.eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        super().train(False)
        self.encoder.eval()
        return self

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        self.encoder.eval()
        with torch.no_grad():
            output = self.encoder(tokens)
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.dim() != 3:
            raise ValueError("frozen PMLM output must have axes time, batch, channel")
        return hidden

    def representations(
        self,
        tokens: torch.Tensor,
        site_token_index: torch.Tensor,
        source_token_index: torch.Tensor,
        other_token_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-sequence site vector and lobe means.

        ``hidden`` is arranged as time, batch, channel. ``site_token_index``
        has one token index per batch row. Source and other indices are one
        vector per row, because the two lobes have different lengths. A single
        integer is not applied to every row.
        """
        hidden = self.forward(tokens)
        batch = int(hidden.shape[1])
        site_index = torch.as_tensor(site_token_index, dtype=torch.long, device=hidden.device)
        if site_index.ndim != 1 or int(site_index.shape[0]) != batch:
            raise ValueError("each sequence needs its own site token index")
        source_rows = _per_row_indices(source_token_index, batch, hidden.device)
        other_rows = _per_row_indices(other_token_index, batch, hidden.device)
        if (
            torch.any(site_index <= 0)
            or torch.any(site_index >= hidden.shape[0])
            or any(torch.any(item <= 0) or torch.any(item >= hidden.shape[0]) for item in source_rows)
            or any(torch.any(item <= 0) or torch.any(item >= hidden.shape[0]) for item in other_rows)
        ):
            raise ValueError("a token index includes <s>, a non-positive index, or the end of the sequence")
        rows = torch.arange(batch, device=hidden.device)
        site = hidden[site_index, rows]
        source = torch.stack([hidden[index, row].mean(dim=0) for row, index in zip(rows, source_rows)])
        other = torch.stack([hidden[index, row].mean(dim=0) for row, index in zip(rows, other_rows)])
        return site, source, other


def load_independent_pretrained(checkpoint_path: str) -> FrozenPmlm:
    """Load ``pmlm`` weights the way ``Trainer`` does, as a separate module.

    ``run_designated_single_seed`` and the small-batch test call this.
    ``embed_sequences`` receives the loaded module. Formal μFormer training
    is a later step.
    """
    from fairseq import checkpoint_utils

    tokendict = Path(__file__).resolve().parents[3] / "Mu-Protein" / "mu-former" / "src" / "protein"
    source = str(tokendict.parent)
    if source not in sys.path:
        sys.path.insert(0, source)
    importlib.import_module("protein")
    models, _args, _task = checkpoint_utils.load_model_ensemble_and_task(
        str(checkpoint_path).split(os.pathsep),
        arg_overrides={"data": str(tokendict)},
    )
    return FrozenPmlm(models[0])


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def embed_sequences(model: FrozenPmlm, sequences, feature_index, *, checkpoint: Path, batch_size: int = 2) -> dict:
    """Read frozen 1280-d vectors for explicit sequences.

    The tensors stay in memory. The returned metadata records the checkpoint
    hash and the sequence identity, and it does not contain the vectors.
    """
    import hashlib

    import pmlm_coords

    ordered = [str(sequence) for sequence in sequences]
    if not ordered:
        raise ValueError("frozen PMLM embedding needs at least one sequence")
    if int(batch_size) < 1:
        raise ValueError("batch size must be positive")
    dictionary = pmlm_coords.load_dictionary()
    sites = []
    sources = []
    others = []
    for start in range(0, len(ordered), int(batch_size)):
        chunk = ordered[start : start + int(batch_size)]
        tokens = torch.stack([pmlm_coords.encode_sequence(dictionary, sequence) for sequence in chunk])
        site_index = []
        source_index = []
        other_index = []
        for sequence in chunk:
            ref_pos = feature_index.row(sequence)["ref_pos"]
            site_index.append(pmlm_coords.token_index_for_reference(dictionary, sequence, ref_pos))
            source, other = pmlm_coords.per_row_lobe_indices(pmlm_coords.DOMAIN_MAP, ref_pos)
            source_index.append(torch.as_tensor(source, dtype=torch.long))
            other_index.append(torch.as_tensor(other, dtype=torch.long))
        site, source, other = model.representations(
            tokens,
            torch.tensor(site_index, dtype=torch.long),
            source_index,
            other_index,
        )
        if int(site.shape[-1]) != 1280 or int(source.shape[-1]) != 1280 or int(other.shape[-1]) != 1280:
            raise RuntimeError("frozen PMLM representations are not 1280-dimensional")
        sites.append(site.detach().cpu())
        sources.append(source.detach().cpu())
        others.append(other.detach().cpu())
    identity = hashlib.sha256("\n".join(ordered).encode()).hexdigest()
    return {
        "site": torch.cat(sites, dim=0),
        "source": torch.cat(sources, dim=0),
        "other": torch.cat(others, dim=0),
        "sequence": ordered,
        "metadata": {
            "checkpoint_sha256": _sha256(checkpoint),
            "sequence_sha256": identity,
            "rows": len(ordered),
            "hidden": 1280,
            "coordinate_map_version": "blat_ref_pos_is_token_after_bos_v1",
            "embeddings_stored": False,
            "lm_head_called": False,
            "fc_reduce_called": False,
        },
    }

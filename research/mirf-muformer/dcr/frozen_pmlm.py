"""A second pretrained PMLM used only by the residual branch.

It is loaded by the same fairseq call as ``Trainer``, then set to ``eval``
with gradients disabled. It does not read a fine-tuned μFormer encoder and it
does not call the language-model head.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn


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
        site_ref_pos: int,
        source_ref_positions: torch.Tensor,
        other_ref_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Site vector and the two lobe means on the variant sequence.

        ``site_ref_pos`` is both the 1-based reference position and the token
        index. The caller supplies lobe positions that already exclude the
        signal peptide, ``<s>`` and the end token.
        """
        hidden = self.forward(tokens)
        site_index = int(site_ref_pos)
        if site_index <= 0 or site_index >= hidden.shape[0]:
            raise ValueError("the site token index is outside the frozen sequence")
        source_index = torch.as_tensor(source_ref_positions, dtype=torch.long, device=hidden.device)
        other_index = torch.as_tensor(other_ref_positions, dtype=torch.long, device=hidden.device)
        if torch.any(source_index <= 0) or torch.any(other_index <= 0):
            raise ValueError("a lobe average includes <s> or a non-positive index")
        site = hidden[site_index]
        source = hidden.index_select(0, source_index).mean(dim=0)
        other = hidden.index_select(0, other_index).mean(dim=0)
        return site, source, other


def load_independent_pretrained(checkpoint_path: str) -> FrozenPmlm:
    """Load ``pmlm`` weights the way ``Trainer`` does, as a separate module.

    Tests do not call this. Formal training is a later step.
    """
    from fairseq import checkpoint_utils

    tokendict = Path(__file__).resolve().parents[3] / "Mu-Protein" / "mu-former" / "src" / "protein"
    models, _args, _task = checkpoint_utils.load_model_ensemble_and_task(
        str(checkpoint_path).split(os.pathsep),
        arg_overrides={"data": str(tokendict)},
    )
    return FrozenPmlm(models[0])

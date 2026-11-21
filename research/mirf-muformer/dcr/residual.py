"""Residual predictor. The μFormer score is not an input of ``forward``.

Fourteen Potts values stay in their sealed columns. The projection sees the
28-dimensional concatenation of masked values and the mask. Widths are the
locked ``d_s=64``, ``d_l=32``, ``d_r=32``. After concatenation the network is
a 64-wide GELU, dropout 0.1, and a zero-initialized scalar.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import potts_features

D_S = 64
D_L = 32
D_R = 32
CONCAT_WIDTH = D_S + 2 * D_L + D_R
HIDDEN = 64
DROPOUT = 0.1
ARM_FULL = "full"
ARM_SUM = "sum"
ARM_NONE = "no_potts"
ARMS = (ARM_FULL, ARM_SUM, ARM_NONE)
DELTA_E_INDEX = potts_features.FEATURE_NAMES.index("delta_e")
STD_DELTA_E_INDEX = potts_features.FEATURE_NAMES.index("std_delta_e")


def arm_mask(observed: torch.Tensor | object, arm: str):
    """Mask for one residual arm. Unobserved slots stay unobserved.

    The sum arm keeps only ``delta_e`` and its standard deviation. The other
    Potts slots are masked, so a stored numeric zero is not treated as an
    observation of those components. The no-Potts arm masks all 14 slots.
    """
    values = torch.as_tensor(observed, dtype=torch.float32)
    if values.shape[-1] != 14:
        raise ValueError("the observation mask must have 14 columns")
    if arm == ARM_FULL:
        return values
    if arm == ARM_SUM:
        keep = torch.zeros(14, dtype=values.dtype, device=values.device)
        keep[DELTA_E_INDEX] = 1.0
        keep[STD_DELTA_E_INDEX] = 1.0
        return values * keep
    if arm == ARM_NONE:
        return torch.zeros_like(values)
    raise ValueError(f"unknown residual arm {arm}")


def pack_observation(values, mask) -> torch.Tensor:
    """Concatenate ``[x odot m; m]``. Masked zeros stay distinguishable."""
    numeric = torch.as_tensor(values, dtype=torch.float32)
    flags = torch.as_tensor(mask, dtype=torch.float32)
    if numeric.shape != flags.shape or numeric.shape[-1] != 14:
        raise ValueError("values and mask must share the 14 feature columns")
    return torch.cat((numeric * flags, flags), dim=-1)


class ResidualPredictor(nn.Module):
    """Map frozen representations and a 14-column Potts row to ``r``."""

    def __init__(self, embed_dim: int):
        super().__init__()
        if CONCAT_WIDTH != 160:
            raise ValueError("the locked widths do not sum to 160")
        if embed_dim <= 0:
            raise ValueError("the frozen width must be positive")
        self.embed_dim = int(embed_dim)
        self.site = nn.Linear(self.embed_dim, D_S)
        self.lobe = nn.Linear(self.embed_dim, D_L)
        self.dcr = nn.Linear(28, D_R)
        self.hidden = nn.Linear(CONCAT_WIDTH, HIDDEN)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(DROPOUT)
        self.output = nn.Linear(HIDDEN, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def project(
        self,
        site: torch.Tensor,
        source_lobe: torch.Tensor,
        other_lobe: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return the 160-wide concatenation, before the GELU layer."""
        packed = pack_observation(values, mask)
        return torch.cat(
            (
                self.site(site),
                self.lobe(source_lobe),
                self.lobe(other_lobe),
                self.dcr(packed),
            ),
            dim=-1,
        )

    def forward(
        self,
        site: torch.Tensor,
        source_lobe: torch.Tensor,
        other_lobe: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.dropout(self.activation(self.hidden(self.project(site, source_lobe, other_lobe, values, mask))))
        return self.output(hidden).squeeze(-1)


def update_mae_epoch(best_epoch, best_mae, stall, epoch: int, mae: float, patience: int):
    """Advance the residual epoch rule by one validation MAE.

    The recorded epoch changes only when the internal MAE is a new low.
    ``stall`` counts epochs that did not set a new low. The schedule stops
    once that count reaches ``patience``.
    """
    if patience < 1:
        raise ValueError("patience must be positive")
    if best_epoch is None or float(mae) < float(best_mae):
        return epoch, float(mae), 0, False
    stall = int(stall) + 1
    return best_epoch, float(best_mae), stall, stall >= int(patience)


def _batches(indices: torch.Tensor, batch_size: int, generator: torch.Generator):
    if indices.numel() == 0:
        raise ValueError("a residual split has no rows")
    order = torch.randperm(indices.numel(), generator=generator)
    shuffled = indices[order]
    for start in range(0, shuffled.numel(), batch_size):
        yield shuffled[start : start + batch_size]


def _epoch(model, optimizer, batch, rows, target):
    model.train()
    pred = model(
        batch["site"][rows].detach(),
        batch["source"][rows].detach(),
        batch["other"][rows].detach(),
        batch["values"][rows].detach(),
        batch["mask"][rows].detach(),
    )
    loss = torch.mean(torch.abs(pred - target[rows].detach()))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def _mae(model, batch, rows, target) -> float:
    model.eval()
    with torch.no_grad():
        pred = model(
            batch["site"][rows],
            batch["source"][rows],
            batch["other"][rows],
            batch["values"][rows],
            batch["mask"][rows],
        )
        return float(torch.mean(torch.abs(pred - target[rows])).detach())


def fit_predictor(
    batch: dict[str, torch.Tensor],
    target: torch.Tensor,
    train_idx,
    val_idx,
    *,
    embed_dim: int,
    max_epochs: int = 300,
    patience: int = 30,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 0,
):
    """Select an epoch on the internal rows, then retrain from a new init.

    The second fit uses every OOF row for exactly the selected number of
    epochs. The output layer starts at zero both times. No learning-rate
    schedule is used.
    """
    if max_epochs < 1:
        raise ValueError("max_epochs must be positive")
    train_rows = torch.as_tensor(train_idx, dtype=torch.long)
    val_rows = torch.as_tensor(val_idx, dtype=torch.long)
    all_rows = torch.arange(target.shape[0], dtype=torch.long)
    generator = torch.Generator()
    generator.manual_seed(int(seed))

    def one_pass(model, optimizer, rows):
        for chunk in _batches(rows, batch_size, generator):
            _epoch(model, optimizer, batch, chunk, target)

    model = ResidualPredictor(embed_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_epoch = None
    best_mae = None
    stall = 0
    ran = 0
    for epoch in range(1, max_epochs + 1):
        one_pass(model, optimizer, train_rows)
        mae = _mae(model, batch, val_rows, target)
        best_epoch, best_mae, stall, stop = update_mae_epoch(best_epoch, best_mae, stall, epoch, mae, patience)
        ran = epoch
        if stop:
            break

    retrained = ResidualPredictor(embed_dim)
    if float(retrained.output.weight.abs().sum()) != 0.0 or float(retrained.output.bias.abs().sum()) != 0.0:
        raise RuntimeError("the reinitialized output layer is not zero")
    optimizer = torch.optim.AdamW(retrained.parameters(), lr=learning_rate, weight_decay=weight_decay)
    for _epoch_index in range(int(best_epoch)):
        one_pass(retrained, optimizer, all_rows)
    retrained.eval()
    return {
        "model": retrained,
        "best_epoch": int(best_epoch),
        "best_mae": float(best_mae),
        "selection_epochs_run": ran,
        "retrain_epochs_run": int(best_epoch),
        "retrain_rows": int(all_rows.numel()),
        "selection_train_rows": int(train_rows.numel()),
    }

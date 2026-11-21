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
_SELECTION = "selection"
_RETRAIN = "retrain"


def _mix_seed(*parts: int) -> int:
    """Fold integers into one seed. The result is not a continuation of a generator."""
    value = 0x9E3779B97F4A7C15
    for part in parts:
        value ^= int(part) & 0xFFFFFFFFFFFFFFFF
        value = (value * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    return int(value)


def stream_seed(outer_seed: int, stage: str) -> int:
    """Seed for one stage's shuffle stream. Selection and retrain do not share it."""
    if stage == _SELECTION:
        return _mix_seed(int(outer_seed), 1)
    if stage == _RETRAIN:
        return _mix_seed(int(outer_seed), 2)
    raise ValueError(f"unknown residual stage {stage}")


def dropout_seed(outer_seed: int, stage: str, epoch: int, batch_index: int) -> int:
    """Seed the dropout mask for one arm-independent batch.

    The same outer seed, stage, epoch and batch index reproduce the same mask.
    A different stage does not continue the previous generator.
    """
    if stage == _SELECTION:
        stage_id = 1
    elif stage == _RETRAIN:
        stage_id = 2
    else:
        raise ValueError(f"unknown residual stage {stage}")
    return _mix_seed(int(outer_seed), 100 + stage_id, int(epoch), int(batch_index))


def dropout_generator_for(outer_seed: int, stage: str, epoch: int, batch_index: int, device) -> torch.Generator:
    """Build a fresh generator. It is not the shuffle generator and not the global RNG."""
    if getattr(device, "type", "cpu") == "cuda":
        generator = torch.Generator(device=device)
    else:
        generator = torch.Generator()
    generator.manual_seed(dropout_seed(outer_seed, stage, epoch, batch_index))
    return generator


def apply_feature_dropout(hidden: torch.Tensor, dropout: nn.Dropout, generator: torch.Generator, mask_log=None) -> torch.Tensor:
    """Apply inverted dropout from ``generator`` so arms can share one mask."""
    probability = float(dropout.p)
    if probability == 0.0 or not dropout.training:
        return hidden
    keep_probability = 1.0 - probability
    keep = torch.bernoulli(
        torch.full(hidden.shape, keep_probability, dtype=hidden.dtype, device=hidden.device),
        generator=generator,
    )
    if mask_log is not None:
        mask_log.append(keep.detach().cpu().clone())
    return hidden * keep / keep_probability


def shuffled_row_order(indices: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Permute rows with ``generator`` only. An empty split is refused."""
    if indices.numel() == 0:
        raise ValueError("a residual split has no rows")
    order = torch.randperm(indices.numel(), generator=generator)
    return indices[order]


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
        dropout_generator: torch.Generator | None = None,
        dropout_mask_log=None,
    ) -> torch.Tensor:
        hidden = self.activation(self.hidden(self.project(site, source_lobe, other_lobe, values, mask)))
        if self.training:
            if dropout_generator is None:
                hidden = self.dropout(hidden)
            else:
                hidden = apply_feature_dropout(hidden, self.dropout, dropout_generator, dropout_mask_log)
        return self.output(hidden).squeeze(-1)


def shared_initial_state(embed_dim: int, outer_seed: int) -> dict[str, torch.Tensor]:
    """One initial parameter set for every arm and both training stages.

    The output layer is the zero initialization from ``ResidualPredictor``.
    A different outer seed may draw a different set. The global RNG is restored.
    """
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        torch.manual_seed(int(outer_seed))
        model = ResidualPredictor(embed_dim)
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _clone_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in state.items()}


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


def _epoch(model, optimizer, batch, rows, target, dropout_generator, mask_log):
    model.train()
    pred = model(
        batch["site"][rows].detach(),
        batch["source"][rows].detach(),
        batch["other"][rows].detach(),
        batch["values"][rows].detach(),
        batch["mask"][rows].detach(),
        dropout_generator=dropout_generator,
        dropout_mask_log=mask_log,
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
    outer_seed: int = 0,
    initial_state: dict[str, torch.Tensor] | None = None,
):
    """Select an epoch on the internal rows, then retrain from the same init.

    Both models load ``initial_state``. The shuffle streams of the two stages
    are seeded separately, so retraining does not continue the selection
    generator. Dropout masks are seeded by stage, epoch and batch index.
    The output layer starts at zero both times. No learning-rate schedule is used.
    """
    if max_epochs < 1:
        raise ValueError("max_epochs must be positive")
    train_rows = torch.as_tensor(train_idx, dtype=torch.long)
    val_rows = torch.as_tensor(val_idx, dtype=torch.long)
    all_rows = torch.arange(target.shape[0], dtype=torch.long)
    if initial_state is None:
        initial_state = shared_initial_state(embed_dim, int(outer_seed))
    initial_state = _clone_state(initial_state)
    selection_generator = torch.Generator()
    selection_generator.manual_seed(stream_seed(int(outer_seed), _SELECTION))
    retrain_generator = torch.Generator()
    retrain_generator.manual_seed(stream_seed(int(outer_seed), _RETRAIN))

    def one_pass(model, optimizer, rows, generator, stage, epoch_number, orders, first_mask):
        permutation = shuffled_row_order(rows, generator)
        orders.append(permutation.detach().cpu().clone())
        for batch_index, start in enumerate(range(0, int(permutation.numel()), batch_size)):
            chunk = permutation[start : start + batch_size]
            mask_log = first_mask if epoch_number == 1 and batch_index == 0 else None
            dropout_generator = dropout_generator_for(int(outer_seed), stage, epoch_number, batch_index, chunk.device)
            _epoch(model, optimizer, batch, chunk, target, dropout_generator, mask_log)

    model = ResidualPredictor(embed_dim)
    model.load_state_dict(initial_state)
    selection_initial = _clone_state(model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best_epoch = None
    best_mae = None
    stall = 0
    ran = 0
    selection_orders = []
    selection_first_mask = []
    for epoch in range(1, max_epochs + 1):
        one_pass(model, optimizer, train_rows, selection_generator, _SELECTION, epoch, selection_orders, selection_first_mask)
        mae = _mae(model, batch, val_rows, target)
        best_epoch, best_mae, stall, stop = update_mae_epoch(best_epoch, best_mae, stall, epoch, mae, patience)
        ran = epoch
        if stop:
            break

    retrained = ResidualPredictor(embed_dim)
    retrained.load_state_dict(initial_state)
    if float(retrained.output.weight.abs().sum()) != 0.0 or float(retrained.output.bias.abs().sum()) != 0.0:
        raise RuntimeError("the reinitialized output layer is not zero")
    retrain_initial = _clone_state(retrained.state_dict())
    optimizer = torch.optim.AdamW(retrained.parameters(), lr=learning_rate, weight_decay=weight_decay)
    retrain_orders = []
    retrain_first_mask = []
    for epoch_number in range(1, int(best_epoch) + 1):
        one_pass(retrained, optimizer, all_rows, retrain_generator, _RETRAIN, epoch_number, retrain_orders, retrain_first_mask)
    retrained.eval()
    return {
        "model": retrained,
        "best_epoch": int(best_epoch),
        "best_mae": float(best_mae),
        "selection_epochs_run": ran,
        "retrain_epochs_run": int(best_epoch),
        "retrain_rows": int(all_rows.numel()),
        "selection_train_rows": int(train_rows.numel()),
        "outer_seed": int(outer_seed),
        "initial_state": initial_state,
        "selection_initial": selection_initial,
        "retrain_initial": retrain_initial,
        "selection_orders": selection_orders,
        "retrain_orders": retrain_orders,
        "selection_index_source": train_rows.detach().cpu().clone(),
        "retrain_index_source": all_rows.detach().cpu().clone(),
        "selection_first_dropout_mask": selection_first_mask[0],
        "retrain_first_dropout_mask": retrain_first_mask[0],
    }

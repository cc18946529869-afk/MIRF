"""One real A6000 step for the production Trainer.

This is not the designated experiment. It constructs one Trainer, runs one
training batch through the forward and backward passes, and does not call
``Trainer.train``. Launch it with ``CUDA_VISIBLE_DEVICES`` set to one free GPU.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path


def _require_one_visible_gpu() -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible == "" or "," in visible:
        raise RuntimeError("gpu_smoke requires CUDA_VISIBLE_DEVICES to name exactly one GPU")
    return visible


def main() -> dict:
    physical = _require_one_visible_gpu()
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to this process")
    if int(torch.cuda.device_count()) != 1:
        raise RuntimeError("this process does not see exactly one GPU")
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        raise RuntimeError("distributed mode is already initialized")

    import designated_seed
    import sealed_config

    trainer_module = designated_seed.load_trainer_class()
    module = sys.modules[trainer_module.__module__]
    distributed_calls = []

    def refuse_distributed(local_rank):
        distributed_calls.append(local_rank)
        raise RuntimeError("distributed initialization was called")

    module.init_distributed_mode = refuse_distributed
    output = Path(tempfile.mkdtemp(prefix="mirf-dcr-gpu-smoke-"))
    if (output / ".git").exists() or any(parent.name == ".git" or (parent / ".git").exists() for parent in output.parents):
        raise RuntimeError("the smoke directory is inside a repository")
    request = {"muformer_config": sealed_config.muformer_config()}
    trainer = designated_seed.production_trainer_factory(
        request,
        trainer_cls=trainer_module,
        output_dir=output,
    )

    def refuse_train(*args, **kwargs):
        raise RuntimeError("Trainer.train was called")

    trainer.train = refuse_train
    if trainer.distributed or trainer.num_gpus != 1 or distributed_calls:
        raise RuntimeError("the Trainer entered the distributed path")
    if len(trainer.models) != 1 or len(trainer.optimizers) != 1:
        raise RuntimeError("the smoke constructed more than one model")
    model = trainer.models[0]
    devices = {str(parameter.device) for parameter in model.parameters()}
    if devices != {"cuda:0"}:
        raise RuntimeError(f"model parameters are not on cuda:0: {sorted(devices)}")
    optimizer = trainer.optimizers[0]
    if [group["lr"] for group in optimizer.param_groups] != [1e-5, 1e-4]:
        raise RuntimeError("the Adam groups are not the sealed encoder and decoder rates")

    # One real pair from the designated training TSV. Two rows are enough to
    # exercise the forward and backward passes without a training epoch.
    (train_loader, _train_df), _valid = trainer.dataset.get_dataloader(
        "train_valid",
        2,
        return_df=True,
        resample_train_valid=False,
        num_workers=0,
        shuffle=False,
        train_row_index=None,
        drop_last=True,
    )
    batch = next(iter(train_loader))
    left, right = batch[0], batch[1]
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output_scores = model(
        left["x"].to(trainer.device),
        right["x"].to(trainer.device),
        x1_mask=left["mask"].to(trainer.device),
        x2_mask=right["mask"].to(trainer.device),
    )
    loss = trainer.criterion(output_scores.view(-1), left["label_af"].to(trainer.device))
    if not bool(torch.isfinite(loss).item()):
        raise RuntimeError("the smoke loss is not finite")
    loss.backward()
    finite_grads = 0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        if not bool(torch.isfinite(parameter.grad).all().item()):
            raise RuntimeError("a gradient is not finite")
        finite_grads += 1
    if finite_grads < 1:
        raise RuntimeError("backward did not produce a gradient")
    if optimizer.state:
        raise RuntimeError("the optimizer was stepped")
    report = {
        "cuda_visible_devices": physical,
        "gpu_name": torch.cuda.get_device_name(0),
        "device_index": int(torch.cuda.current_device()),
        "device_count": int(torch.cuda.device_count()),
        "parameter_device": "cuda:0",
        "distributed": False,
        "distributed_init_called": False,
        "trainer_train_called": False,
        "models": 1,
        "batch_rows": int(left["label_af"].shape[0]),
        "loss": float(loss.detach().cpu()),
        "finite_gradients": finite_grads,
        "adam_lrs": [1e-5, 1e-4],
        "optimizer_stepped": False,
    }
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return report


if __name__ == "__main__":
    main()

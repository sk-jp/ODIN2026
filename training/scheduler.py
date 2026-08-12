from __future__ import annotations

import inspect
import math
from typing import Any

import torch

from cosine_annealing_with_warmup_lr import CosineAnnealingWithWarmupLR


_LIGHTNING_KEYS = {"interval", "frequency", "monitor", "strict", "name"}


def _pytorch_scheduler_class(name: str) -> type:
    scheduler_class = getattr(torch.optim.lr_scheduler, name, None)
    base_class = getattr(
        torch.optim.lr_scheduler, "LRScheduler", torch.optim.lr_scheduler._LRScheduler
    )
    if (
        scheduler_class is None
        or not inspect.isclass(scheduler_class)
        or not issubclass(scheduler_class, base_class)
    ):
        raise ValueError(
            f"Unknown learning-rate scheduler {name!r}. Specify "
            "CosineAnnealingWithWarmupLR or a torch.optim.lr_scheduler class name."
        )
    return scheduler_class


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
    *,
    total_steps: int,
    max_epochs: int,
) -> dict[str, Any]:
    """Build a scheduler and its Lightning configuration from YAML values."""
    scheduler_cfg = dict(cfg)
    scheduler_name = str(scheduler_cfg.pop("name", "CosineAnnealingWithWarmupLR"))
    nested_kwargs = scheduler_cfg.pop("kwargs", {})
    if not isinstance(nested_kwargs, dict):
        raise TypeError("Scheduler.kwargs must be a mapping")

    lightning_cfg = {
        key: scheduler_cfg.pop(key)
        for key in tuple(scheduler_cfg)
        if key in _LIGHTNING_KEYS
    }
    duplicated = set(nested_kwargs).intersection(scheduler_cfg)
    if duplicated:
        names = ", ".join(sorted(duplicated))
        raise ValueError(f"Scheduler arguments specified twice: {names}")
    scheduler_kwargs = {**nested_kwargs, **scheduler_cfg}

    if scheduler_name == "CosineAnnealingWithWarmupLR":
        aliases = {
            "start_learning_rate": "warmup_start_lr",
            "end_learning_rate": "eta_min",
        }
        for source, target in aliases.items():
            if source in scheduler_kwargs:
                if target in scheduler_kwargs:
                    raise ValueError(
                        f"Specify only one of Scheduler.{source} and Scheduler.{target}"
                    )
                scheduler_kwargs[target] = scheduler_kwargs.pop(source)

        epochs = int(scheduler_kwargs.pop("max_epochs", max_epochs))
        scheduler_kwargs.setdefault(
            "steps_per_epoch", max(1, math.ceil(int(total_steps) / max(1, epochs)))
        )
        scheduler = CosineAnnealingWithWarmupLR(
            optimizer,
            max_epochs=epochs,
            **scheduler_kwargs,
        )
        lightning_cfg.setdefault("interval", "step")
    else:
        scheduler_class = _pytorch_scheduler_class(scheduler_name)
        try:
            scheduler = scheduler_class(optimizer, **scheduler_kwargs)
        except TypeError as error:
            raise TypeError(
                f"Invalid arguments for Scheduler {scheduler_name}: {error}"
            ) from error
        lightning_cfg.setdefault("interval", "epoch")

    lightning_cfg.setdefault("frequency", 1)
    lightning_cfg["scheduler"] = scheduler
    return lightning_cfg

import math

import torch


class CosineAnnealingWithWarmupLR(torch.optim.lr_scheduler._LRScheduler):
    """Step-wise warmup + cosine decay with max duration specified in epochs.

    Warmup can be specified in one of three ways:
      - warmup_percent: percentage of total optimizer steps, e.g. 5.0
      - warmup_ratio: fraction of total optimizer steps, e.g. 0.05
      - warmup_steps: absolute number of optimizer steps

    Args:
        max_epochs: Cosine horizon in epochs, as written in YAML.
        steps_per_epoch: Optimizer steps per epoch, computed by Lightning.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_epochs: int,
        steps_per_epoch: int,
        warmup_steps: int | None = None,
        warmup_ratio: float | None = None,
        warmup_percent: float | None = None,
        warmup_start_lr: float = 0.00001,
        eta_min: float = 0.00001,
        last_epoch: int = -1,
    ):
        self.max_epochs = max(1, int(max_epochs))
        self.steps_per_epoch = max(1, int(steps_per_epoch))
        self.total_steps = max(1, self.max_epochs * self.steps_per_epoch)
        self.warmup_steps = self._resolve_warmup_steps(
            warmup_steps=warmup_steps,
            warmup_ratio=warmup_ratio,
            warmup_percent=warmup_percent,
        )
        # Some YAML parsers treat values such as ``1e-7`` as strings. Normalize
        # the public numeric arguments here so scientific notation works too.
        self.warmup_start_lr = float(warmup_start_lr)
        self.eta_min = float(eta_min)
        super().__init__(optimizer, last_epoch)

    def _resolve_warmup_steps(
        self,
        warmup_steps: int | None,
        warmup_ratio: float | None,
        warmup_percent: float | None,
    ) -> int:
        if warmup_percent is not None:
            percent = float(warmup_percent)
            if not 0.0 <= percent <= 100.0:
                raise ValueError(f"warmup_percent must be in [0, 100], got {percent}")
            return int(round(self.total_steps * percent / 100.0))

        if warmup_ratio is not None:
            ratio = float(warmup_ratio)
            if not 0.0 <= ratio <= 1.0:
                raise ValueError(f"warmup_ratio must be in [0, 1], got {ratio}")
            return int(round(self.total_steps * ratio))

        if warmup_steps is None:
            return 0
        return max(0, int(warmup_steps))

    def get_lr(self):
        step = max(0, self.last_epoch)

        if self.warmup_steps > 0 and step < self.warmup_steps:
            denom = max(1, self.warmup_steps)
            warmup_progress = step / denom
            return [
                self.warmup_start_lr
                + (base_lr - self.warmup_start_lr) * warmup_progress
                for base_lr in self.base_lrs
            ]

        decay_steps = max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, max(0.0, (step - self.warmup_steps) / decay_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return [
            self.eta_min + (base_lr - self.eta_min) * cosine
            for base_lr in self.base_lrs
        ]

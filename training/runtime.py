from __future__ import annotations

import argparse
import os
import shutil
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import yaml
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.strategies import DDPStrategy
from torch.nn.parallel import DistributedDataParallel


def build_loggers(cfg: dict, output: Path, debug: bool):
    """Always persist metrics locally and optionally mirror them to W&B."""
    loggers = [
        CSVLogger(
            save_dir=output,
            name="logs",
            flush_logs_every_n_steps=1,
        )
    ]
    if not bool(cfg.get("Logger", {}).get("use_wandb", True)):
        return loggers

    try:
        from lightning.pytorch.loggers import WandbLogger

        wandb_logger = WandbLogger(
            project=cfg["Logger"]["project"],
            name=output.name,
            save_dir=output,
            offline=debug,
        )
        wandb_logger.log_hyperparams(
            {
                "learning_rate": float(cfg["Optimizer"]["learning_rate"]),
                "projector_learning_rate": float(
                    cfg["Optimizer"].get(
                        "projector_learning_rate", cfg["Optimizer"]["learning_rate"]
                    )
                ),
                "lora_learning_rate": float(
                    cfg["Optimizer"].get(
                        "lora_learning_rate", cfg["Optimizer"]["learning_rate"]
                    )
                ),
                "structured_learning_rate": float(
                    cfg["Optimizer"].get(
                        "structured_learning_rate",
                        cfg["Optimizer"].get(
                            "projector_learning_rate", cfg["Optimizer"]["learning_rate"]
                        ),
                    )
                ),
                "projector_only_epochs": int(
                    cfg["Optimizer"].get("projector_only_epochs", 0)
                ),
                "metric_schema_version": int(
                    cfg.get("Logger", {}).get("metric_schema_version", 1)
                ),
                "lr_scheduler": cfg["Scheduler"].get(
                    "name", "CosineAnnealingWithWarmupLR"
                ),
                "lr_start": float(
                    cfg["Scheduler"].get(
                        "start_learning_rate", cfg["Optimizer"]["learning_rate"]
                    )
                ),
                "lr_end": float(
                    cfg["Scheduler"].get(
                        "end_learning_rate", cfg["Optimizer"]["learning_rate"]
                    )
                ),
            }
        )
        loggers.append(wandb_logger)
    except Exception as error:
        print(f"W&B unavailable ({error}); continuing with local CSV logs.", flush=True)
    return loggers


class DefaultStreamDDPStrategy(DDPStrategy):
    """Initialize DDP on the default CUDA stream during ordinary training.

    Lightning 2.6.0 initializes DDP on a side stream. With recent PyTorch
    versions this leaves the AccumulateGrad nodes on a different stream from
    the forward/backward pass and produces a synchronization warning.
    """

    def _setup_model(self, model):
        device_ids = self.determine_ddp_device_ids()
        context = (
            torch.cuda.stream(torch.cuda.default_stream())
            if device_ids is not None
            else nullcontext()
        )
        with context:
            return DistributedDataParallel(
                module=model, device_ids=device_ids, **self._ddp_kwargs
            )


def parse_gpu_ids(value: str) -> list[int]:
    try:
        gpu_ids = [int(item.strip()) for item in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "GPU IDs must be comma-separated integers (for example: 0,1)"
        ) from error
    if not gpu_ids or any(gpu_id < 0 for gpu_id in gpu_ids):
        raise argparse.ArgumentTypeError("GPU IDs must be non-negative integers")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise argparse.ArgumentTypeError("GPU IDs must not contain duplicates")
    return gpu_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train CBCT feature-to-report model (ver.1)"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--gpus",
        type=parse_gpu_ids,
        metavar="ID[,ID...]",
        help="comma-separated GPU IDs (for example: 0 or 0,1)",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--lr", type=float, help="override all trainable-group learning rates"
    )
    parser.add_argument("--projector-lr", type=float)
    parser.add_argument("--lora-lr", type=float)
    parser.add_argument("--structured-lr", type=float)
    parser.add_argument("--projector-only-epochs", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument(
        "--overfit-cases",
        type=int,
        metavar="N",
        help="train and validate on the same first N training cases",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--outdir-ext", "--outdir_ext", dest="outdir_ext")
    return parser.parse_args()


def load_config(path: Path, args: argparse.Namespace) -> dict:
    with path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if args.gpus is not None:
        cfg["General"]["devices"] = args.gpus
    if args.lr is not None:
        cfg["Optimizer"]["learning_rate"] = args.lr
        cfg["Optimizer"]["projector_learning_rate"] = args.lr
        cfg["Optimizer"]["lora_learning_rate"] = args.lr
        cfg["Optimizer"]["structured_learning_rate"] = args.lr
    if args.projector_lr is not None:
        cfg["Optimizer"]["projector_learning_rate"] = args.projector_lr
    if args.lora_lr is not None:
        cfg["Optimizer"]["lora_learning_rate"] = args.lora_lr
    if args.structured_lr is not None:
        cfg["Optimizer"]["structured_learning_rate"] = args.structured_lr
    if args.projector_only_epochs is not None:
        if args.projector_only_epochs < 0:
            raise ValueError("--projector-only-epochs must be non-negative")
        cfg["Optimizer"]["projector_only_epochs"] = args.projector_only_epochs
    if args.epochs is not None:
        if args.epochs <= 0:
            raise ValueError("--epochs must be positive")
        cfg["General"]["epochs"] = args.epochs
    if args.overfit_cases is not None:
        if args.overfit_cases <= 1:
            raise ValueError("--overfit-cases must be at least 2")
        cfg["Data"]["overfit_cases"] = args.overfit_cases
    if args.seed is not None:
        cfg["General"]["seed"] = args.seed
    if args.debug:
        cfg["General"]["debug"] = True
        cfg["General"]["epochs"] = 2
    return cfg


def _format_run_value(value: float) -> str:
    return f"{float(value):.8g}"


def run_condition_suffix(cfg: dict, args: argparse.Namespace) -> str:
    optimizer = cfg["Optimizer"]
    fallback = float(optimizer["learning_rate"])
    separated = "StructuredFindings" in cfg or any(
        key in optimizer
        for key in (
            "projector_learning_rate",
            "lora_learning_rate",
            "structured_learning_rate",
        )
    )
    if separated:
        projector = optimizer.get("projector_learning_rate", fallback)
        lora = optimizer.get("lora_learning_rate", fallback)
        structured = optimizer.get("structured_learning_rate", projector)
        stage = int(optimizer.get("projector_only_epochs", 0))
        return (
            f"-PLR{_format_run_value(projector)}"
            f"-LLR{_format_run_value(lora)}"
            f"-SLR{_format_run_value(structured)}"
            f"-STAGE{stage}"
        )
    if args.lr is not None:
        return f"-LR{_format_run_value(args.lr)}"
    return ""


def make_run_directory(cfg: dict, config_path: Path, args: argparse.Namespace) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{stamp}_{config_path.stem}{run_condition_suffix(cfg, args)}"
    if cfg["Data"].get("overfit_cases"):
        name += f"-overfit{int(cfg['Data']['overfit_cases'])}"
    if args.outdir_ext:
        name += f"-{args.outdir_ext}"
    output = Path(cfg["General"]["output_root"]) / name
    output.mkdir(parents=True, exist_ok=True)
    cfg["General"]["output_path"] = str(output)
    shutil.copy2(config_path, output / "source_config.yaml")
    with (output / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False, allow_unicode=True)
    return output

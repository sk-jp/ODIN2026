from __future__ import annotations

import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor

from balanced_datamodule import BalancedCBCTReportDataModule
from balanced_lightning_module import BalancedCBCTReportLightningModule
from callbacks import BestArtifactSaver, CleanProgressBar
from runtime import (
    DefaultStreamDDPStrategy,
    build_loggers,
    load_config,
    make_run_directory,
    parse_args,
)


def main():
    args = parse_args()
    cfg = load_config(args.config, args)
    seed_everything(int(cfg["General"]["seed"]), workers=True)
    torch.set_float32_matmul_precision("high")
    output = make_run_directory(cfg, args.config, args)
    datamodule = BalancedCBCTReportDataModule(cfg)
    datamodule.setup("fit")
    model = BalancedCBCTReportLightningModule(cfg, datamodule.tokenizer)
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        CleanProgressBar(refresh_rate=1),
        BestArtifactSaver(output, metrics=("clinical_score_v2",)),
    ]
    debug = bool(cfg["General"].get("debug", False))
    loggers = build_loggers(cfg, output, debug)
    devices = cfg["General"]["devices"]
    count = len(devices) if isinstance(devices, list) else int(devices)
    strategy = (
        DefaultStreamDDPStrategy(find_unused_parameters=False) if count > 1 else "auto"
    )
    trainer = Trainer(
        accelerator=cfg["General"]["accelerator"],
        devices=devices,
        num_nodes=int(cfg["General"]["num_nodes"]),
        strategy=strategy,
        precision=cfg["General"]["precision"],
        max_epochs=int(cfg["General"]["epochs"]),
        val_check_interval=cfg["General"]["validation_interval"],
        check_val_every_n_epoch=int(cfg["General"].get("validation_every_n_epochs", 1)),
        accumulate_grad_batches=int(cfg["Optimizer"]["accumulate_grad_batches"]),
        gradient_clip_val=float(cfg["Optimizer"]["gradient_clip_val"]),
        logger=loggers,
        callbacks=callbacks,
        log_every_n_steps=1,
        limit_train_batches=0.03 if debug else 1.0,
        limit_val_batches=0.05 if debug else 1.0,
        num_sanity_val_steps=0
        if debug
        else int(cfg["General"]["num_sanity_val_steps"]),
        deterministic=bool(cfg["General"]["deterministic"]),
        enable_checkpointing=False,
        use_distributed_sampler=False,
    )
    trainer.fit(model, datamodule=datamodule)
    if trainer.is_global_zero:
        model.save_components(output / "final")


if __name__ == "__main__":
    main()

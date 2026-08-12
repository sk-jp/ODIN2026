from __future__ import annotations

from dataclasses import replace

from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from balanced_collate import BalancedReportCollator
from balanced_dataset import (
    DistributedEvalSampler,
    EpochTemplateSampler,
    ManifestDataset,
    load_manifest,
)
from collate import DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_PROMPT


class BalancedCBCTReportDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.train_sampler = None

    def setup(self, stage=None):
        if stage not in ("fit", None):
            return
        data, model = self.cfg["Data"], self.cfg["Model"]
        train = load_manifest(data["train_manifest"])
        overfit_cases = int(data.get("overfit_cases", 0))
        self.overfit = overfit_cases > 0
        if self.overfit:
            if overfit_cases > len(train):
                raise ValueError("overfit_cases exceeds train cases")
            train = [
                replace(record, sampling_weight=1.0) for record in train[:overfit_cases]
            ]
            valid = train
        else:
            valid = load_manifest(data["valid_manifest"])
        grid, channels = tuple(data["pool_grid"]), int(data["feature_channels"])
        shuffle_enabled = bool(
            self.cfg.get("Evaluation", {}).get("image_shuffle_enabled", True)
        )
        seed = int(self.cfg["General"].get("seed", 42))
        dataset_options = {
            "token_mode": data.get("image_token_mode", "global_region"),
            "max_region_tokens": int(data.get("max_region_tokens", 48)),
            "include_global_tokens": bool(data.get("include_global_tokens", True)),
        }
        grounding_enabled = bool(self.cfg.get("Grounding", {}).get("enabled", False))
        self.train_dataset = ManifestDataset(
            train,
            grid,
            channels,
            include_shuffled_image=grounding_enabled,
            shuffle_seed=seed,
            **dataset_options,
        )
        self.val_dataset = ManifestDataset(
            valid,
            grid,
            channels,
            include_shuffled_image=shuffle_enabled,
            shuffle_seed=seed,
            **dataset_options,
        )
        self.train_sampler = EpochTemplateSampler(
            train,
            data.get("template_cap", 3),
            seed,
            enabled=not self.overfit,
            near_cap=data.get("near_template_cap", data.get("template_cap", 3)),
        )
        self.valid_sampler = DistributedEvalSampler(len(valid))
        self.tokenizer = AutoTokenizer.from_pretrained(
            model["pretrained_path"], local_files_only=True, use_fast=True
        )
        self.collator = BalancedReportCollator(
            self.tokenizer,
            int(data["max_report_tokens"]),
            data.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
            data.get("user_prompt", DEFAULT_USER_PROMPT),
            float(data.get("coverage_token_weight", 0.5)),
            float(data.get("finding_token_weight", 1.5)),
            bool(data.get("clause_weighting", True)),
        )

    def set_epoch(self, epoch):
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)

    def _loader(self, dataset, sampler, section):
        options = dict(self.cfg["Data"][section])
        workers = int(options.pop("num_workers"))
        persistent = bool(options.pop("persistent_workers", False))
        return DataLoader(
            dataset,
            sampler=sampler,
            num_workers=workers,
            collate_fn=self.collator,
            persistent_workers=workers > 0 and persistent,
            **options,
        )

    def train_dataloader(self):
        return self._loader(self.train_dataset, self.train_sampler, "train_loader")

    def val_dataloader(self):
        return self._loader(self.val_dataset, self.valid_sampler, "valid_loader")

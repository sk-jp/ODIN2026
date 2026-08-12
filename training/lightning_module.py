from __future__ import annotations

import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
from lightning.pytorch import LightningModule
from transformers import AutoModelForImageTextToText, BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

from metrics import compute_overfit_metrics, compute_report_metrics
from model import CBCTReportModel, FeatureProjector, StructuredFindingHead
from scheduler import build_lr_scheduler


class CBCTReportLightningModule(LightningModule):
    def __init__(self, cfg: dict, tokenizer) -> None:
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        model_cfg = cfg["Model"]
        compute_dtype = (
            torch.bfloat16
            if cfg["General"]["precision"] == "bf16-mixed"
            else torch.float16
        )
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=model_cfg["quantization"]["quant_type"],
            bnb_4bit_use_double_quant=bool(model_cfg["quantization"]["double_quant"]),
            bnb_4bit_compute_dtype=compute_dtype,
        )
        base = AutoModelForImageTextToText.from_pretrained(
            model_cfg["pretrained_path"],
            local_files_only=True,
            quantization_config=quantization,
            dtype=compute_dtype,
            device_map={"": int(os.environ.get("LOCAL_RANK", "0"))},
        )
        base = prepare_model_for_kbit_training(
            base,
            use_gradient_checkpointing=bool(model_cfg["gradient_checkpointing"]),
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
        lora = model_cfg["lora"]
        base = get_peft_model(
            base,
            LoraConfig(
                r=int(lora["rank"]),
                lora_alpha=int(lora["alpha"]),
                lora_dropout=float(lora["dropout"]),
                bias=lora["bias"],
                target_modules=list(lora["target_modules"]),
                exclude_modules=lora.get("exclude_modules"),
                task_type=TaskType.CAUSAL_LM,
            ),
        )
        hidden_size = int(base.config.text_config.hidden_size)
        self._lora_parameters = tuple(
            parameter for parameter in base.parameters() if parameter.requires_grad
        )
        projector = FeatureProjector(
            int(cfg["Data"]["feature_channels"]) * 2,
            int(model_cfg["projector_hidden_dim"]),
            hidden_size,
            float(model_cfg["projector_dropout"]),
            max_region_label=int(model_cfg.get("max_region_label", 64)),
            region_metadata_dim=10,
        ).to(dtype=compute_dtype)
        structured_cfg = cfg.get("StructuredFindings", {})
        structured_head = None
        if bool(structured_cfg.get("enabled", False)):
            structured_head = StructuredFindingHead(
                hidden_size, float(structured_cfg.get("dropout", 0.1))
            ).to(dtype=compute_dtype)
        self.model = CBCTReportModel(base, projector, structured_head)
        self.projector_only_epochs = int(
            cfg.get("Optimizer", {}).get("projector_only_epochs", 0)
        )
        self.validation_outputs: list[tuple[list, list, list, list, list]] = []
        self.latest_valid_metrics: dict[str, float] = {}
        self._train_loss_sum = 0.0
        self._train_samples = 0
        if not any(p.requires_grad for p in self.parameters()):
            raise RuntimeError("No trainable parameters found")

    def _reset_rope_deltas(self) -> None:
        # Qwen3.5 caches a generation-batch-sized tensor on the base model.
        # Soft-prefix training also uses inputs_embeds, so a stale validation cache
        # can otherwise be reused by the next (usually smaller) training batch.
        for module in self.model.language_model.modules():
            if hasattr(module, "rope_deltas"):
                module.rope_deltas = None

    @staticmethod
    def _image_arguments(batch, prefix: str = ""):
        return {
            "image_features": batch[f"{prefix}image_features"],
            "image_attention_mask": batch.get(f"{prefix}image_attention_mask"),
            "image_token_type_ids": batch.get(f"{prefix}image_token_type_ids"),
            "region_label_ids": batch.get(f"{prefix}region_label_ids"),
            "region_metadata": batch.get(f"{prefix}region_metadata"),
        }

    def forward(self, batch, image_prefix: str = ""):
        self._reset_rope_deltas()
        return self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            **self._image_arguments(batch, image_prefix),
        )

    def on_train_epoch_start(self) -> None:
        self.log(
            "train_lora_active",
            float(self.current_epoch >= self.projector_only_epochs),
            on_step=False,
            on_epoch=True,
            logger=True,
            sync_dist=True,
        )

    def on_after_backward(self) -> None:
        if self.current_epoch < self.projector_only_epochs:
            for parameter in self._lora_parameters:
                parameter.grad = None

    def training_step(self, batch, _batch_idx):
        loss = self(batch).loss
        size = len(batch["case_ids"])
        self.log(
            "train_loss",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=size,
            sync_dist=True,
        )
        self._train_loss_sum += float(loss.detach()) * size
        self._train_samples += size
        return loss

    def on_before_optimizer_step(self, optimizer) -> None:
        """Log the LR actually used by every named optimizer group."""
        for group in optimizer.param_groups:
            name = str(group.get("name", "default"))
            self.log(
                f"learning_rate/{name}",
                float(group["lr"]),
                on_step=True,
                on_epoch=False,
                logger=True,
                prog_bar=False,
                sync_dist=False,
            )
        self.log(
            "learning_rate",
            float(optimizer.param_groups[0]["lr"]),
            on_step=True,
            on_epoch=False,
            logger=True,
            prog_bar=False,
            sync_dist=False,
        )

    @torch.no_grad()
    def validation_step(self, batch, _batch_idx):
        output = self(batch)
        size = len(batch["case_ids"])
        self.log(
            "valid_loss",
            output.loss,
            on_step=False,
            on_epoch=True,
            batch_size=size,
            sync_dist=True,
        )
        generated = self.generate(batch)
        swapped = [None] * size
        swap_sources = [None] * size
        if self.cfg["Data"].get("overfit_cases") and size > 1:
            swapped_batch = dict(batch)
            swapped_batch["image_features"] = batch["image_features"].roll(1, dims=0)
            swapped = self.generate(swapped_batch)
            swap_sources = batch["case_ids"][-1:] + batch["case_ids"][:-1]
        self.validation_outputs.append(
            (generated, batch["references"], batch["case_ids"], swapped, swap_sources)
        )

    @torch.no_grad()
    def generate(self, batch, image_prefix: str = "") -> list[str]:
        self._reset_rope_deltas()
        multimodal = self.model.multimodal_inputs(
            input_ids=batch["prompt_input_ids"],
            attention_mask=torch.ones_like(batch["prompt_input_ids"]),
            **self._image_arguments(batch, image_prefix),
        )
        evaluation = self.cfg["Evaluation"]
        generation_kwargs = {
            "max_new_tokens": int(evaluation["max_new_tokens"]),
            "do_sample": False,
            "use_cache": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if "repetition_penalty" in evaluation:
            generation_kwargs["repetition_penalty"] = float(
                evaluation["repetition_penalty"]
            )
        if "no_repeat_ngram_size" in evaluation:
            generation_kwargs["no_repeat_ngram_size"] = int(
                evaluation["no_repeat_ngram_size"]
            )
        ids = self.model.language_model.generate(
            **multimodal,
            **generation_kwargs,
        )
        # Do not leak Qwen3.5's generation-batch-sized RoPE cache into training.
        self._reset_rope_deltas()
        # With inputs_embeds, Transformers returns generated tokens only for this model.
        return self.tokenizer.batch_decode(ids, skip_special_tokens=True)

    def on_validation_epoch_end(self) -> None:
        local = [item for batch in self.validation_outputs for item in zip(*batch)]
        gathered = (
            [None] * dist.get_world_size()
            if dist.is_available() and dist.is_initialized()
            else None
        )
        if gathered is not None:
            dist.all_gather_object(gathered, local)
            rows = [row for rank_rows in gathered for row in rank_rows]
        else:
            rows = local
        # DistributedSampler may pad validation with duplicate cases.
        rows = list({row[2]: row for row in rows}.values())
        predictions = [row[0] for row in rows]
        references = [row[1] for row in rows]
        overfit = bool(self.cfg["Data"].get("overfit_cases"))
        if overfit and all(row[3] is not None for row in rows):
            swapped_predictions = [row[3] for row in rows]
            prediction_by_case = {row[2]: row[0] for row in rows}
            source_predictions = [prediction_by_case[row[4]] for row in rows]
            metrics = compute_overfit_metrics(
                predictions, references, swapped_predictions, source_predictions
            )
        elif overfit:
            metrics = compute_overfit_metrics(predictions, references)
        else:
            metrics = compute_report_metrics(predictions, references)
        self.latest_valid_metrics = metrics
        for name, value in metrics.items():
            self.log(
                f"valid_{name}", value, on_step=False, on_epoch=True, sync_dist=True
            )
        if self.trainer.is_global_zero:
            train_loss = float(
                self.trainer.callback_metrics.get("train_loss", float("nan"))
            )
            valid_loss = float(
                self.trainer.callback_metrics.get("valid_loss", float("nan"))
            )
            print(
                f"epoch={self.current_epoch + 1} train_loss={train_loss:.6f}",
                flush=True,
            )
            print(
                f"epoch={self.current_epoch + 1} valid_loss={valid_loss:.6f} "
                f"BLEU-4={metrics['bleu4']:.6f} "
                f"METEOR={metrics['meteor']:.6f}",
                flush=True,
            )
            if overfit:
                self._save_overfit_result(rows, metrics)
                print(
                    f"overfit exact={metrics['exact_match']:.3f} "
                    f"similarity={metrics['character_similarity']:.3f} "
                    f"swap_change={metrics.get('image_swap_change_rate', float('nan')):.3f} "
                    f"swap_stays={metrics.get('swap_stays_case_similarity', float('nan')):.3f} "
                    f"swap_follows={metrics.get('swap_follows_image_similarity', float('nan')):.3f}",
                    flush=True,
                )
        self.validation_outputs.clear()
        self._train_loss_sum = 0.0
        self._train_samples = 0

    def _save_overfit_result(
        self, rows: list[tuple], metrics: dict[str, float]
    ) -> None:
        evaluation = self.cfg["Evaluation"]
        reproduced = (
            metrics["bleu4"] >= float(evaluation.get("overfit_bleu4_threshold", 0.90))
            and metrics["meteor"]
            >= float(evaluation.get("overfit_meteor_threshold", 0.95))
            and metrics["character_similarity"]
            >= float(evaluation.get("overfit_similarity_threshold", 0.95))
        )
        follows = metrics.get("swap_follows_image_similarity")
        stays = metrics.get("swap_stays_case_similarity")
        responsive = (
            metrics.get("image_swap_change_rate", 0.0)
            >= float(evaluation.get("overfit_swap_change_threshold", 0.75))
            and follows is not None
            and stays is not None
            and follows - stays >= float(evaluation.get("overfit_swap_margin", 0.10))
        )
        if not reproduced:
            status = "not_reproduced"
            interpretation = (
                "Check implementation, gradients, input connection, or optimization."
            )
        elif not responsive:
            status = "reproduced_but_image_insensitive"
            interpretation = (
                "Reports were reproduced, but generation did not follow swapped images."
            )
        else:
            status = "reproduced_and_image_sensitive"
            interpretation = (
                "Wiring passed; investigate full-data training settings and capacity."
            )
        epoch_result = {
            "epoch": self.current_epoch + 1,
            "status": status,
            "interpretation": interpretation,
            "metrics": metrics,
            "predictions": [
                {
                    "case_id": row[2],
                    "swap_source_case_id": row[4],
                    "reference": row[1],
                    "prediction": row[0],
                    "swapped_prediction": row[3],
                }
                for row in rows
            ],
        }
        text = json.dumps(epoch_result, indent=2, ensure_ascii=False) + "\n"
        output = Path(self.cfg["General"]["output_path"])
        (
            output / f"overfit_predictions_epoch_{self.current_epoch + 1:04d}.json"
        ).write_text(text, encoding="utf-8")
        (output / "overfit_result.json").write_text(text, encoding="utf-8")
        if reproduced and bool(evaluation.get("overfit_stop_when_reproduced", True)):
            self.trainer.should_stop = True

    def optimizer_parameter_groups(self) -> list[dict]:
        cfg = self.cfg["Optimizer"]
        fallback = float(cfg.get("learning_rate", 1e-5))
        groups = [
            {
                "name": "projector",
                "params": list(self.model.projector.parameters()),
                "lr": float(cfg.get("projector_learning_rate", fallback)),
            },
            {
                "name": "lora",
                "params": list(self._lora_parameters),
                "lr": float(cfg.get("lora_learning_rate", fallback)),
            },
        ]
        if self.model.structured_head is not None:
            groups.append(
                {
                    "name": "structured",
                    "params": list(self.model.structured_head.parameters()),
                    "lr": float(
                        cfg.get(
                            "structured_learning_rate",
                            cfg.get("projector_learning_rate", fallback),
                        )
                    ),
                }
            )
        return groups

    def configure_optimizers(self):
        cfg = self.cfg["Optimizer"]
        optimizer = torch.optim.AdamW(
            self.optimizer_parameter_groups(),
            weight_decay=float(cfg["weight_decay"]),
        )
        scheduler = build_lr_scheduler(
            optimizer,
            self.cfg["Scheduler"],
            total_steps=int(self.trainer.estimated_stepping_batches),
            max_epochs=int(self.trainer.max_epochs),
        )
        if self.trainer.is_global_zero:
            lr_scheduler = scheduler["scheduler"]
            group_lrs = ",".join(
                f"{group['name']}={float(group['lr']):.8g}"
                for group in optimizer.param_groups
            )
            details = f" peak_lrs={group_lrs}"
            if hasattr(lr_scheduler, "total_steps"):
                details += (
                    f" total_steps={lr_scheduler.total_steps}"
                    f" warmup_steps={lr_scheduler.warmup_steps}"
                    f" end_lr={lr_scheduler.eta_min:.8g}"
                )
            print(
                f"lr_scheduler={type(lr_scheduler).__name__}"
                f" interval={scheduler['interval']}{details}"
                f" projector_only_epochs={self.projector_only_epochs}",
                flush=True,
            )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def save_components(self, directory) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.model.language_model.save_pretrained(directory / "adapter")
        torch.save(self.model.projector.state_dict(), directory / "projector.pt")
        if self.model.structured_head is not None:
            torch.save(
                self.model.structured_head.state_dict(),
                directory / "structured_head.pt",
            )

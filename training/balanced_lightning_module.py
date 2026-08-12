from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

from balanced_metrics import clinical_metrics
from lightning_module import CBCTReportLightningModule
from metrics import compute_overfit_metrics, sequence_similarity


def weighted_causal_loss(logits, labels, token_weights, sample_weights):
    shifted_logits = logits[:, :-1].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    shifted_weights = token_weights[:, 1:].to(shifted_logits.device)
    valid = shifted_labels.ne(-100)
    safe_labels = shifted_labels.masked_fill(~valid, 0)
    losses = F.cross_entropy(
        shifted_logits.view(-1, shifted_logits.shape[-1]),
        safe_labels.view(-1),
        reduction="none",
    ).view_as(safe_labels)
    unweighted = losses[valid].mean() if valid.any() else losses.sum() * 0
    active_weights = shifted_weights * valid
    token_denominators = active_weights.sum(dim=1)
    per_sample = (losses * active_weights).sum(dim=1) / token_denominators.clamp_min(
        1e-8
    )
    active = token_denominators.gt(0)
    sample_weights = sample_weights.to(shifted_logits.device)
    weighted = (
        (per_sample[active] * sample_weights[active]).mean()
        if active.any()
        else losses.sum() * 0
    )
    return weighted, unweighted


def structured_finding_loss(logits, targets, positive_weight: float):
    target = targets.to(device=logits.device, dtype=torch.float32)
    prediction = logits.float()
    pos_weight = torch.full(
        (prediction.shape[-1],),
        float(positive_weight),
        device=prediction.device,
        dtype=prediction.dtype,
    )
    return F.binary_cross_entropy_with_logits(prediction, target, pos_weight=pos_weight)


def grounding_margin_loss(correct, shuffled, report, margin: float):
    correct_similarity = F.cosine_similarity(correct.float(), report.float(), dim=-1)
    shuffled_similarity = F.cosine_similarity(shuffled.float(), report.float(), dim=-1)
    return F.relu(float(margin) - correct_similarity + shuffled_similarity).mean()


def clinical_score_v1(metrics):
    """Legacy recall-heavy score retained only for cross-generation dashboards."""
    reward = (
        0.30 * float(metrics.get("tooth_f1_micro", 0.0))
        + 0.10 * float(metrics.get("tooth_f1_macro", 0.0))
        + 0.25 * float(metrics.get("case_specific_recall_micro", 0.0))
        + 0.15 * float(metrics.get("case_specific_recall_macro", 0.0))
        + 0.10 * float(metrics.get("side_accuracy", 0.0))
        + 0.10 * (1.0 - float(metrics.get("negation_contradiction_rate", 0.0)))
    )
    return reward * (1.0 - 0.5 * float(metrics.get("unsupported_finding_rate", 0.0)))


def clinical_checkpoint_score(metrics, evaluation):
    selection = evaluation.get("checkpoint_selection", {})
    positive = selection.get(
        "positive_weights",
        {
            "tooth_f1_micro": 0.35,
            "tooth_f1_macro": 0.15,
            "case_specific_f1_micro": 0.25,
            "side_accuracy": 0.10,
            "negation_consistency": 0.10,
            "image_shuffle_grounding_gap": 0.05,
        },
    )
    penalties = selection.get(
        "penalty_weights",
        {
            "unsupported_finding_rate": 0.50,
            "truncated_generation_rate": 0.20,
            "repetition_case_rate": 0.20,
            "invalid_fdi_case_rate": 0.20,
        },
    )
    values = dict(metrics)
    values["negation_consistency"] = 1.0 - float(
        metrics.get("negation_contradiction_rate", 0.0)
    )
    denominator = sum(float(weight) for weight in positive.values())
    reward = sum(
        float(weight) * float(values.get(name, 0.0))
        for name, weight in positive.items()
    )
    reward = reward / denominator if denominator else 0.0
    penalty = sum(
        float(weight) * float(metrics.get(name, 0.0))
        for name, weight in penalties.items()
    )
    empty_penalty = float(selection.get("empty_generation_penalty", 0.50)) * float(
        metrics.get("empty_generation_rate", 0.0)
    )
    return reward - penalty - empty_penalty


def clinical_checkpoint_eligibility(metrics, evaluation):
    selection = evaluation.get("checkpoint_selection", {})
    gates = selection.get("hard_gates", {})
    failures = []
    maximums = (
        ("unsupported_finding_rate", "max_unsupported_finding_rate", 0.60),
        ("truncated_generation_rate", "max_truncated_generation_rate", 0.05),
        ("invalid_fdi_case_rate", "max_invalid_fdi_case_rate", 0.0),
    )
    for metric, setting, default in maximums:
        threshold = float(gates.get(setting, default))
        if float(metrics.get(metric, 0.0)) > threshold:
            failures.append(f"{metric}>{threshold:g}")
    minimum = float(gates.get("min_tooth_precision_micro", 0.15))
    if float(metrics.get("tooth_precision_micro", 0.0)) < minimum:
        failures.append(f"tooth_precision_micro<{minimum:g}")
    return not failures, failures


def image_shuffle_metrics(predictions, shuffled_predictions, source_predictions):
    size = len(predictions)
    if not size:
        return {}
    return {
        "image_shuffle_change_rate": sum(
            normal.strip() != shuffled.strip()
            for normal, shuffled in zip(predictions, shuffled_predictions)
        )
        / size,
        "image_shuffle_stays_case_similarity": sum(
            sequence_similarity(normal, shuffled)
            for normal, shuffled in zip(predictions, shuffled_predictions)
        )
        / size,
        "image_shuffle_follows_source_similarity": sum(
            sequence_similarity(source, shuffled)
            for source, shuffled in zip(source_predictions, shuffled_predictions)
        )
        / size,
    }


class BalancedCBCTReportLightningModule(CBCTReportLightningModule):
    def on_train_epoch_start(self):
        super().on_train_epoch_start()
        self.trainer.datamodule.set_epoch(self.current_epoch)

    def _report_embedding(self, batch):
        embeddings = self.model.language_model.get_input_embeddings()(
            batch["input_ids"]
        )
        mask = batch["labels"].ne(-100) & batch["attention_mask"].bool()
        weights = mask.to(embeddings.dtype).unsqueeze(-1)
        pooled = (embeddings * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return pooled.detach()

    def _pooled_image_embedding(self, batch, prefix=""):
        arguments = self._image_arguments(batch, prefix)
        embeddings = self.model.project_image_tokens(
            arguments["image_features"],
            arguments["image_token_type_ids"],
            arguments["region_label_ids"],
            arguments["region_metadata"],
        )
        mask = arguments["image_attention_mask"].to(embeddings.dtype).unsqueeze(-1)
        return (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

    def training_step(self, batch, _batch_idx):
        output = self(batch)
        prefix = batch["image_features"].shape[1]
        labels = torch.cat(
            (
                torch.full(
                    (batch["labels"].shape[0], prefix),
                    -100,
                    dtype=batch["labels"].dtype,
                    device=batch["labels"].device,
                ),
                batch["labels"],
            ),
            dim=1,
        )
        token_weights = torch.cat(
            (
                torch.zeros(
                    (batch["loss_weights"].shape[0], prefix),
                    dtype=batch["loss_weights"].dtype,
                    device=batch["loss_weights"].device,
                ),
                batch["loss_weights"],
            ),
            dim=1,
        )
        weighted, unweighted = weighted_causal_loss(
            output.logits, labels, token_weights, batch["sample_weights"]
        )
        total = weighted
        size = len(batch["case_ids"])

        structured_cfg = self.cfg.get("StructuredFindings", {})
        if self.model.structured_head is not None and "tooth_targets" in batch:
            predictions = self.model.structured_predictions(
                **self._image_arguments(batch)
            )
            positive_weight = float(structured_cfg.get("positive_weight", 3.0))
            tooth_loss = structured_finding_loss(
                predictions["tooth"], batch["tooth_targets"], positive_weight
            )
            case_loss = structured_finding_loss(
                predictions["case"], batch["case_targets"], positive_weight
            )
            auxiliary = (
                float(structured_cfg.get("tooth_weight", 1.0)) * tooth_loss
                + float(structured_cfg.get("case_weight", 1.0)) * case_loss
            )
            total = total + float(structured_cfg.get("loss_weight", 0.3)) * auxiliary
            self.log(
                "train_structured_loss",
                auxiliary,
                on_step=False,
                on_epoch=True,
                batch_size=size,
                sync_dist=True,
            )
            self.log(
                "train_tooth_aux_loss",
                tooth_loss,
                on_step=False,
                on_epoch=True,
                batch_size=size,
                sync_dist=True,
            )
            self.log(
                "train_case_aux_loss",
                case_loss,
                on_step=False,
                on_epoch=True,
                batch_size=size,
                sync_dist=True,
            )

        grounding_cfg = self.cfg.get("Grounding", {})
        if (
            bool(grounding_cfg.get("enabled", False))
            and "shuffled_image_features" in batch
        ):
            grounding = grounding_margin_loss(
                self._pooled_image_embedding(batch),
                self._pooled_image_embedding(batch, "shuffled_"),
                self._report_embedding(batch),
                float(grounding_cfg.get("margin", 0.1)),
            )
            total = total + float(grounding_cfg.get("loss_weight", 0.1)) * grounding
            self.log(
                "train_grounding_loss",
                grounding,
                on_step=False,
                on_epoch=True,
                batch_size=size,
                sync_dist=True,
            )

        self.log(
            "train_loss",
            total,
            on_step=False,
            on_epoch=True,
            batch_size=size,
            sync_dist=True,
        )
        self.log(
            "train_language_loss",
            weighted,
            on_step=False,
            on_epoch=True,
            batch_size=size,
            sync_dist=True,
        )
        self.log(
            "train_unweighted_loss",
            unweighted,
            on_step=False,
            on_epoch=True,
            batch_size=size,
            sync_dist=True,
        )
        return total

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
        shuffled = [None] * size
        shuffle_sources = [None] * size
        if "shuffled_image_features" in batch:
            shuffled = self.generate(batch, "shuffled_")
            shuffle_sources = batch["shuffle_source_case_ids"]
        self.validation_outputs.append(
            (
                generated,
                batch["references"],
                batch["case_ids"],
                shuffled,
                shuffle_sources,
            )
        )

    def on_validation_epoch_end(self):
        local = [item for batch in self.validation_outputs for item in zip(*batch)]
        if dist.is_available() and dist.is_initialized():
            gathered = [None] * dist.get_world_size()
            dist.all_gather_object(gathered, local)
            rows = [row for rank_rows in gathered for row in rank_rows]
        else:
            rows = local
        rows = list({row[2]: row for row in rows}.values())
        predictions = [row[0] for row in rows]
        references = [row[1] for row in rows]
        case_ids = [row[2] for row in rows]
        metrics, details, case_details = clinical_metrics(
            predictions, references, case_ids
        )
        overfit = bool(self.cfg["Data"].get("overfit_cases"))
        if overfit:
            metrics.update(
                compute_overfit_metrics(predictions, [item[0] for item in references])
            )
        shuffle_rows = bool(rows) and all(
            row[3] is not None and row[4] is not None for row in rows
        )
        if shuffle_rows:
            shuffled_predictions = [row[3] for row in rows]
            prediction_by_case = {row[2]: row[0] for row in rows}
            source_predictions = [prediction_by_case[row[4]] for row in rows]
            metrics.update(
                image_shuffle_metrics(
                    predictions, shuffled_predictions, source_predictions
                )
            )
            shuffled_metrics, _, _ = clinical_metrics(
                shuffled_predictions, references, case_ids
            )
            base_score = clinical_checkpoint_score(metrics, self.cfg["Evaluation"])
            shuffled_score = clinical_checkpoint_score(
                shuffled_metrics, self.cfg["Evaluation"]
            )
            for name in (
                "tooth_f1_micro",
                "case_specific_recall_micro",
                "unsupported_finding_rate",
                "negation_contradiction_rate",
                "unique_rate",
            ):
                metrics[f"image_shuffle_{name}"] = shuffled_metrics[name]
            grounding_gap = base_score - shuffled_score
            metrics["image_shuffle_clinical_score_drop"] = grounding_gap
            metrics["image_shuffle_grounding_gap"] = max(0.0, grounding_gap)
            for detail, row in zip(case_details, rows):
                detail.update(shuffled_prediction=row[3], shuffle_source_case_id=row[4])

        metrics["clinical_score_v1"] = clinical_score_v1(metrics)
        metrics["clinical_score_v2"] = clinical_checkpoint_score(
            metrics, self.cfg["Evaluation"]
        )
        eligible, failures = clinical_checkpoint_eligibility(
            metrics, self.cfg["Evaluation"]
        )
        metrics["checkpoint_hard_gate_pass"] = float(eligible)
        metrics["checkpoint_hard_gate_fail_count"] = float(len(failures))
        details["checkpoint_selection"] = {
            "metric_schema_version": 2,
            "eligible": eligible,
            "failed_gates": failures,
        }
        self.latest_valid_metrics = metrics
        for name, value in metrics.items():
            self.log(
                f"valid_{name}",
                float(value),
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        if self.trainer.is_global_zero:
            self._save_clinical_results(metrics, details, case_details)
            print(
                f"epoch={self.current_epoch + 1} clinical-v1={metrics['clinical_score_v1']:.6f} "
                f"clinical-v2={metrics['clinical_score_v2']:.6f} "
                f"tooth-P/R={metrics['tooth_precision_micro']:.4f}/{metrics['tooth_recall_micro']:.4f} "
                f"unsupported={int(metrics['unsupported_count'])} unique={metrics['unique_rate']:.4f}",
                flush=True,
            )
        self.validation_outputs.clear()

    def _save_clinical_results(self, metrics, details, cases):
        output = Path(self.cfg["General"]["output_path"])
        epoch = self.current_epoch + 1
        summary = {
            "epoch": epoch,
            "metric_schema_version": 2,
            "metrics": metrics,
            "details": details,
        }
        (output / f"clinical_metrics_epoch_{epoch:04d}.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        with (output / f"validation_predictions_epoch_{epoch:04d}.jsonl").open(
            "w", encoding="utf-8"
        ) as handle:
            for row in cases:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

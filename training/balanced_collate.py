from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from clinical import token_clause_weights
from collate import DEFAULT_SYSTEM_PROMPT, DEFAULT_USER_PROMPT, build_prompt


def _pad_image_tokens(
    samples: list[dict[str, Any]], prefix: str = ""
) -> dict[str, torch.Tensor]:
    feature_key = f"{prefix}image_features"
    maximum = max(sample[feature_key].shape[0] for sample in samples)
    feature_dim = samples[0][feature_key].shape[1]
    batch = len(samples)
    features = torch.zeros(batch, maximum, feature_dim, dtype=torch.float32)
    attention = torch.zeros(batch, maximum, dtype=torch.long)
    token_types = torch.zeros(batch, maximum, dtype=torch.long)
    region_labels = torch.zeros(batch, maximum, dtype=torch.long)
    metadata = torch.zeros(batch, maximum, 10, dtype=torch.float32)
    for row, sample in enumerate(samples):
        values = sample[feature_key].float()
        size = values.shape[0]
        features[row, :size] = values
        attention[row, :size] = sample.get(
            f"{prefix}image_attention_mask", torch.ones(size, dtype=torch.long)
        )
        token_types[row, :size] = sample.get(
            f"{prefix}image_token_type_ids", torch.zeros(size, dtype=torch.long)
        )
        region_labels[row, :size] = sample.get(
            f"{prefix}region_label_ids", torch.zeros(size, dtype=torch.long)
        )
        metadata[row, :size] = sample.get(
            f"{prefix}region_metadata", torch.zeros(size, 10, dtype=torch.float32)
        )
    return {
        feature_key: features,
        f"{prefix}image_attention_mask": attention,
        f"{prefix}image_token_type_ids": token_types,
        f"{prefix}region_label_ids": region_labels,
        f"{prefix}region_metadata": metadata,
    }


@dataclass
class BalancedReportCollator:
    tokenizer: Any
    max_report_tokens: int
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    user_prompt: str = DEFAULT_USER_PROMPT
    coverage_weight: float = 0.5
    finding_weight: float = 1.5
    clause_weighting: bool = True

    def __post_init__(self):
        self.prompt_ids = build_prompt(
            self.tokenizer, self.system_prompt, self.user_prompt
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.clause_weighting and getattr(self.tokenizer, "is_fast", True) is False:
            raise ValueError(
                "Clause weighting requires a fast tokenizer with offset mapping"
            )

    def _encode(self, text: str) -> tuple[list[int], list[float]]:
        if self.clause_weighting:
            encoded = self.tokenizer(
                text,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_report_tokens,
                return_offsets_mapping=True,
            )
            ids = list(encoded["input_ids"])
            offsets = [tuple(value) for value in encoded["offset_mapping"]]
            weights = token_clause_weights(
                text, offsets, self.coverage_weight, self.finding_weight
            )
        else:
            ids = self.tokenizer.encode(
                text,
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_report_tokens,
            )
            weights = [1.0] * len(ids)
        return ids, weights

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        prompt, eos = self.prompt_ids, self.tokenizer.eos_token_id
        encoded, labels, weights = [], [], []
        for sample in samples:
            answer, answer_weights = self._encode(sample["report"])
            add_eos = eos is not None and (not answer or answer[-1] != eos)
            ids = prompt + answer + ([eos] if add_eos else [])
            encoded.append(ids)
            labels.append([-100] * len(prompt) + answer + ([eos] if add_eos else []))
            weights.append(
                [0.0] * len(prompt) + answer_weights + ([1.0] if add_eos else [])
            )
        width, batch = max(map(len, encoded)), len(samples)
        input_ids = torch.full(
            (batch, width), self.tokenizer.pad_token_id, dtype=torch.long
        )
        attention = torch.zeros((batch, width), dtype=torch.long)
        label_tensor = torch.full((batch, width), -100, dtype=torch.long)
        loss_weights = torch.zeros((batch, width), dtype=torch.float32)
        for row, (ids, target, token_weights) in enumerate(
            zip(encoded, labels, weights)
        ):
            size = len(ids)
            input_ids[row, :size] = torch.tensor(ids)
            attention[row, :size] = 1
            label_tensor[row, :size] = torch.tensor(target)
            loss_weights[row, :size] = torch.tensor(token_weights)
        batch_dict = {
            "case_ids": [sample["case_id"] for sample in samples],
            **_pad_image_tokens(samples),
            "input_ids": input_ids,
            "attention_mask": attention,
            "labels": label_tensor,
            "loss_weights": loss_weights,
            "sample_weights": torch.tensor(
                [sample.get("sample_weight", 1.0) for sample in samples],
                dtype=torch.float32,
            ),
            "prompt_input_ids": torch.tensor(prompt).expand(batch, -1).clone(),
            "references": [
                sample.get("references", [sample["report"]]) for sample in samples
            ],
        }
        if all("tooth_targets" in sample for sample in samples):
            batch_dict["tooth_targets"] = torch.stack(
                [sample["tooth_targets"] for sample in samples]
            )
            batch_dict["case_targets"] = torch.stack(
                [sample["case_targets"] for sample in samples]
            )
        if all("shuffled_image_features" in sample for sample in samples):
            batch_dict.update(_pad_image_tokens(samples, "shuffled_"))
            batch_dict["shuffle_source_case_ids"] = [
                sample["shuffle_source_case_id"] for sample in samples
            ]
        return batch_dict

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


DEFAULT_SYSTEM_PROMPT = (
    "You are an expert oral and maxillofacial radiologist. Generate a concise, factual English "
    "diagnostic report from the supplied CBCT feature tokens. Do not invent findings that are not "
    "supported by the input. Explicitly state uncertainty or non-evaluable anatomy."
)

DEFAULT_USER_PROMPT = """Review the CBCT in this exact clinical order:
1. scan coverage;
2. findings by jaw;
3. tooth-level findings using FDI notation;
4. relationship to the mandibular canals;
5. maxillary sinuses and temporomandibular joints;
6. lesions;
7. non-evaluable findings and uncertainty.
Then produce one coherent natural-language diagnostic report. Return only the report."""


def build_prompt(tokenizer, system_prompt: str, user_prompt: str) -> list[int]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    kwargs = dict(tokenize=True, add_generation_prompt=True, return_dict=False)
    try:
        ids = tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:  # Allows tokenizers whose templates predate enable_thinking.
        ids = tokenizer.apply_chat_template(messages, **kwargs)
    return list(ids)


@dataclass
class ReportCollator:
    tokenizer: Any
    max_report_tokens: int
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    user_prompt: str = DEFAULT_USER_PROMPT

    def __post_init__(self) -> None:
        self.prompt_ids = build_prompt(
            self.tokenizer, self.system_prompt, self.user_prompt
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        prompt = self.prompt_ids
        eos = self.tokenizer.eos_token_id
        encoded: list[list[int]] = []
        labels: list[list[int]] = []
        for sample in samples:
            answer = self.tokenizer.encode(
                sample["report"],
                add_special_tokens=False,
                truncation=True,
                max_length=self.max_report_tokens,
            )
            ids = (
                prompt
                + answer
                + (
                    [eos]
                    if eos is not None and (not answer or answer[-1] != eos)
                    else []
                )
            )
            encoded.append(ids)
            labels.append([-100] * len(prompt) + ids[len(prompt) :])

        width = max(map(len, encoded))
        input_ids = torch.full(
            (len(samples), width), self.tokenizer.pad_token_id, dtype=torch.long
        )
        attention_mask = torch.zeros((len(samples), width), dtype=torch.long)
        label_tensor = torch.full((len(samples), width), -100, dtype=torch.long)
        for row, (ids, target) in enumerate(zip(encoded, labels)):
            size = len(ids)
            input_ids[row, :size] = torch.tensor(ids)
            attention_mask[row, :size] = 1
            label_tensor[row, :size] = torch.tensor(target)
        return {
            "case_ids": [sample["case_id"] for sample in samples],
            "image_features": torch.stack(
                [sample["image_features"] for sample in samples]
            ),
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": label_tensor,
            "prompt_input_ids": torch.tensor(prompt, dtype=torch.long)
            .expand(len(samples), -1)
            .clone(),
            "references": [sample["report"] for sample in samples],
        }

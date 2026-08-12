import torch

from collate import ReportCollator


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    eos_token = "<eos>"
    pad_token = "<pad>"

    def apply_chat_template(self, _messages, **_kwargs):
        return [10, 11, 12]

    def encode(self, text, **_kwargs):
        return list(range(20, 20 + len(text.split())))


def test_collator_masks_prompt_and_padding():
    collate = ReportCollator(FakeTokenizer(), max_report_tokens=10)
    batch = collate(
        [
            {"case_id": "a", "image_features": torch.ones(2, 4), "report": "one two"},
            {"case_id": "b", "image_features": torch.zeros(2, 4), "report": "one"},
        ]
    )
    assert batch["image_features"].shape == (2, 2, 4)
    assert batch["labels"][0, :3].tolist() == [-100, -100, -100]
    assert batch["labels"][1, -1].item() == -100
    assert batch["attention_mask"][1, -1].item() == 0

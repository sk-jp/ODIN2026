from pathlib import Path
from types import SimpleNamespace

import torch
from lightning.pytorch import LightningModule
from torch import nn

from balanced_collate import _pad_image_tokens
from balanced_dataset import ManifestDataset, ManifestRecord
from balanced_lightning_module import grounding_margin_loss, structured_finding_loss
from lightning_module import CBCTReportLightningModule
from model import CBCTReportModel, FeatureProjector, StructuredFindingHead
from structured import CASE_CONCEPT_INDEX, TOOTH_CONCEPT_INDEX


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(20, 8)

    def get_input_embeddings(self):
        return self.embedding


class ModelParts(nn.Module):
    def __init__(self):
        super().__init__()
        self.projector = nn.Linear(4, 8)
        self.language_model = nn.Linear(8, 8)
        self.structured_head = nn.Linear(8, 3)


def test_region_tokens_and_structured_targets(tmp_path):
    feature = tmp_path / "features.pt"
    report = tmp_path / "report.txt"
    report.write_text(
        "Tooth 11 is missing. An osteolytic lesion is present.", encoding="utf-8"
    )
    torch.save(
        {
            "spatial": torch.ones(2, 2, 2, 2),
            "regions": {
                "labels": torch.tensor([11, 3]),
                "names": [
                    "Upper Right Central Incisor",
                    "Left Inferior Alveolar Canal",
                ],
                "mean": torch.ones(2, 2),
                "max": torch.full((2, 2), 2.0),
                "voxel_count": torch.tensor([100, 200]),
                "centroid_world_xyz": torch.tensor([[1.0, 2.0, 3.0], [4.0, 6.0, 8.0]]),
                "bbox_xyz": torch.tensor([[0, 0, 0, 2, 2, 2], [2, 2, 2, 4, 4, 4]]),
            },
        },
        feature,
    )
    record = ManifestRecord("A", feature, (report,), (), "A", "A", 1.0)
    sample = ManifestDataset(
        [record], (1, 1, 1), 2, token_mode="global_region", max_region_tokens=8
    )[0]
    assert sample["image_features"].shape == (3, 4)
    assert sample["image_token_type_ids"].tolist() == [0, 1, 1]
    assert sample["region_label_ids"].tolist() == [0, 11, 3]
    assert sample["region_metadata"].shape == (3, 10)
    assert sample["tooth_targets"][0, TOOTH_CONCEPT_INDEX["missing"]] == 1
    assert sample["case_targets"][CASE_CONCEPT_INDEX["lesion"]] == 1


def test_variable_length_image_tokens_are_padded_with_masks():
    batch = _pad_image_tokens(
        [
            {"image_features": torch.ones(2, 4)},
            {"image_features": torch.ones(3, 4)},
        ]
    )
    assert batch["image_features"].shape == (2, 3, 4)
    assert batch["image_attention_mask"].tolist() == [[1, 1, 0], [1, 1, 1]]
    assert torch.all(batch["image_features"][0, 2] == 0)


def test_region_aware_model_and_structured_head_shapes():
    model = CBCTReportModel(
        TinyLM(), FeatureProjector(4, 6, 8, 0.0), StructuredFindingHead(8, 0.0)
    )
    image = torch.randn(2, 3, 4)
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    token_types = torch.tensor([[0, 1, 0], [0, 1, 1]])
    labels = torch.tensor([[0, 11, 0], [0, 48, 3]])
    metadata = torch.zeros(2, 3, 10)
    result = model.structured_predictions(image, mask, token_types, labels, metadata)
    assert result["tooth"].shape == (2, 32, 6)
    assert result["case"].shape == (2, 7)
    multimodal = model.multimodal_inputs(
        image,
        torch.ones(2, 2, dtype=torch.long),
        torch.ones(2, 2, dtype=torch.long),
        image_attention_mask=mask,
        image_token_type_ids=token_types,
        region_label_ids=labels,
        region_metadata=metadata,
    )
    assert multimodal["attention_mask"].tolist() == [[1, 1, 0, 1, 1], [1, 1, 1, 1, 1]]


def test_auxiliary_and_grounding_losses_have_expected_ordering():
    logits = torch.tensor([[[-3.0, 3.0]]])
    targets = torch.tensor([[[0.0, 1.0]]])
    good = structured_finding_loss(logits, targets, 3.0)
    bad = structured_finding_loss(-logits, targets, 3.0)
    assert good < bad
    report = torch.tensor([[1.0, 0.0]])
    correct = torch.tensor([[1.0, 0.0]])
    shuffled = torch.tensor([[0.0, 1.0]])
    assert grounding_margin_loss(correct, shuffled, report, 0.1) == 0
    assert grounding_margin_loss(shuffled, correct, report, 0.1) > 0


def test_optimizer_groups_and_staged_lora_gradient_masking():
    module = CBCTReportLightningModule.__new__(CBCTReportLightningModule)
    LightningModule.__init__(module)
    module.cfg = {
        "Optimizer": {
            "learning_rate": 1e-5,
            "projector_learning_rate": 1e-4,
            "lora_learning_rate": 1e-5,
            "structured_learning_rate": 2e-4,
        }
    }
    module.model = ModelParts()
    module._lora_parameters = tuple(module.model.language_model.parameters())
    module.projector_only_epochs = 2
    groups = module.optimizer_parameter_groups()
    assert [group["name"] for group in groups] == ["projector", "lora", "structured"]
    assert [group["lr"] for group in groups] == [1e-4, 1e-5, 2e-4]

    for parameter in module._lora_parameters:
        parameter.grad = torch.ones_like(parameter)
    module._trainer = SimpleNamespace(current_epoch=0)
    module.on_after_backward()
    assert all(parameter.grad is None for parameter in module._lora_parameters)
    for parameter in module._lora_parameters:
        parameter.grad = torch.ones_like(parameter)
    module._trainer = SimpleNamespace(current_epoch=2)
    module.on_after_backward()
    assert all(parameter.grad is not None for parameter in module._lora_parameters)

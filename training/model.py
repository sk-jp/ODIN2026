from __future__ import annotations

import torch
from torch import nn

from structured import CASE_CONCEPTS, FDI_TEETH, TOOTH_CONCEPTS


class FeatureProjector(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float,
        *,
        max_region_label: int = 64,
        region_metadata_dim: int = 10,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
        self.token_type_embedding = nn.Embedding(2, output_dim)
        self.region_label_embedding = nn.Embedding(
            max_region_label + 1, output_dim, padding_idx=0
        )
        self.region_metadata_projection = nn.Linear(
            region_metadata_dim, output_dim, bias=False
        )
        self.conditioning_norm = nn.LayerNorm(output_dim)
        self.max_region_label = int(max_region_label)
        self.region_metadata_dim = int(region_metadata_dim)

    def forward(
        self,
        features: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        region_label_ids: torch.Tensor | None = None,
        region_metadata: torch.Tensor | None = None,
    ) -> torch.Tensor:
        projected = self.net(features)
        batch, tokens = features.shape[:2]
        device = features.device
        if token_type_ids is None:
            token_type_ids = torch.zeros(batch, tokens, dtype=torch.long, device=device)
        if region_label_ids is None:
            region_label_ids = torch.zeros(
                batch, tokens, dtype=torch.long, device=device
            )
        if region_metadata is None:
            region_metadata = torch.zeros(
                batch,
                tokens,
                self.region_metadata_dim,
                dtype=features.dtype,
                device=device,
            )
        conditioned = (
            projected
            + self.token_type_embedding(token_type_ids.clamp(0, 1))
            + self.region_label_embedding(
                region_label_ids.clamp(0, self.max_region_label)
            )
            + self.region_metadata_projection(region_metadata.to(projected.dtype))
        )
        return self.conditioning_norm(conditioned)


class StructuredFindingHead(nn.Module):
    def __init__(self, hidden_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.register_buffer(
            "tooth_label_ids",
            torch.tensor([int(value) for value in FDI_TEETH]),
            persistent=False,
        )
        self.tooth_classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, len(TOOTH_CONCEPTS)),
        )
        self.case_classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, len(CASE_CONCEPTS)),
        )

    @staticmethod
    def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.to(values.dtype).unsqueeze(-1)
        return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_attention_mask: torch.Tensor,
        region_label_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        pooled = self.masked_mean(image_embeddings, image_attention_mask)
        tooth_masks = (
            region_label_ids.unsqueeze(-1)
            == self.tooth_label_ids.to(region_label_ids.device)
        ) & image_attention_mask.bool().unsqueeze(-1)
        weights = tooth_masks.to(image_embeddings.dtype)
        tooth_embeddings = torch.einsum("bth,btk->bkh", image_embeddings, weights)
        denominators = weights.sum(dim=1).unsqueeze(-1)
        tooth_embeddings = tooth_embeddings / denominators.clamp_min(1.0)
        tooth_embeddings = torch.where(
            denominators.gt(0),
            tooth_embeddings,
            pooled.unsqueeze(1).expand_as(tooth_embeddings),
        )
        return {
            "tooth": self.tooth_classifier(tooth_embeddings),
            "case": self.case_classifier(pooled),
            "pooled": pooled,
        }


class CBCTReportModel(nn.Module):
    def __init__(
        self,
        language_model: nn.Module,
        projector: FeatureProjector,
        structured_head: StructuredFindingHead | None = None,
    ) -> None:
        super().__init__()
        self.language_model = language_model
        self.projector = projector
        self.structured_head = structured_head

    def project_image_tokens(
        self,
        image_features: torch.Tensor,
        image_token_type_ids: torch.Tensor | None = None,
        region_label_ids: torch.Tensor | None = None,
        region_metadata: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dtype = self.projector.net[1].weight.dtype
        return self.projector(
            image_features.to(dtype=dtype),
            image_token_type_ids,
            region_label_ids,
            region_metadata,
        )

    def multimodal_inputs(
        self,
        image_features: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        image_attention_mask: torch.Tensor | None = None,
        image_token_type_ids: torch.Tensor | None = None,
        region_label_ids: torch.Tensor | None = None,
        region_metadata: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        embedding_layer = self.language_model.get_input_embeddings()
        text_embeddings = embedding_layer(input_ids)
        image_embeddings = self.project_image_tokens(
            image_features, image_token_type_ids, region_label_ids, region_metadata
        ).to(device=text_embeddings.device, dtype=text_embeddings.dtype)
        inputs_embeds = torch.cat((image_embeddings, text_embeddings), dim=1)
        if image_attention_mask is None:
            image_attention_mask = torch.ones(
                attention_mask.shape[0],
                image_embeddings.shape[1],
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
        else:
            image_attention_mask = image_attention_mask.to(
                device=attention_mask.device, dtype=attention_mask.dtype
            )
        result = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": torch.cat((image_attention_mask, attention_mask), dim=1),
        }
        if labels is not None:
            ignored = torch.full(
                (labels.shape[0], image_embeddings.shape[1]),
                -100,
                dtype=labels.dtype,
                device=labels.device,
            )
            result["labels"] = torch.cat((ignored, labels), dim=1)
        return result

    def structured_predictions(
        self,
        image_features: torch.Tensor,
        image_attention_mask: torch.Tensor,
        image_token_type_ids: torch.Tensor,
        region_label_ids: torch.Tensor,
        region_metadata: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if self.structured_head is None:
            raise RuntimeError("Structured finding head is disabled")
        embeddings = self.project_image_tokens(
            image_features, image_token_type_ids, region_label_ids, region_metadata
        )
        return self.structured_head(embeddings, image_attention_mask, region_label_ids)

    def forward(
        self,
        image_features,
        input_ids,
        attention_mask,
        labels=None,
        image_attention_mask=None,
        image_token_type_ids=None,
        region_label_ids=None,
        region_metadata=None,
    ):
        return self.language_model(
            **self.multimodal_inputs(
                image_features,
                input_ids,
                attention_mask,
                labels,
                image_attention_mask,
                image_token_type_ids,
                region_label_ids,
                region_metadata,
            )
        )

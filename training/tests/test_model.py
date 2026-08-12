import torch
from torch import nn

from model import CBCTReportModel, FeatureProjector


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(20, 8)

    def get_input_embeddings(self):
        return self.embedding


def test_multimodal_sequence_and_label_mask():
    model = CBCTReportModel(TinyLM(), FeatureProjector(4, 6, 8, 0.0))
    result = model.multimodal_inputs(
        torch.randn(2, 3, 4),
        torch.ones(2, 5, dtype=torch.long),
        torch.ones(2, 5, dtype=torch.long),
        torch.ones(2, 5, dtype=torch.long),
    )
    assert result["inputs_embeds"].shape == (2, 8, 8)
    assert result["attention_mask"].shape == (2, 8)
    assert torch.all(result["labels"][:, :3] == -100)

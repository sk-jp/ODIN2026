import json
from pathlib import Path

import pytest
import torch

import inference


def test_interface_key_requires_inputs_json(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(inference, "INPUT_PATH", tmp_path)
    with pytest.raises(FileNotFoundError):
        inference.get_interface_key()

    (tmp_path / "inputs.json").write_text(
        json.dumps([{"socket": {"slug": "cbct-image"}}]), encoding="utf-8"
    )
    assert inference.get_interface_key() == ("cbct-image",)


def test_load_cbct(tmp_path: Path, monkeypatch):
    image_dir = tmp_path / "images" / "cbct"
    image_dir.mkdir(parents=True)
    image = inference.sitk.Image([2, 3, 4], inference.sitk.sitkInt16)
    inference.sitk.WriteImage(image, str(image_dir / "case.mha"))
    monkeypatch.setattr(inference, "INPUT_PATH", tmp_path)

    path, loaded = inference.load_cbct()

    assert path.name == "case.mha"
    assert loaded.GetSize() == (2, 3, 4)


def test_make_image_tokens_matches_training_shape(tmp_path: Path):
    feature_path = tmp_path / "features.pt"
    payload = {
        "spatial": torch.randn(320, 2, 3, 4, dtype=torch.float16),
        "regions": {
            "labels": torch.tensor([11, 48]),
            "mean": torch.randn(2, 320, dtype=torch.float16),
            "max": torch.randn(2, 320, dtype=torch.float16),
            "voxel_count": torch.tensor([10, 20]),
            "centroid_world_xyz": torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]),
            "bbox_xyz": torch.tensor([[0, 0, 0, 2, 2, 2], [2, 2, 2, 5, 5, 5]]),
        },
    }
    torch.save(payload, feature_path)

    tokens = inference.make_image_tokens(feature_path)

    assert tokens["features"].shape == (1, 66, 640)
    assert tokens["attention_mask"].shape == (1, 66)
    assert tokens["token_type_ids"][0, :64].eq(0).all()
    assert tokens["token_type_ids"][0, 64:].eq(1).all()
    assert tokens["region_label_ids"][0, 64:].tolist() == [11, 48]
    assert torch.isfinite(tokens["region_metadata"]).all()


def test_feature_projector_loads_rev3_weights():
    state_path = Path(__file__).parents[1] / "model" / "report" / "projector.pt"
    if not state_path.is_file():
        pytest.skip("Model weights are not included in the source repository")
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    projector = inference.FeatureProjector(
        input_dim=state["net.1.weight"].shape[1],
        hidden_dim=state["net.1.weight"].shape[0],
        output_dim=state["net.4.weight"].shape[0],
        max_region_label=state["region_label_embedding.weight"].shape[0] - 1,
        region_metadata_dim=state["region_metadata_projection.weight"].shape[1],
    )
    projector.load_state_dict(state, strict=True)

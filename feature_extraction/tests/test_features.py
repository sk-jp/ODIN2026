import numpy as np
import torch

from toothfairy2_feature_extraction.features import (
    SpatialFeatures,
    _aligned_starts,
    global_pool,
    label_to_feature_grid,
    region_pool,
)
from toothfairy2_feature_extraction.pipeline import (
    build_feature_payload,
    region_geometry,
)


def fake_spatial(tensor):
    return SpatialFeatures(
        tensor=tensor,
        stride_zyx=(2, 2, 2),
        spacing_zyx=(0.6, 0.6, 0.6),
        affine=torch.eye(4),
        preprocessed_shape=(4, 4, 4),
        properties={"bbox_used_for_cropping": [[0, 4], [0, 4], [0, 4]]},
    )


def test_aligned_starts_cover_volume_on_stride():
    starts, padded = _aligned_starts(13, 8, 4)
    assert starts == [0, 4, 8]
    assert padded == 16
    assert all(value % 4 == 0 for value in starts)


def test_pooling_and_label_grid():
    features = torch.arange(16, dtype=torch.float16).reshape(2, 2, 2, 2)
    labels = np.zeros((4, 4, 4), dtype=np.uint8)
    labels[:2] = 1
    spatial = fake_spatial(features)
    grid = label_to_feature_grid(labels, spatial)
    assert grid.shape == (2, 2, 2)
    mean, maximum = global_pool(features)
    torch.testing.assert_close(mean, torch.tensor([3.5, 11.5], dtype=torch.float16))
    torch.testing.assert_close(maximum, torch.tensor([7, 15], dtype=torch.float16))
    region_mean, region_max = region_pool(features, grid, torch.tensor([1]))
    assert region_mean.shape == (1, 2)
    assert region_max.shape == (1, 2)


def test_geometry_and_payload_are_weights_only_loadable(tmp_path):
    segmentation = np.zeros((4, 4, 4), dtype=np.uint8)
    segmentation[1:3, 1:3, 1:3] = 11
    spatial = fake_spatial(torch.ones((3, 2, 2, 2), dtype=torch.float16))
    grid = label_to_feature_grid(segmentation, spatial)
    payload = build_feature_payload(
        spatial, grid, segmentation, np.eye(4), {11: "tooth"}
    )
    path = tmp_path / "features.pt"
    torch.save(payload, path)
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    assert loaded["spatial"].dtype == torch.float16
    assert loaded["regions"]["labels"].tolist() == [11]
    assert loaded["regions"]["names"] == ["tooth"]
    assert loaded["regions"]["mean"].shape == (1, 3)
    centroids, boxes, counts = region_geometry(
        segmentation, np.eye(4), torch.tensor([11])
    )
    torch.testing.assert_close(centroids, torch.tensor([[1.5, 1.5, 1.5]]))
    assert boxes.tolist() == [[1, 1, 1, 3, 3, 3]]
    assert counts.tolist() == [8]

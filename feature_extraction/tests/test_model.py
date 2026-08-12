import numpy as np
import pytest
import torch

from toothfairy2_feature_extraction.model import (
    convert_logits_to_segmentation_memory_efficient,
    validate_checkpoint,
)


def test_checkpoint_validation():
    checkpoint = {
        "network_weights": {"weight": object()},
        "trainer_name": "nnUNetTrainerUMambaBot",
        "init_args": {"configuration": "3d_fullres"},
    }
    assert validate_checkpoint(checkpoint) is checkpoint
    with pytest.raises(ValueError, match="missing required keys"):
        validate_checkpoint({"network_weights": {}})
    with pytest.raises(ValueError, match="non-empty"):
        validate_checkpoint(
            {"network_weights": {}, "trainer_name": "x", "init_args": {}}
        )


def test_memory_efficient_conversion_matches_resample_then_argmax():
    from nnunetv2.preprocessing.resampling.default_resampling import (
        resample_data_or_seg_to_shape,
    )

    class Plans:
        transpose_forward = (0, 1, 2)
        transpose_backward = (0, 1, 2)

    class Configuration:
        spacing = (1.0, 1.0, 1.0)

        @staticmethod
        def resampling_fn_probabilities(data, shape, current_spacing, new_spacing):
            return resample_data_or_seg_to_shape(
                data,
                shape,
                current_spacing,
                new_spacing,
                is_seg=False,
                order=1,
                order_z=0,
                force_separate_z=None,
            )

    class Labels:
        foreground_labels = list(range(1, 4))
        has_regions = False
        num_segmentation_heads = 4

    generator = np.random.default_rng(4)
    logits = generator.normal(size=(4, 3, 4, 5)).astype(np.float32)
    properties = {
        "spacing": (1.0, 1.0, 1.0),
        "shape_after_cropping_and_before_resampling": (5, 6, 7),
        "shape_before_cropping": (7, 9, 10),
        "bbox_used_for_cropping": ((1, 6), (2, 8), (1, 8)),
    }
    expected_logits = Configuration.resampling_fn_probabilities(
        logits, (5, 6, 7), Configuration.spacing, properties["spacing"]
    )
    expected = np.zeros(properties["shape_before_cropping"], dtype=np.uint8)
    expected[1:6, 2:8, 1:8] = expected_logits.argmax(0)

    actual = convert_logits_to_segmentation_memory_efficient(
        torch.from_numpy(logits), Plans(), Configuration(), Labels(), properties
    )
    np.testing.assert_array_equal(actual, expected)

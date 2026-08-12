from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class PreprocessedCase:
    data: torch.Tensor
    properties: dict


def validate_checkpoint(checkpoint: object) -> dict:
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must contain a dictionary")
    required = {"network_weights", "trainer_name", "init_args"}
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise ValueError("Checkpoint is missing required keys: " + ", ".join(missing))
    if (
        not isinstance(checkpoint["network_weights"], dict)
        or not checkpoint["network_weights"]
    ):
        raise ValueError(
            "Checkpoint network_weights must be a non-empty state dictionary"
        )
    return checkpoint


def load_predictor(
    checkpoint_path: Path, plans_path: Path, dataset_path: Path, device: torch.device
):
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.nets.UMambaBot_3d import get_umamba_bot_3d_from_plans
    from nnunetv2.utilities.label_handling.label_handling import (
        determine_num_input_channels,
    )
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    for path, label in (
        (checkpoint_path, "checkpoint"),
        (plans_path, "plans"),
        (dataset_path, "dataset JSON"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label}: {path}")
    with plans_path.open(encoding="utf-8") as handle:
        plans = json.load(handle)
    with dataset_path.open(encoding="utf-8") as handle:
        dataset_json = json.load(handle)

    checkpoint = validate_checkpoint(
        torch.load(checkpoint_path, map_location="cpu", weights_only=False, mmap=True)
    )
    init_args = checkpoint["init_args"]
    configuration_name = (
        init_args.get("configuration", "3d_fullres")
        if isinstance(init_args, dict)
        else "3d_fullres"
    )
    if configuration_name not in plans.get("configurations", {}):
        configuration_name = "3d_fullres"

    plans_manager = PlansManager(copy.deepcopy(plans))
    configuration_manager = plans_manager.get_configuration(configuration_name)
    input_channels = determine_num_input_channels(
        plans_manager, configuration_manager, dataset_json
    )
    network = get_umamba_bot_3d_from_plans(
        plans_manager,
        dataset_json,
        configuration_manager,
        input_channels,
        deep_supervision=False,
    )
    weights = checkpoint["network_weights"]
    network.load_state_dict(weights, strict=True)

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=False,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )
    predictor.manual_initialization(
        network,
        plans_manager,
        configuration_manager,
        [weights],
        dataset_json,
        str(checkpoint["trainer_name"]),
        checkpoint.get("inference_allowed_mirroring_axes"),
    )
    return predictor, checkpoint


def preprocess_file(predictor, input_path: Path) -> PreprocessedCase:
    preprocessor = predictor.configuration_manager.preprocessor_class(verbose=False)
    data, _seg, properties = preprocessor.run_case(
        [str(input_path)],
        None,
        predictor.plans_manager,
        predictor.configuration_manager,
        predictor.dataset_json,
    )
    return PreprocessedCase(
        torch.from_numpy(np.ascontiguousarray(data)).float(), properties
    )


def predict_segmentation(predictor, case: PreprocessedCase) -> np.ndarray:
    predictor.use_mirroring = True
    logits = predictor.predict_logits_from_preprocessed_data(case.data).cpu()
    return convert_logits_to_segmentation_memory_efficient(
        logits,
        predictor.plans_manager,
        predictor.configuration_manager,
        predictor.label_manager,
        case.properties,
    )


def convert_logits_to_segmentation_memory_efficient(
    predicted_logits: torch.Tensor | np.ndarray,
    plans_manager,
    configuration_manager,
    label_manager,
    properties: dict,
) -> np.ndarray:
    """Restore the original image shape without materializing all output channels.

    nnU-Net's standard exporter resamples the complete C x D x H x W tensor. For
    ToothFairy2 that means 49 full-resolution floating-point volumes at once. The
    argmax (or region threshold) can instead be updated one channel at a time.
    """
    from nnunetv2.configuration import default_num_processes

    old_threads = torch.get_num_threads()
    torch.set_num_threads(default_num_processes)
    try:
        if isinstance(predicted_logits, torch.Tensor):
            logits = predicted_logits.detach().cpu().numpy()
        else:
            logits = np.asarray(predicted_logits)
        if logits.ndim != 4:
            raise ValueError(f"Expected C,D,H,W logits, got shape {logits.shape}")

        target_shape = tuple(
            int(value)
            for value in properties["shape_after_cropping_and_before_resampling"]
        )
        spacing_transposed = [
            properties["spacing"][i] for i in plans_manager.transpose_forward
        ]
        current_spacing = (
            configuration_manager.spacing
            if len(configuration_manager.spacing) == len(target_shape)
            else [spacing_transposed[0], *configuration_manager.spacing]
        )
        output_spacing = spacing_transposed
        segmentation_dtype = (
            np.uint8 if len(label_manager.foreground_labels) < 255 else np.uint16
        )
        segmentation_cropped = np.zeros(target_shape, dtype=segmentation_dtype)

        if label_manager.has_regions:
            if len(label_manager.regions_class_order) != logits.shape[0]:
                raise ValueError("Region/channel count mismatch")
            for channel, label in enumerate(label_manager.regions_class_order):
                resampled = configuration_manager.resampling_fn_probabilities(
                    logits[channel : channel + 1],
                    target_shape,
                    current_spacing,
                    output_spacing,
                )
                # sigmoid(x) > 0.5 is exactly x > 0; later regions overwrite earlier ones.
                np.copyto(
                    segmentation_cropped, int(label), where=np.asarray(resampled)[0] > 0
                )
                del resampled
        else:
            if logits.shape[0] != label_manager.num_segmentation_heads:
                raise ValueError("Class/channel count mismatch")
            best_scores = None
            update_mask = np.empty(target_shape, dtype=bool)
            for channel in range(logits.shape[0]):
                resampled = configuration_manager.resampling_fn_probabilities(
                    logits[channel : channel + 1],
                    target_shape,
                    current_spacing,
                    output_spacing,
                )
                scores = np.asarray(resampled)[0]
                if best_scores is None:
                    # Usually this takes ownership of the one-channel resampling result.
                    # Copy only when resampling returned a view into the input logits.
                    best_scores = (
                        scores.copy() if np.shares_memory(scores, logits) else scores
                    )
                else:
                    np.greater(scores, best_scores, out=update_mask)
                    np.copyto(segmentation_cropped, channel, where=update_mask)
                    np.maximum(best_scores, scores, out=best_scores)
                del resampled, scores
            del best_scores, update_mask

        segmentation = np.zeros(
            tuple(int(value) for value in properties["shape_before_cropping"]),
            dtype=segmentation_dtype,
        )
        bbox = properties["bbox_used_for_cropping"]
        segmentation[tuple(slice(int(low), int(high)) for low, high in bbox)] = (
            segmentation_cropped
        )
        return segmentation.transpose(plans_manager.transpose_backward)
    finally:
        torch.set_num_threads(old_threads)

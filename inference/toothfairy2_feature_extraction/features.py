from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from nnunetv2.inference.sliding_window_prediction import compute_gaussian

from .model import PreprocessedCase


@dataclass
class SpatialFeatures:
    tensor: torch.Tensor
    stride_zyx: tuple[int, int, int]
    spacing_zyx: tuple[float, float, float]
    affine: torch.Tensor
    preprocessed_shape: tuple[int, int, int]
    properties: dict


def _aligned_starts(size: int, patch: int, step: int) -> tuple[list[int], int]:
    if size <= patch:
        return [0], patch
    count = math.ceil((size - patch) / step)
    padded = patch + count * step
    return [index * step for index in range(count + 1)], padded


def feature_affine(
    input_affine: np.ndarray,
    input_spacing_xyz: tuple[float, float, float],
    target_spacing_zyx: tuple[float, float, float],
    stride_zyx: tuple[int, int, int],
    properties: dict,
) -> torch.Tensor:
    crop_start_zyx = np.asarray(properties["bbox_used_for_cropping"], dtype=float)[:, 0]
    crop_start_xyz = crop_start_zyx[::-1]
    base = np.asarray(input_affine, dtype=np.float64)
    directions = base[:3, :3] / np.asarray(input_spacing_xyz, dtype=float)[None]
    affine = np.eye(4, dtype=np.float64)
    affine[:3, 3] = nib.affines.apply_affine(base, crop_start_xyz)
    feature_spacing_xyz = np.asarray(target_spacing_zyx[::-1]) * np.asarray(
        stride_zyx[::-1]
    )
    affine[:3, :3] = directions * feature_spacing_xyz[None]
    return torch.from_numpy(affine.astype(np.float32))


@torch.inference_mode()
def extract_bottleneck_features(
    predictor,
    case: PreprocessedCase,
    input_affine: np.ndarray,
    input_spacing_xyz: tuple[float, float, float],
) -> SpatialFeatures:
    network = predictor.network.to(predictor.device)
    network.eval()
    captured: list[torch.Tensor] = []

    def hook(_module, _inputs, output):
        captured.append(output.detach())

    handle = network.mamba_layer.register_forward_hook(hook)
    data = case.data
    patch = tuple(int(value) for value in predictor.configuration_manager.patch_size)
    original_shape = tuple(int(value) for value in data.shape[1:])
    try:
        probe = torch.zeros(
            (1, data.shape[0], *patch), dtype=data.dtype, device=predictor.device
        )
        with torch.autocast(
            predictor.device.type, enabled=predictor.device.type == "cuda"
        ):
            output = network(probe)
        del output, probe
        patch_feature_shape = tuple(int(value) for value in captured.pop().shape[2:])
        if any(
            width % feature_width
            for width, feature_width in zip(patch, patch_feature_shape)
        ):
            raise RuntimeError(
                f"Non-integral bottleneck stride: patch={patch}, feature={patch_feature_shape}"
            )
        stride = tuple(
            width // feature_width
            for width, feature_width in zip(patch, patch_feature_shape)
        )
        step = tuple(
            max(scale, (width // 2 // scale) * scale)
            for width, scale in zip(patch, stride)
        )
        starts_and_sizes = [
            _aligned_starts(size, width, delta)
            for size, width, delta in zip(original_shape, patch, step)
        ]
        starts = [item[0] for item in starts_and_sizes]
        padded_shape = tuple(item[1] for item in starts_and_sizes)
        pad_width = []
        for current, target in reversed(list(zip(original_shape, padded_shape))):
            pad_width.extend((0, target - current))
        padded = (
            data
            if original_shape == padded_shape
            else F.pad(data, pad_width, mode="constant", value=0)
        )

        feature_shape = tuple(
            target // scale for target, scale in zip(padded_shape, stride)
        )
        accumulator = None
        # Keep whole-volume accumulation off CUDA; only the current patch uses VRAM.
        weights = torch.zeros(feature_shape, dtype=torch.float32, device="cpu")
        gaussian = compute_gaussian(
            patch_feature_shape,
            sigma_scale=1 / 8,
            value_scaling_factor=10,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        for start in itertools.product(*starts):
            slices = tuple(
                slice(begin, begin + width) for begin, width in zip(start, patch)
            )
            patch_data = padded[(slice(None), *slices)][None].to(
                predictor.device, non_blocking=True
            )
            with torch.autocast(
                predictor.device.type, enabled=predictor.device.type == "cuda"
            ):
                output = network(patch_data)
            del output, patch_data
            feature = captured.pop()[0].to(device="cpu", dtype=torch.float32)
            if accumulator is None:
                accumulator = torch.zeros(
                    (feature.shape[0], *feature_shape),
                    dtype=torch.float32,
                    device="cpu",
                )
            feature_start = tuple(begin // scale for begin, scale in zip(start, stride))
            feature_slices = tuple(
                slice(begin, begin + width)
                for begin, width in zip(feature_start, patch_feature_shape)
            )
            accumulator[(slice(None), *feature_slices)] += feature * gaussian
            weights[feature_slices] += gaussian
        if accumulator is None:
            raise RuntimeError("No sliding-window features were generated")
        accumulator /= weights.clamp_min_(1e-7)[None]
        cropped_shape = tuple(
            math.ceil(size / scale) for size, scale in zip(original_shape, stride)
        )
        spatial = accumulator[
            (slice(None), *(slice(0, size) for size in cropped_shape))
        ]
        spatial = spatial.to(dtype=torch.float16).contiguous()
        target_spacing = tuple(
            float(value) for value in predictor.configuration_manager.spacing
        )
        return SpatialFeatures(
            tensor=spatial,
            stride_zyx=stride,
            spacing_zyx=tuple(a * b for a, b in zip(target_spacing, stride)),
            affine=feature_affine(
                input_affine, input_spacing_xyz, target_spacing, stride, case.properties
            ),
            preprocessed_shape=original_shape,
            properties=case.properties,
        )
    finally:
        handle.remove()
        captured.clear()


def label_to_feature_grid(
    label_zyx: np.ndarray, features: SpatialFeatures, transpose_forward=(0, 1, 2)
) -> torch.Tensor:
    transposed = label_zyx.transpose(tuple(int(value) for value in transpose_forward))
    bbox = features.properties["bbox_used_for_cropping"]
    cropped = transposed[tuple(slice(int(low), int(high)) for low, high in bbox)]
    tensor = torch.from_numpy(np.ascontiguousarray(cropped)).float()[None, None]
    preprocessed = F.interpolate(
        tensor, size=features.preprocessed_shape, mode="nearest"
    )
    grid = F.interpolate(preprocessed, size=features.tensor.shape[1:], mode="nearest")
    return grid[0, 0].to(torch.int64)


def global_pool(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    flat = features.float().flatten(1)
    return flat.mean(1).half(), flat.amax(1).half()


def region_pool(
    features: torch.Tensor, labels: torch.Tensor, region_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    means = torch.zeros((len(region_ids), features.shape[0]), dtype=torch.float16)
    maxima = torch.zeros_like(means)
    flat_features = features.float().flatten(1)
    flat_labels = labels.flatten()
    for row, label in enumerate(region_ids.tolist()):
        mask = flat_labels == int(label)
        if mask.any():
            selected = flat_features[:, mask]
            means[row] = selected.mean(1).half()
            maxima[row] = selected.amax(1).half()
    return means, maxima

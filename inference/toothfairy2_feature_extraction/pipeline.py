from __future__ import annotations

import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from . import __version__
from .features import (
    extract_bottleneck_features,
    global_pool,
    label_to_feature_grid,
    region_pool,
)
from .io_utils import (
    atomic_replace_directory,
    case_id_for,
    save_segmentation,
    sha256,
    write_json,
)
from .model import load_predictor, predict_segmentation, preprocess_file


def region_geometry(
    segmentation_zyx: np.ndarray, affine: np.ndarray, labels: torch.Tensor
):
    centroids = torch.zeros((len(labels), 3), dtype=torch.float32)
    boxes = torch.zeros((len(labels), 6), dtype=torch.int64)
    counts = torch.zeros(len(labels), dtype=torch.int64)
    for row, label in enumerate(labels.tolist()):
        coordinates = np.argwhere(segmentation_zyx == int(label))
        if not len(coordinates):
            continue
        centroid_xyz = coordinates.mean(0)[::-1]
        centroids[row] = torch.from_numpy(
            nib.affines.apply_affine(affine, centroid_xyz).astype(np.float32)
        )
        low_xyz = coordinates.min(0)[::-1]
        high_xyz = coordinates.max(0)[::-1] + 1
        boxes[row] = torch.from_numpy(
            np.concatenate((low_xyz, high_xyz)).astype(np.int64)
        )
        counts[row] = len(coordinates)
    return centroids, boxes, counts


def build_feature_payload(
    spatial_features, label_grid, segmentation_zyx, input_affine, label_names
):
    labels = torch.tensor(
        [int(value) for value in np.unique(segmentation_zyx) if value != 0],
        dtype=torch.int64,
    )
    global_mean, global_max = global_pool(spatial_features.tensor)
    means, maxima = region_pool(spatial_features.tensor, label_grid, labels)
    centroids, boxes, counts = region_geometry(segmentation_zyx, input_affine, labels)
    return {
        "spatial": spatial_features.tensor.cpu(),
        "global_mean": global_mean.cpu(),
        "global_max": global_max.cpu(),
        "spacing": torch.tensor(spatial_features.spacing_zyx, dtype=torch.float32),
        "affine": spatial_features.affine.cpu(),
        "stride": torch.tensor(spatial_features.stride_zyx, dtype=torch.int64),
        "regions": {
            "labels": labels,
            "names": [
                label_names.get(int(value), f"label_{int(value)}") for value in labels
            ],
            "mean": means,
            "max": maxima,
            "voxel_count": counts,
            "centroid_world_xyz": centroids,
            "bbox_xyz": boxes,
        },
    }


class ToothFairy2Pipeline:
    def __init__(
        self, checkpoint: Path, plans: Path, dataset: Path, device: torch.device
    ):
        self.checkpoint_path = checkpoint.resolve()
        self.plans_path = plans.resolve()
        self.dataset_path = dataset.resolve()
        self.device = device
        self.predictor, self.checkpoint = load_predictor(
            self.checkpoint_path, self.plans_path, self.dataset_path, device
        )
        labels = self.predictor.dataset_json["labels"]
        self.label_names = {int(value): name for name, value in labels.items()}
        self.checkpoint_digest = sha256(self.checkpoint_path)

    def process(
        self, input_path: Path, output_dir: Path, overwrite: bool = False
    ) -> str:
        case_id = case_id_for(input_path)
        destination = output_dir / case_id
        expected = [
            destination / name
            for name in ("segmentation.nii.gz", "features.pt", "metadata.json")
        ]
        if not overwrite and all(path.is_file() for path in expected):
            print(f"[{case_id}] skipped (complete)", flush=True)
            return "skipped"
        output_dir.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{case_id}.tmp-", dir=output_dir))
        try:
            print(f"[{case_id}] preprocessing", flush=True)
            case = preprocess_file(self.predictor, input_path)
            reference = nib.load(str(input_path))
            input_spacing_xyz = tuple(
                float(value) for value in reference.header.get_zooms()[:3]
            )
            print(f"[{case_id}] segmentation inference", flush=True)
            segmentation = predict_segmentation(self.predictor, case)
            print(f"[{case_id}] bottleneck feature inference", flush=True)
            spatial = extract_bottleneck_features(
                self.predictor, case, reference.affine, input_spacing_xyz
            )
            grid = label_to_feature_grid(
                segmentation, spatial, self.predictor.plans_manager.transpose_forward
            )
            payload = build_feature_payload(
                spatial, grid, segmentation, reference.affine, self.label_names
            )
            save_segmentation(
                temporary / "segmentation.nii.gz", segmentation, input_path
            )
            torch.save(payload, temporary / "features.pt")
            metadata = {
                "schema_version": 1,
                "pipeline_version": __version__,
                "case_id": case_id,
                "input_path": str(input_path.resolve()),
                "checkpoint_path": str(self.checkpoint_path),
                "checkpoint_sha256": self.checkpoint_digest,
                "trainer_name": str(self.checkpoint["trainer_name"]),
                "plans_name": self.predictor.plans_manager.plans.get("plans_name"),
                "configuration": "3d_fullres",
                "device": str(self.device),
                "axis_conventions": {
                    "segmentation_nifti": "x,y,z (original CBCT geometry)",
                    "spatial_feature_tensor": "C,D,H,W where D,H,W are preprocessed z,y,x",
                    "region_bbox": "x_min,y_min,z_min,x_max,y_max,z_max; upper bounds exclusive",
                },
                "input_shape_xyz": list(reference.shape[:3]),
                "input_spacing_xyz": list(input_spacing_xyz),
                "preprocessed_shape_zyx": list(spatial.preprocessed_shape),
                "feature_shape_czyx": list(spatial.tensor.shape),
                "feature_stride_zyx": list(spatial.stride_zyx),
                "processed_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            write_json(temporary / "metadata.json", metadata)
            atomic_replace_directory(temporary, destination)
            print(f"[{case_id}] complete", flush=True)
            return "complete"
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

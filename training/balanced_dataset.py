from __future__ import annotations

import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler

from structured import structured_targets


@dataclass(frozen=True)
class ManifestRecord:
    case_id: str
    feature_path: Path
    report_paths: tuple[Path, ...]
    labels: tuple[str, ...]
    duplicate_group: str
    template_group: str
    sampling_weight: float


def load_manifest(path: str | Path) -> list[ManifestRecord]:
    path = Path(path).expanduser().resolve()
    records = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            reports = tuple(Path(value) for value in row["report_paths"])
            feature = Path(row["feature_path"])
            if not feature.is_file():
                raise FileNotFoundError(f"{path}:{number}: {feature}")
            if not reports or any(not item.is_file() for item in reports):
                raise FileNotFoundError(f"{path}:{number}: invalid report_paths")
            records.append(
                ManifestRecord(
                    str(row["case_id"]),
                    feature,
                    reports,
                    tuple(row.get("labels", [])),
                    str(row["duplicate_group"]),
                    str(row["template_group"]),
                    float(row.get("sampling_weight", 1.0)),
                )
            )
    if not records:
        raise ValueError(f"Empty manifest: {path}")
    return records


def _region_metadata(regions: dict, indices: torch.Tensor) -> torch.Tensor:
    count = int(indices.numel())
    if not count:
        return torch.zeros(0, 10, dtype=torch.float32)
    centroid = regions["centroid_world_xyz"].float()[indices]
    bbox = regions["bbox_xyz"].float()[indices]
    voxels = regions["voxel_count"].float()[indices]
    lower, upper = bbox[:, :3], bbox[:, 3:]
    centroid_origin = centroid.min(dim=0).values
    centroid_extent = (centroid.max(dim=0).values - centroid_origin).clamp_min(1.0)
    centroid = 2.0 * (centroid - centroid_origin) / centroid_extent - 1.0
    bbox_origin = lower.min(dim=0).values
    bbox_extent = (upper.max(dim=0).values - bbox_origin).clamp_min(1.0)
    normalized_bbox = torch.cat(
        ((lower - bbox_origin) / bbox_extent, (upper - bbox_origin) / bbox_extent),
        dim=1,
    )
    volume = torch.log1p(voxels) / torch.log1p(voxels.max()).clamp_min(1.0)
    return torch.cat((centroid, normalized_bbox, volume.unsqueeze(1)), dim=1)


class ManifestDataset(Dataset):
    def __init__(
        self,
        records,
        pool_grid=(4, 4, 4),
        expected_channels=320,
        include_shuffled_image=False,
        shuffle_seed=42,
        *,
        token_mode="global_region",
        max_region_tokens=48,
        include_global_tokens=True,
    ):
        self.records = records
        self.pool_grid = tuple(pool_grid)
        self.expected_channels = expected_channels
        self.token_mode = str(token_mode)
        self.max_region_tokens = int(max_region_tokens)
        self.include_global_tokens = bool(include_global_tokens)
        if self.token_mode not in {"global", "region", "global_region"}:
            raise ValueError(f"Unknown image token mode: {self.token_mode}")
        self.shuffle_sources = None
        if include_shuffled_image and len(records) > 1:
            order = list(range(len(records)))
            random.Random(int(shuffle_seed)).shuffle(order)
            self.shuffle_sources = {
                destination: order[(position + 1) % len(order)]
                for position, destination in enumerate(order)
            }

    def __len__(self):
        return len(self.records)

    def _load_features(self, record) -> dict[str, torch.Tensor]:
        payload = torch.load(record.feature_path, map_location="cpu", weights_only=True)
        spatial = payload.get("spatial")
        if not isinstance(spatial, torch.Tensor) or spatial.ndim != 4:
            raise ValueError(f"{record.feature_path}: spatial must be C,D,H,W")
        if (
            self.expected_channels is not None
            and spatial.shape[0] != self.expected_channels
        ):
            raise ValueError(
                f"{record.feature_path}: expected {self.expected_channels} channels, got {spatial.shape[0]}"
            )
        feature_dim = int(spatial.shape[0]) * 2
        feature_parts, type_parts, label_parts, metadata_parts = [], [], [], []
        use_global = (
            self.token_mode in {"global", "global_region"}
            and self.include_global_tokens
        )
        if use_global:
            x = spatial.float().unsqueeze(0)
            pooled = (
                torch.cat(
                    (
                        F.adaptive_avg_pool3d(x, self.pool_grid),
                        F.adaptive_max_pool3d(x, self.pool_grid),
                    ),
                    1,
                )
                .flatten(2)
                .transpose(1, 2)[0]
                .contiguous()
            )
            size = pooled.shape[0]
            feature_parts.append(pooled)
            type_parts.append(torch.zeros(size, dtype=torch.long))
            label_parts.append(torch.zeros(size, dtype=torch.long))
            metadata_parts.append(torch.zeros(size, 10, dtype=torch.float32))

        if self.token_mode in {"region", "global_region"}:
            regions = payload.get("regions")
            required = {
                "labels",
                "mean",
                "max",
                "voxel_count",
                "centroid_world_xyz",
                "bbox_xyz",
            }
            if not isinstance(regions, dict) or not required.issubset(regions):
                if self.token_mode == "region":
                    raise ValueError(
                        f"{record.feature_path}: required region features are missing"
                    )
            else:
                labels = regions["labels"].long()
                order = sorted(
                    range(len(labels)),
                    key=lambda index: (int(labels[index]) < 11, int(labels[index])),
                )[: self.max_region_tokens]
                indices = torch.tensor(order, dtype=torch.long)
                region_features = torch.cat(
                    (regions["mean"].float()[indices], regions["max"].float()[indices]),
                    dim=1,
                )
                feature_parts.append(region_features)
                type_parts.append(torch.ones(len(indices), dtype=torch.long))
                label_parts.append(labels[indices])
                metadata_parts.append(_region_metadata(regions, indices))

        if not feature_parts:
            raise ValueError(f"{record.feature_path}: no image tokens were produced")
        features = torch.cat(feature_parts, dim=0)
        if features.shape[1] != feature_dim:
            raise ValueError(
                f"{record.feature_path}: inconsistent image feature dimension"
            )
        return {
            "image_features": features,
            "image_attention_mask": torch.ones(features.shape[0], dtype=torch.long),
            "image_token_type_ids": torch.cat(type_parts),
            "region_label_ids": torch.cat(label_parts),
            "region_metadata": torch.cat(metadata_parts),
        }

    def __getitem__(self, index):
        if isinstance(index, tuple):
            case_index, report_index = index
        else:
            case_index, report_index = index, 0
        record = self.records[case_index]
        references = [
            path.read_text(encoding="utf-8").strip() for path in record.report_paths
        ]
        tooth_targets, case_targets = structured_targets(references)
        sample = {
            "case_id": record.case_id,
            **self._load_features(record),
            "report": references[report_index % len(references)],
            "references": references,
            "sample_weight": record.sampling_weight,
            "tooth_targets": tooth_targets,
            "case_targets": case_targets,
        }
        if self.shuffle_sources is not None:
            source = self.records[self.shuffle_sources[case_index]]
            shuffled = self._load_features(source)
            for key, value in shuffled.items():
                sample[f"shuffled_{key}"] = value
            sample["shuffle_source_case_id"] = source.case_id
        return sample


def _rank_world() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))


def _offset(case_id: str) -> int:
    return int(hashlib.sha1(case_id.encode()).hexdigest()[:8], 16)


class EpochTemplateSampler(Sampler):
    def __init__(self, records, cap=3, seed=42, enabled=True, near_cap=None):
        self.records, self.cap, self.seed, self.enabled, self.epoch = (
            records,
            int(cap),
            int(seed),
            enabled,
            0,
        )
        self.near_cap = int(near_cap if near_cap is not None else cap)
        self.groups = defaultdict(list)
        for index, record in enumerate(records):
            self.groups[record.template_group].append(index)
        self.near_groups = defaultdict(list)
        for index, record in enumerate(records):
            self.near_groups[record.duplicate_group].append(index)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def global_indices(self) -> list[tuple[int, int]]:
        chosen_indices = []
        if self.enabled:
            for group in sorted(self.near_groups):
                members = sorted(
                    self.near_groups[group],
                    key=lambda index: self.records[index].case_id,
                )
                limit = min(len(members), self.near_cap)
                start = (self.epoch * limit) % len(members)
                rotated = [
                    members[(start + position) % len(members)]
                    for position in range(len(members))
                ]
                exact_counts = Counter()
                for index in rotated:
                    exact = self.records[index].template_group
                    if exact_counts[exact] >= self.cap:
                        continue
                    chosen_indices.append(index)
                    exact_counts[exact] += 1
                    if sum(exact_counts.values()) >= limit:
                        break
        else:
            chosen_indices = list(range(len(self.records)))
        chosen = []
        for index in chosen_indices:
            count = len(self.records[index].report_paths)
            annotation = (
                self.seed + self.epoch + _offset(self.records[index].case_id)
            ) % count
            chosen.append((index, annotation))
        random.Random(self.seed + self.epoch).shuffle(chosen)
        return chosen

    def __iter__(self):
        values = self.global_indices()
        rank, world = _rank_world()
        usable = len(values) - len(values) % world
        return iter(values[:usable][rank::world])

    def __len__(self):
        total = len(self.global_indices())
        return total // _rank_world()[1]


class DistributedEvalSampler(Sampler):
    def __init__(self, size: int):
        self.size = size

    def __iter__(self):
        rank, world = _rank_world()
        return iter(range(rank, self.size, world))

    def __len__(self):
        rank, world = _rank_world()
        return max(0, (self.size - rank + world - 1) // world)

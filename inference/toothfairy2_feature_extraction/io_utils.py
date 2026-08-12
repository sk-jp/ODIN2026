from __future__ import annotations

import glob
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


def case_id_for(path: Path) -> str:
    path = path.resolve()
    if path.name == "volume.nii.gz" and path.parent.name == "cbct":
        return path.parent.parent.name
    name = path.name
    return name[:-7] if name.endswith(".nii.gz") else path.stem


def _expand_input(value: str) -> list[Path]:
    path = Path(value).expanduser()
    if path.is_file():
        return [path.resolve()]
    if path.is_dir():
        dataset_cases = sorted(path.glob("*/cbct/volume.nii.gz"))
        if dataset_cases:
            return [item.resolve() for item in dataset_cases]
        return sorted(item.resolve() for item in path.glob("*.nii.gz"))
    return sorted(
        Path(item).resolve()
        for item in glob.glob(value, recursive=True)
        if Path(item).is_file()
    )


def resolve_inputs(values: list[str] | None) -> list[Path]:
    if not values:
        raise ValueError("At least one input path, directory, or glob is required")
    found: list[Path] = []
    for value in values:
        found.extend(_expand_input(value))
    if not found:
        raise FileNotFoundError("No input NIfTI volumes matched: " + ", ".join(values))
    by_case: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = {}
    for path in sorted(set(found), key=lambda item: (case_id_for(item), str(item))):
        case_id = case_id_for(path)
        if case_id in by_case and by_case[case_id] != path:
            duplicates.setdefault(case_id, [by_case[case_id]]).append(path)
        by_case[case_id] = path
    if duplicates:
        detail = "; ".join(
            f"{key}: {', '.join(map(str, paths))}" for key, paths in duplicates.items()
        )
        raise ValueError(f"Duplicate case IDs: {detail}")
    return [by_case[key] for key in sorted(by_case)]


def shard_inputs(inputs: list[Path], num_shards: int, shard_index: int) -> list[Path]:
    if num_shards < 1:
        raise ValueError("--num-shards must be at least 1")
    if not 0 <= shard_index < num_shards:
        raise ValueError("--shard-index must satisfy 0 <= index < num_shards")
    return inputs[shard_index::num_shards]


def save_segmentation(
    path: Path, segmentation_zyx: np.ndarray, reference_path: Path
) -> None:
    reference = nib.load(str(reference_path))
    header = reference.header.copy()
    header.set_data_dtype(np.uint8)
    segmentation_xyz = np.ascontiguousarray(segmentation_zyx.transpose(2, 1, 0))
    output = nib.Nifti1Image(
        segmentation_xyz.astype(np.uint8, copy=False), reference.affine, header=header
    )
    output.set_qform(reference.get_qform(), int(reference.header["qform_code"]))
    output.set_sform(reference.get_sform(), int(reference.header["sform_code"]))
    nib.save(output, str(path))


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: Any):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON serialize {type(value).__name__}")


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=_json_default)
        handle.write("\n")


def atomic_replace_directory(source: Path, destination: Path) -> None:
    if destination.exists():
        import shutil

        shutil.rmtree(destination)
    os.replace(source, destination)

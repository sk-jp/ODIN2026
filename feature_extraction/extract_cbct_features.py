#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import gc
import os
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
BENCHMARK_ROOT = Path(
    os.environ.get("TOOTHFAIRY2_BENCHMARK_ROOT", HERE.parent / "ToothFairy2-Benchmark")
).expanduser()
BENCHMARK_PACKAGE = BENCHMARK_ROOT / "benchmark_networks"
if str(BENCHMARK_PACKAGE) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_PACKAGE))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import torch

from toothfairy2_feature_extraction.io_utils import resolve_inputs, shard_inputs
from toothfairy2_feature_extraction.pipeline import ToothFairy2Pipeline


DEFAULT_CHECKPOINT = os.environ.get("TOOTHFAIRY2_CHECKPOINT")
DEFAULT_PLANS = BENCHMARK_ROOT / "nnUNetplans_files/nnUNetPlans.json"
DEFAULT_DATASET = BENCHMARK_ROOT / "dataset.json"


def parse_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type != "cuda":
        raise argparse.ArgumentTypeError("UMambaBot inference requires a CUDA device")
    if not torch.cuda.is_available():
        raise argparse.ArgumentTypeError("CUDA was requested but is not available")
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index < 0 or index >= torch.cuda.device_count():
        raise argparse.ArgumentTypeError(
            f"CUDA device index {index} is unavailable (found {torch.cuda.device_count()} device(s))"
        )
    return torch.device("cuda", index)


def release_case_memory(device: torch.device) -> None:
    """Return per-case Python, CUDA, and glibc caches where possible."""
    gc.collect()
    if device.type == "cuda":
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
    try:
        ctypes.CDLL(None).malloc_trim(0)
    except (AttributeError, OSError):
        pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract ToothFairy2 UMambaBot bottleneck features and maxillofacial segmentations."
    )
    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help="One or more NIfTI paths, directories, or quoted glob patterns",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(DEFAULT_CHECKPOINT).expanduser() if DEFAULT_CHECKPOINT else None,
        required=DEFAULT_CHECKPOINT is None,
        help="UMambaBot checkpoint (or set TOOTHFAIRY2_CHECKPOINT)",
    )
    parser.add_argument(
        "--plans", type=Path, default=DEFAULT_PLANS, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--dataset-json", type=Path, default=DEFAULT_DATASET, help=argparse.SUPPRESS
    )
    parser.add_argument("--device", type=parse_device, default="cuda:0")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--limit", type=int, default=None, help="Process at most N cases after sharding"
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be at least 1")
    try:
        inputs = shard_inputs(
            resolve_inputs(args.input), args.num_shards, args.shard_index
        )
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error
    if args.limit is not None:
        inputs = inputs[: args.limit]
    print(
        f"Shard {args.shard_index + 1}/{args.num_shards}: {len(inputs)} case(s) on {args.device}",
        flush=True,
    )
    if not inputs:
        return 0
    pipeline = ToothFairy2Pipeline(
        args.checkpoint, args.plans, args.dataset_json, args.device
    )
    completed = skipped = 0
    for input_path in inputs:
        try:
            status = pipeline.process(
                input_path, args.output_dir.resolve(), args.overwrite
            )
            completed += status == "complete"
            skipped += status == "skipped"
        finally:
            release_case_memory(args.device)
    print(f"Finished: {completed} completed, {skipped} skipped", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import glob
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import SimpleITK as sitk
import torch
import torch.nn.functional as F
from torch import nn


APP_PATH = Path(__file__).resolve().parent
INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
MODEL_PATH = Path("/opt/ml/model")
WORK_PATH = Path("/tmp/toothfairy4")

# The challenge checkpoint was trained with the benchmark's custom nnU-Net
# package. It must precede the pip package on sys.path.
sys.path.insert(0, str(APP_PATH / "benchmark_networks"))

from toothfairy2_feature_extraction.pipeline import ToothFairy2Pipeline  # noqa: E402


SYSTEM_PROMPT = (
    "You are an expert oral and maxillofacial radiologist. Generate a concise, factual English "
    "diagnostic report from the supplied CBCT feature tokens. Do not invent findings that are not "
    "supported by the input. Explicitly state uncertainty or non-evaluable anatomy."
)
USER_PROMPT = """Review the CBCT in this exact clinical order:
1. scan coverage;
2. findings by jaw;
3. tooth-level findings using FDI notation;
4. relationship to the mandibular canals;
5. maxillary sinuses and temporomandibular joints;
6. lesions;
7. non-evaluable findings and uncertainty.
Then produce one coherent natural-language diagnostic report. Return only the report."""


class FeatureProjector(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        max_region_label: int,
        region_metadata_dim: int = 10,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
        self.token_type_embedding = nn.Embedding(2, output_dim)
        self.region_label_embedding = nn.Embedding(
            max_region_label + 1, output_dim, padding_idx=0
        )
        self.region_metadata_projection = nn.Linear(
            region_metadata_dim, output_dim, bias=False
        )
        self.conditioning_norm = nn.LayerNorm(output_dim)
        self.max_region_label = max_region_label

    def forward(
        self,
        features: torch.Tensor,
        token_type_ids: torch.Tensor,
        region_label_ids: torch.Tensor,
        region_metadata: torch.Tensor,
    ) -> torch.Tensor:
        projected = self.net(features)
        conditioned = (
            projected
            + self.token_type_embedding(token_type_ids.clamp(0, 1))
            + self.region_label_embedding(
                region_label_ids.clamp(0, self.max_region_label)
            )
            + self.region_metadata_projection(region_metadata.to(projected.dtype))
        )
        return self.conditioning_norm(conditioned)


def get_interface_key() -> tuple[str, ...]:
    with (INPUT_PATH / "inputs.json").open(encoding="utf-8") as handle:
        inputs = json.load(handle)
    return tuple(sorted(item["socket"]["slug"] for item in inputs))


def load_cbct() -> tuple[Path, sitk.Image]:
    matches = sorted(glob.glob(str(INPUT_PATH / "images" / "cbct" / "*.mha")))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one .mha CBCT image, found {len(matches)}"
        )
    return Path(matches[0]), sitk.ReadImage(matches[0])


def convert_to_nifti(image: sitk.Image, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(image, str(destination), True)
    return destination


def _region_metadata(regions: dict[str, Any], indices: torch.Tensor) -> torch.Tensor:
    if indices.numel() == 0:
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


def make_image_tokens(feature_path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(feature_path, map_location="cpu", weights_only=True)
    spatial = payload.get("spatial")
    if not isinstance(spatial, torch.Tensor) or spatial.ndim != 4:
        raise ValueError("Extracted spatial features must have shape C,D,H,W")
    if spatial.shape[0] != 320:
        raise ValueError(f"Expected 320 feature channels, got {spatial.shape[0]}")

    values = spatial.float().unsqueeze(0)
    global_tokens = (
        torch.cat(
            (
                F.adaptive_avg_pool3d(values, (4, 4, 4)),
                F.adaptive_max_pool3d(values, (4, 4, 4)),
            ),
            dim=1,
        )
        .flatten(2)
        .transpose(1, 2)[0]
        .contiguous()
    )

    feature_parts = [global_tokens]
    type_parts = [torch.zeros(len(global_tokens), dtype=torch.long)]
    label_parts = [torch.zeros(len(global_tokens), dtype=torch.long)]
    metadata_parts = [torch.zeros(len(global_tokens), 10, dtype=torch.float32)]
    regions = payload.get("regions")
    required = {
        "labels",
        "mean",
        "max",
        "voxel_count",
        "centroid_world_xyz",
        "bbox_xyz",
    }
    if isinstance(regions, dict) and required.issubset(regions):
        labels = regions["labels"].long()
        order = sorted(
            range(len(labels)),
            key=lambda index: (int(labels[index]) < 11, int(labels[index])),
        )[:48]
        indices = torch.tensor(order, dtype=torch.long)
        feature_parts.append(
            torch.cat(
                (regions["mean"].float()[indices], regions["max"].float()[indices]),
                dim=1,
            )
        )
        type_parts.append(torch.ones(len(indices), dtype=torch.long))
        label_parts.append(labels[indices])
        metadata_parts.append(_region_metadata(regions, indices))

    features = torch.cat(feature_parts).unsqueeze(0)
    return {
        "features": features,
        "attention_mask": torch.ones(features.shape[:2], dtype=torch.long),
        "token_type_ids": torch.cat(type_parts).unsqueeze(0),
        "region_label_ids": torch.cat(label_parts).unsqueeze(0),
        "region_metadata": torch.cat(metadata_parts).unsqueeze(0),
    }


def build_prompt(tokenizer: Any) -> list[int]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT},
    ]
    options = {"tokenize": True, "add_generation_prompt": True, "return_dict": False}
    try:
        ids = tokenizer.apply_chat_template(messages, enable_thinking=False, **options)
    except TypeError:
        ids = tokenizer.apply_chat_template(messages, **options)
    return list(ids)


def stage_report_model(destination: Path) -> Path:
    started_at = time.perf_counter()
    destination.mkdir(parents=True)
    for directory in ("qwen-bnb4", "report"):
        shutil.copytree(MODEL_PATH / directory, destination / directory)
    print(f"Staged report model in {time.perf_counter() - started_at:.2f}s")
    return destination


def load_report_model(device: torch.device, model_path: Path):
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    base_path = model_path / "qwen-bnb4"
    report_path = model_path / "report"
    tokenizer = AutoTokenizer.from_pretrained(base_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    started_at = time.perf_counter()
    base = AutoModelForImageTextToText.from_pretrained(
        base_path,
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": str(device)},
    )
    torch.cuda.synchronize(device)
    print(f"Loaded Qwen weights in {time.perf_counter() - started_at:.2f}s")
    language_model = PeftModel.from_pretrained(
        base, report_path / "adapter", local_files_only=True
    ).eval()

    state = torch.load(
        report_path / "projector.pt", map_location="cpu", weights_only=True
    )
    projector = FeatureProjector(
        input_dim=int(state["net.1.weight"].shape[1]),
        hidden_dim=int(state["net.1.weight"].shape[0]),
        output_dim=int(state["net.4.weight"].shape[0]),
        max_region_label=int(state["region_label_embedding.weight"].shape[0] - 1),
        region_metadata_dim=int(state["region_metadata_projection.weight"].shape[1]),
    ).to(device=device, dtype=torch.bfloat16)
    projector.load_state_dict(state, strict=True)
    return language_model, projector.eval(), tokenizer


@torch.inference_mode()
def generate_report(
    tokens: dict[str, torch.Tensor], device: torch.device, model_path: Path
) -> str:
    language_model, projector, tokenizer = load_report_model(device, model_path)
    image_embeddings = projector(
        tokens["features"].to(device=device, dtype=torch.bfloat16),
        tokens["token_type_ids"].to(device),
        tokens["region_label_ids"].to(device),
        tokens["region_metadata"].to(device),
    )
    prompt = torch.tensor(
        build_prompt(tokenizer), dtype=torch.long, device=device
    ).unsqueeze(0)
    text_embeddings = language_model.get_input_embeddings()(prompt)
    inputs_embeds = torch.cat(
        (image_embeddings.to(text_embeddings.dtype), text_embeddings), dim=1
    )
    attention_mask = torch.cat(
        (tokens["attention_mask"].to(device), torch.ones_like(prompt)), dim=1
    )
    generated = language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        max_new_tokens=384,
        do_sample=False,
        use_cache=True,
        repetition_penalty=1.10,
        no_repeat_ngram_size=6,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    report = tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
    if not report:
        raise RuntimeError("The report model returned an empty report")
    return report


def run_model(image: sitk.Image) -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    device = torch.device("cuda:0")
    shutil.rmtree(WORK_PATH, ignore_errors=True)
    staged_model_path = WORK_PATH / "model"

    # The mounted model directory may be backed by slow network storage. Copy the
    # report model sequentially while segmentation is running; safetensors can
    # then memory-map local files without repeated network page faults.
    with ThreadPoolExecutor(max_workers=1) as executor:
        stage_future = executor.submit(stage_report_model, staged_model_path)
        nifti_path = convert_to_nifti(image, WORK_PATH / "input" / "volume.nii.gz")
        feature_root = WORK_PATH / "features"
        extractor = ToothFairy2Pipeline(
            MODEL_PATH / "toothfairy2" / "checkpoint_best.pth",
            APP_PATH / "nnUNetPlans.json",
            APP_PATH / "dataset.json",
            device,
        )
        extractor.process(nifti_path, feature_root, overwrite=True)
        del extractor
        torch.cuda.empty_cache()
        tokens = make_image_tokens(feature_root / "volume" / "features.pt")
        stage_future.result()

    return generate_report(tokens, device, staged_model_path)


def run() -> int:
    interface_key = get_interface_key()
    if interface_key != ("cbct-image",):
        raise RuntimeError(f"Unsupported input interface: {interface_key}")
    _, image = load_cbct()
    report = run_model(image)
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_PATH / "diagnostic-imaging-report.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump({"report": report}, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())

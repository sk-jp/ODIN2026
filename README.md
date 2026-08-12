# CBCT feature-to-report pipeline

This repository contains the three reproducible stages used for an ODIN 2026 Task 1 submission:

1. `feature_extraction/` converts CBCT NIfTI volumes into ToothFairy2 segmentations and 3D feature tensors.
2. `training/` trains a Qwen-based report generator from the extracted image tokens.
3. `inference/` builds the offline, GPU-enabled Grand Challenge inference container.

Training data, patient reports, model weights, experiment outputs, and built container archives are intentionally excluded.

## Repository layout

```text
.
├── feature_extraction/  # ToothFairy2 UMambaBot segmentation and bottleneck extraction
├── training/            # balanced metric-v2 training and manifest generation
└── inference/           # Grand Challenge entry point and Docker build
```

## Requirements

- Linux with an NVIDIA CUDA GPU
- Python 3.9 or newer for feature extraction
- A recent Python/PyTorch CUDA environment for training
- Docker with the NVIDIA Container Toolkit for container inference
- A licensed local copy of the ToothFairy2 checkpoint and Qwen3.5-4B model

The stages use separate pinned or minimum dependency sets. Create a separate virtual environment for each stage to avoid CUDA-package conflicts.

## 1. Extract CBCT features

Place a `ToothFairy2-Benchmark` checkout beside `feature_extraction`, or set `TOOTHFAIRY2_BENCHMARK_ROOT` to it. The checkout must contain `benchmark_networks/`, `nnUNetplans_files/nnUNetPlans.json`, and `dataset.json`.

```bash
cd feature_extraction
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt

export TOOTHFAIRY2_BENCHMARK_ROOT=/path/to/ToothFairy2-Benchmark
python extract_cbct_features.py \
  --input /path/to/cases \
  --output-dir /path/to/features \
  --checkpoint /path/to/checkpoint_best.pth \
  --device cuda:0
```

`--input` accepts NIfTI files, directories, and quoted glob patterns. A dataset directory may use `<case_id>/cbct/volume.nii.gz`; a flat directory of `*.nii.gz` is also accepted. Each output case contains `segmentation.nii.gz`, `features.pt`, and `metadata.json`.

For multiple GPUs, run one process per device with the same `--num-shards` and a distinct zero-based `--shard-index`.

## 2. Build manifests and train

Reports are never bundled with this repository. Prepare this layout:

```text
DATA_ROOT/cases/<case_id>/reports_en/*.txt
FEATURE_ROOT/<case_id>/features.pt
```

Then generate leakage-aware manifests. Exact and near-duplicate report groups are kept in one split.

```bash
cd training
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt

python make_manifest.py \
  --data-root /path/to/DATA_ROOT \
  --feature-root /path/to/FEATURE_ROOT \
  --output-dir datalist
```

Edit `configs/default.yaml`, especially `Model.pretrained_path`, which must point to a local Hugging Face-format Qwen3.5-4B directory. Start training with:

```bash
python train_balanced.py --config configs/default.yaml --gpus 0,1,2
```

The final export is written under `results/<run>/final/` and contains the PEFT adapter, `projector.pt`, and, when enabled, `structured_head.pt`. Weights & Biases logging is disabled by default; local CSV metrics are always written.

## 3. Build and run inference

Arrange the private model resources under `inference/model/` as described in `inference/model/README.md`. Then:

```bash
cd inference
./do_build.sh
INPUT_DIR=/absolute/path/to/one/case ./do_test_run.sh
```

The input directory must contain `inputs.json` and exactly one `images/cbct/*.mha`. The result is written to `test/output/diagnostic-imaging-report.json` by default. The container has no network access at runtime and requires an NVIDIA GPU.

## Tests

CPU-only unit tests do not require private weights:

```bash
python -m pytest -q feature_extraction/tests
python -m pytest -q training/tests
python -m pytest -q inference/tests
```

Dependency-heavy tests may be skipped when their optional runtime packages or private weights are unavailable.

## Data, weights, and licensing

Do not commit clinical data, generated manifests, model weights, model archives, or Docker image archives. Review `feature_extraction/THIRD_PARTY_NOTICE.md` and `inference/benchmark_networks/LICENSE` before distribution. No top-level project license is assigned here; the repository owner should add one only after confirming ownership and third-party compatibility.

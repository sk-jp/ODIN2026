# Grand Challenge inference container

This directory implements the `cbct-image` input socket and the `diagnostic-imaging-report` output socket. It converts an input MHA image to NIfTI, extracts ToothFairy2 features, and generates an English report from a local Qwen adapter.

## Commands

```bash
./do_build.sh
INPUT_DIR=/absolute/path/to/case ./do_test_run.sh
SKIP_BUILD=1 INPUT_DIR=/absolute/path/to/case ./do_test_run.sh
./do_save.sh
```

`do_test_run.sh` expects `INPUT_DIR/inputs.json` and exactly one `INPUT_DIR/images/cbct/*.mha`. `SKIP_BUILD=1` reuses the existing Docker image. The model directory is mounted read-only and is not copied into the image.

The runtime is offline (`--network none`) and requires an NVIDIA CUDA GPU. See `model/README.md` for private artifact placement.

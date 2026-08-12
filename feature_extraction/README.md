# ToothFairy2 feature extraction

This stage runs a pretrained ToothFairy2 UMambaBot, saves the 49-class segmentation in the original CBCT geometry, and extracts its Mamba-processed encoder bottleneck.

See the root `README.md` for installation and execution. Use `--limit 1` for a smoke test and `--overwrite` to replace complete outputs. Segmentation inference uses the checkpoint's mirroring axes; feature extraction uses one deterministic, non-mirrored sliding-window pass.

Each `features.pt` contains `spatial` (`C,D,H,W`, float16), global mean/max values, geometry metadata, and per-region mean/max features, voxel count, centroid, and bounding box. Load it with `torch.load(path, map_location="cpu", weights_only=True)`.

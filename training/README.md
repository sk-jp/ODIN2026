# Balanced feature-to-report training

The active entry point is `train_balanced.py`. It combines 64 global tokens with up to 48 region tokens, projects them into the Qwen embedding space, and trains LoRA, the feature projector, and an optional structured-finding head.

The objective combines causal language modeling, weakly supervised structured findings, and image/report grounding. Validation records precision-aware clinical metrics and a deterministic shuffled-image comparison. See the root `README.md` for the complete preparation and execution commands.

`make_manifest.py` writes only paths and derived labels. Because those paths may reveal local infrastructure and the source reports may contain sensitive information, generated `datalist/` content is excluded from version control.

# Private model resources

This directory is mounted read-only at `/opt/ml/model`. Populate it locally with:

```text
model/
├── qwen-bnb4/                    # local, pre-quantized Qwen3.5-4B HF model
├── report/
│   ├── adapter/                  # PEFT adapter exported by training
│   ├── projector.pt
│   └── structured_head.pt        # optional at inference time
└── toothfairy2/
    └── checkpoint_best.pth       # UMambaBot checkpoint
```

The Qwen directory must load with `AutoModelForImageTextToText.from_pretrained(..., local_files_only=True)` and include its tokenizer. The report directory can be copied from a selected `training/results/<run>/final/` export. None of these files is tracked by Git.

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
IMAGE_TAG=${IMAGE_TAG:-odin2026-task1-sk}
INPUT_DIR=${INPUT_DIR:-"$SCRIPT_DIR/test/input/samples/P001"}
OUTPUT_DIR=${OUTPUT_DIR:-"$SCRIPT_DIR/test/output"}

if [ "${SKIP_BUILD:-0}" != "1" ]; then
  "$SCRIPT_DIR/do_build.sh"
fi
mkdir -p "$OUTPUT_DIR"
test -f "$INPUT_DIR/inputs.json"
test -d "$SCRIPT_DIR/model/toothfairy2"
test -d "$SCRIPT_DIR/model/qwen-bnb4"
test -d "$SCRIPT_DIR/model/report"

docker run --rm \
  --platform=linux/amd64 \
  --network none \
  --gpus all \
  --volume "$INPUT_DIR":/input:ro \
  --volume "$OUTPUT_DIR":/output \
  --volume "$SCRIPT_DIR/model":/opt/ml/model:ro \
  "$IMAGE_TAG"

python - <<'PY' "$OUTPUT_DIR/diagnostic-imaging-report.json"
import json, sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
assert isinstance(payload, dict) and isinstance(payload.get("report"), str) and payload["report"].strip()
print(payload["report"])
PY

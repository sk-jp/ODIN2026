#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
IMAGE_TAG=${IMAGE_TAG:-odin2026-task1-sk}

export DOCKER_QUIET_BUILD=1
"$SCRIPT_DIR/do_build.sh"
build_timestamp=$( docker inspect --format='{{ .Created }}' "$IMAGE_TAG")
if [ -z "$build_timestamp" ]; then
    echo "Error: Failed to retrieve build information for container $IMAGE_TAG"
    exit 1
fi

formatted_build_info=$(echo "$build_timestamp" | sed -E 's/^([0-9]{2})([0-9]{2})-([0-9]{2})-([0-9]{2})T([0-9]{2}):([0-9]{2}).*/\2\3\4\5\6/')
output_filename="${SCRIPT_DIR}/${IMAGE_TAG}_${formatted_build_info}.tar.gz"

docker save "$IMAGE_TAG" | gzip -c > "$output_filename"
tar -czf "$SCRIPT_DIR/model.tar.gz" \
    --exclude='./qwen' \
    -C "$SCRIPT_DIR/model" .

echo "Saved algorithm image and model.tar.gz in $SCRIPT_DIR"

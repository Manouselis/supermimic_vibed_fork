#!/usr/bin/env bash
# Native Thor build. Never installs anything into the host Python/CUDA environment.
set -euo pipefail
task_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
case "$(uname -m)" in
  aarch64|arm64) ;;
  *) echo 'Build the Thor image on the ARM64 Thor. The laptop simulator image is not a Thor image.' >&2; exit 2 ;;
esac
task_image=${THOR_DEPLOY_IMAGE:-supermimic-g1-thor:dev}
task_base=${THOR_BASE_IMAGE:-nvcr.io/nvidia/pytorch:26.03-py3@sha256:9b1b9d6477afce05a5e941a3711093366b43de1b4eb7c1963251ade6a07c34fe}
task_revision=$(git -C "$task_root" rev-parse HEAD)
if [[ -n "$(git -C "$task_root" status --porcelain)" ]]; then task_revision="${task_revision}-dirty"; fi
echo "Building $task_image from $task_base"
echo 'Host driver compatibility must already be checked. No host packages/services are changed.'
docker build --pull --platform linux/arm64 \
  --build-arg "BASE_IMAGE=$task_base" --build-arg "SOURCE_COMMIT=$task_revision" \
  --file "$task_root/deploy/docker/Dockerfile" --tag "$task_image" "$task_root"

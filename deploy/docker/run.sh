#!/usr/bin/env bash
# Offline by default. --robot shares only networking for the receive-only LowState subscriber.
set -euo pipefail
task_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
task_gpu=0
task_robot=0
task_config="$task_root/deploy/config.yaml"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) task_gpu=1; shift ;;
    --robot) task_robot=1; shift ;;
    --config)
      if [[ $# -lt 2 ]]; then echo '--config needs a file' >&2; exit 2; fi
      task_config=$(realpath -- "$2"); shift 2 ;;
    --) shift; break ;;
    *) break ;;
  esac
done
if [[ $# -eq 0 ]]; then
  echo 'Usage: bash deploy/docker/run.sh [--gpu] [--robot] [--config FILE] [CLI global options] COMMAND [options]' >&2
  exit 2
fi
# Identify the application command without mistaking a global-option value for one.
task_args=("$@")
task_command=''
task_index=0
while (( task_index < ${#task_args[@]} )); do
  case "${task_args[$task_index]}" in
    --device|--bundle|--output) task_index=$((task_index+2)) ;;
    --*) echo 'Use --config before the application arguments. Unknown global option.' >&2; exit 2 ;;
    *) task_command="${task_args[$task_index]}"; break ;;
  esac
done
case "$task_command" in
  capture|shadow)
    if [[ $task_robot != 1 ]]; then echo 'Receive-only robot commands require explicit --robot network access.' >&2; exit 2; fi ;;
  inspect|check|list-motions|benchmark|replay)
    if [[ $task_robot == 1 ]]; then echo '--robot is only valid for capture/shadow. Offline commands use no network.' >&2; exit 2; fi ;;
  *) echo 'Supported commands: inspect, check, list-motions, benchmark, replay, capture, shadow.' >&2; exit 2 ;;
esac
task_image=${THOR_DEPLOY_IMAGE:-supermimic-g1-thor:dev}
task_bundle=${THOR_BUNDLE:-$task_root/deploy/artifacts/g1-lafan1-040000}
task_runs=${THOR_RUNS:-$task_root/deploy/runs}
[[ -f "$task_config" && -d "$task_bundle" ]] || { echo 'Missing config or bundle. Transfer and verify them first.' >&2; exit 2; }
task_bundle=$(realpath -- "$task_bundle")
mkdir -p -- "$task_runs"
task_runs=$(realpath -- "$task_runs")
task_role=$(docker image inspect --format '{{index .Config.Labels "org.supermimic.role"}}' "$task_image")
[[ "$task_role" == receive-only ]] || { echo 'Expected our receive-only deployment image. Build it first.' >&2; exit 2; }
task_arch=$(docker image inspect --format '{{.Architecture}}' "$task_image")
case "$(uname -m):$task_arch" in
  aarch64:arm64|arm64:arm64|x86_64:amd64) ;;
  *) echo 'Image architecture does not match this machine.' >&2; exit 2 ;;
esac
task_docker=(run --rm --read-only --cap-drop ALL --security-opt no-new-privileges:true
  --user "$(id -u):$(id -g)" --tmpfs /tmp:rw,noexec,nosuid,nodev,size=256m
  --mount "type=bind,src=$task_bundle,dst=/bundle,readonly"
  --mount "type=bind,src=$task_config,dst=/config/config.yaml,readonly"
  --mount "type=bind,src=$task_runs,dst=/runs")
if [[ $task_robot == 1 ]]; then task_docker+=(--network host); else task_docker+=(--network none); fi
if [[ $task_gpu == 1 ]]; then task_docker+=(--runtime nvidia --gpus all); fi
task_run_id="$(date -u +%Y%m%dT%H%M%S)_$$"
echo "Results: $task_runs/$task_run_id"
exec docker "${task_docker[@]}" "$task_image" \
  --config /config/config.yaml --bundle /bundle --output "/runs/$task_run_id" "$@"

#!/usr/bin/env bash
# Read-only remote inventory. Does not install anything or start DDS/control processes.
set -euo pipefail
if [[ $# != 1 || "$1" == -* ]]; then
  echo 'Usage: bash deploy/check_thor.sh user@NEW_THOR_IP' >&2
  exit 2
fi
task_script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ssh -o StrictHostKeyChecking=ask -o ConnectTimeout=10 -- "$1" python3 - < "$task_script_dir/inspect_host.py"

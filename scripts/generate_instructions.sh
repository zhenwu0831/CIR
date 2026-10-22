#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
cmd=(python "workflows/run_generator.py" "$@")
if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf "%q " "${cmd[@]}"
    printf "\n"
else
    exec "${cmd[@]}"
fi

#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

config_dir="${1:?usage: bash scripts/train_sweep.sh CONFIG_DIRECTORY [key=value ...]}"
shift
if [[ ! -d "$config_dir" ]]; then
    echo "Config directory not found: $config_dir" >&2
    exit 1
fi

shopt -s nullglob
configs=("$config_dir"/*.yaml "$config_dir"/*.yml)
if [[ "${#configs[@]}" == "0" ]]; then
    echo "No YAML training configs in: $config_dir" >&2
    exit 1
fi

for config in "${configs[@]}"; do
    bash scripts/train.sh "$config" "$@"
done

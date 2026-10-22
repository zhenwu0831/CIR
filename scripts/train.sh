#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

config="${1:?usage: bash scripts/train.sh CONFIG [key=value ...]}"
shift
if [[ ! -f "$config" ]]; then
    echo "Config not found: $config" >&2
    exit 1
fi

export FORCE_TORCHRUN=1
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
if [[ ! "$NPROC_PER_NODE" =~ ^[1-9][0-9]*$ ]]; then
    echo "NPROC_PER_NODE must be a positive integer" >&2
    exit 1
fi

command=(llamafactory-cli train "$config" "$@")
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf 'FORCE_TORCHRUN=%q NPROC_PER_NODE=%q ' "$FORCE_TORCHRUN" "$NPROC_PER_NODE"
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi

if ! command -v llamafactory-cli > /dev/null; then
    echo "Install LLaMA-Factory first: python -m pip install -r requirements/train.txt" >&2
    exit 1
fi

mkdir -p runs/launches
run_dir="$(mktemp -d "runs/launches/$(basename "${config%.*}").XXXXXX")"
cp "$config" "$run_dir/config.yaml"
printf '%q ' "${command[@]}" > "$run_dir/command.txt"
printf '\n' >> "$run_dir/command.txt"
printf 'FORCE_TORCHRUN=%s\nNPROC_PER_NODE=%s\nCUDA_VISIBLE_DEVICES=%s\n' \
    "$FORCE_TORCHRUN" "$NPROC_PER_NODE" "${CUDA_VISIBLE_DEVICES:-}" > "$run_dir/environment.txt"
python -m pip freeze > "$run_dir/requirements.txt"
git rev-parse HEAD > "$run_dir/git-commit.txt"
git diff > "$run_dir/git-diff.patch"

"${command[@]}" 2>&1 | tee "$run_dir/train.log"

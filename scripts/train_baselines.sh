#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# Qwen3-0.6B-Base: code-only, natural-language-only, and 50/50 mixed SFT.
for config in \
    configs/train/Qwen3-0.6B-Base/code_data_120K.yml \
    configs/train/Qwen3-0.6B-Base/natural_language_data_120K.yml \
    configs/train/Qwen3-0.6B-Base/combined_data_120K.yml; do
    bash scripts/train.sh "$config" "$@"
done

#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# Rule-based perturbations of CodeAtCMU/CodeInstructions-120K, keyed as in configs/datasets.json.
perturbations=(
    remove_comments
    remove_whitespace
    replace_variables
    replace_keywords_en
    replace_keywords_nonen
    swap_comments_local
    replace_comments_global
)

for perturbation in "${perturbations[@]}"; do
    command=(
        python -m cir.perturb
        --perturbation "$perturbation"
        --name "code_data_120K_$perturbation"
        "$@"
    )
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        printf '%q ' "${command[@]}"
        printf '\n'
    else
        "${command[@]}"
    fi
done

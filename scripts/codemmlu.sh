#!/usr/bin/env bash
set -euo pipefail
model="${1:?usage: codemmlu.sh MODEL OUTPUT_DIR [OPTIONS]}"
output_dir="${2:?usage: codemmlu.sh MODEL OUTPUT_DIR [OPTIONS]}"
shift 2
for subset in programming_syntax api_frameworks software_principles dbms_sql others code_completion fill_in_the_middle code_repair execution_prediction; do
    cmd=(python -m codemmlu --model_name "$model" --output_dir "$output_dir" --subset "$subset" --backend vllm --batch_size 16 --max_new_tokens 128 --temperature 0.0 --prompt_mode zeroshot --trust_remote_code "$@")
    if [[ "${DRY_RUN:-0}" == 1 ]]; then
        printf '%q ' "${cmd[@]}"
        printf '\n'
    else
        "${cmd[@]}"
    fi
done

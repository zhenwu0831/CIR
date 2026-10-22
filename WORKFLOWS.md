# Data construction and code evaluation

Run commands from the repository root, in an environment with `python` available.
Use `cir-prepare`, mixture tools, training launchers, and harness evaluation
for formatting, training, and reasoning evaluation.

```bash
python -m pip install -e .
python -m pip install -r requirements/generation.txt
python -m pip install -e third_party/codemmlu --no-deps
```

CodeMMLU also requires its dependencies listed in `third_party/codemmlu/pyproject.toml`.
Use a separate environment if its requirements conflict with training.

## Instruction construction

The scripts in `workflows/` write local Hugging Face datasets. Source datasets
retain their own terms.

```bash
python workflows/dataset_prep.py --output_dir data/instructions
python workflows/prepare_natural_language_data.py --output_dir data/natural_language
python workflows/process_data.py --output_dir data/code
python workflows/process_data_recreate.py --output_dir data/raw_code_instructions.json
```

Collection combines `open-r1/codeforces-cots` with TokenBender, Evol-Instruct,
red1xe, AtlasUnified, and rombodawg instruction datasets. Filtering and input
concatenation follow the collection script's order. The combined pool is shuffled three
times with seed 42. Instructions receive six-character SHA-256 identifiers.
`process_data.py` reads the published `CodeAtCMU/CodeInstructions-12K`,
expands its first ten languages, then shuffles with seed 42 and selects 120K rows.
Instruction suffix selection is unseeded; `process_data_recreate.py` independently
samples suffixes too. Use the published dataset for the paper's training inputs.

Instruction generation uses an Azure-style HTTP payload and endpoint:

```bash
export API_KEY=YOUR_KEY
export ENDPOINT=YOUR_CHAT_COMPLETIONS_ENDPOINT
bash scripts/generate_instructions.sh --languages_fp workflows/languages.txt
```

The payload uses temperature 0.6 and max_tokens 1000. Configure the hosted model
through your endpoint. Language lines include their newline
in formatting and output keys. The loop reloads saved output but does not skip
existing IDs.

## Generative perturbations

Set `OPENAI_API_KEY` and optionally `OPENAI_BASE_URL` before API commands.
The implementation uses GPT-4o-mini, temperature 0.6, batches of 500, three
attempts with exponential backoff, ID-based resume, and literal placeholder
replacement. The requests do not set max_tokens or a seed.

```bash
bash scripts/generate_perturbation.sh --prompt_fp prompts/good_code.txt
bash scripts/generate_perturbation.sh --prompt_fp prompts/bad_code.txt
bash scripts/generate_perturbation.sh --prompt_fp prompts/gen_pseducode.txt
bash scripts/generate_perturbation.sh --prompt_fp prompts/gen_step_by_step.txt
bash scripts/generate_perturbation.sh --prompt_fp prompts/gen_flowchart.txt
bash scripts/generate_perturbation.sh --prompt_fp prompts/gen_imaginary.txt
```

These commands cover comment enhancement and
obfuscation, pseudocode, step-by-step solutions, flowcharts, and imaginary language.
The suffix application script normalizes the filename's `pseducode` spelling
to `pseudocode`. It preserves random seed 42 across the ordered four-file traversal,
checks raw-instruction containment, then shuffles each output with seed 42:

```bash
python workflows/process_json_files.py \
    --raw_instructions_fp data/raw_code_instructions.json \
    --output_dir data/generative \
    --input_files \
    code-modification/CodeInstructions-120K_instructions_gen_imaginary.json \
    code-modification/CodeInstructions-120K_instructions_gen_pseducode.json \
    code-modification/CodeInstructions-120K_instructions_gen_step_by_step.json \
    code-modification/CodeInstructions-120K_instructions_gen_flowchart.json
```

The raw-instruction map must correspond to the training dataset. For comment enhancement
and obfuscation, each generated JSON contains ID-keyed `instruction`, `old_response`,
and `new_response`.

## HumanEvalX

```bash
bash scripts/humanevalx_generate.sh --model_name_or_path MODEL_OR_CHECKPOINT --output_dir runs/humanevalx/generation
bash scripts/humanevalx_rubrics.sh --output_dir runs/humanevalx/rubrics
bash scripts/humanevalx_judge.sh --model_output_dir runs/humanevalx/generation --rubrics_dir runs/humanevalx/rubrics --output_dir runs/humanevalx/judged
python workflows/aggregate_humanevalx.py --input_dir runs/humanevalx/judged --output_file runs/humanevalx/scores.csv
```

Generation uses raw HumanEvalX prompts for cpp/go/java/js/python, temperature 0.5,
2048 new tokens, batch size 1, and the checkpoint's automatic generation config
for other settings. It skips existing language CSVs. Rubrics use GPT-4o-mini at
temperature 0.6 with 164 concurrent requests; judging uses GPT-4o-mini at temperature
0 with 32. Both use the system message and `.txt` prompt templates in this repository.
Judging extracts integer `<score>` and `<reasoning>` tags; missing scores become
-1. Each response is parsed independently, including the final partial batch.

The summary reports per-model, per-language mean and sample standard deviation,
including -1, and separately counts invalid scores. API errors are saved as
`[ERROR]` strings by the generation loops and skipped on resume; inspect
these before accepting a run.

## CodeMMLU

```bash
bash scripts/codemmlu.sh MODEL_OR_CHECKPOINT runs/codemmlu/MODEL_NAME
python third_party/codemmlu/codemmlu_results2csv.py --root runs/codemmlu --strict --write-combined
```

The launcher preserves the nine subsets, vLLM backend, zeroshot prompts, batch size
16, temperature 0, and 128 new tokens. The local backend first extracts A–D using
ordered regexes, then matches normalized option text. Accuracy is computed per
subset; the aggregator averages the nine subset accuracies without
weighting by example count. Use `--strict` to reject incomplete models. Use the
vLLM backend for the paper's evaluation. Dataset loading uses `Fsoft-AIC/codemmlu`.

Set `DRY_RUN=1` on these Bash launchers to print commands without execution.

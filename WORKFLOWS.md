# Data construction and code evaluation

Run commands from the repository root, in an environment with `python` available.
Existing `cir-prepare`, mixture tools, training launchers, and harness evaluation
remain the interfaces for formatting, training, and reasoning evaluation.

```bash
python -m pip install -e .
python -m pip install -r requirements/generation.txt
python -m pip install -e third_party/codemmlu --no-deps
```

CodeMMLU also requires its dependencies listed in `third_party/codemmlu/pyproject.toml`.
Use a separate environment if its requirements conflict with training.

## Instruction construction

The author scripts are in `workflows/`. Construction writes local Hugging Face
datasets; no command here uploads them. Source datasets retain their own terms.

```bash
python workflows/dataset_prep.py --output_dir data/instructions
python workflows/prepare_natural_language_data.py --output_dir data/natural_language
python workflows/process_data.py --output_dir data/code
python workflows/process_data_recreate.py --output_dir data/raw_code_instructions.json
```

Collection combines `open-r1/codeforces-cots` with TokenBender, Evol-Instruct,
red1xe, AtlasUnified, and rombodawg instruction datasets. Filtering and input
concatenation retain the original order. The combined pool is shuffled three
times with seed 42. Six-character SHA-256 IDs are identifiers, not deduplication.
No explicit deduplication or selection from this pool to the final 12K instructions
was found. `process_data.py` reads the published `CodeAtCMU/CodeInstructions-12K`,
expands its first ten languages, then shuffles with seed 42 and selects 120K rows.
Its random instruction suffix selection has no seed in the original script;
`process_data_recreate.py` independently samples suffixes too. Neither reproduces
the exact historical wording deterministically. Use the published dataset for
the final training inputs.

Instruction generation uses the original Azure-style HTTP payload and endpoint:

```bash
export API_KEY=YOUR_KEY
export ENDPOINT=YOUR_CHAT_COMPLETIONS_ENDPOINT
bash scripts/generate_instructions.sh --languages_fp workflows/languages.txt
```

This is an API command. Its original payload uses temperature 0.6 and max_tokens
1000, and does not identify the hosted model. Language lines include their newline
in formatting and output keys. The loop reloads saved output but does not skip
existing IDs. Confirm the original endpoint/model and final 12K selection before
regenerating the paper dataset.

## Generative perturbations

Set `OPENAI_API_KEY` and optionally `OPENAI_BASE_URL` before API commands.
The implementation uses GPT-4o-mini, temperature 0.6, batches of 500, three
attempts with exponential backoff, ID-based resume, and literal placeholder
replacement. No max_tokens or seed was set in this source implementation.

```bash
bash scripts/generate_perturbation.sh --prompt_fp prompts/good_code.txt
bash scripts/generate_perturbation.sh --prompt_fp prompts/bad_code.txt
bash scripts/generate_perturbation.sh --prompt_fp prompts/gen_pseducode.txt
bash scripts/generate_perturbation.sh --prompt_fp prompts/gen_step_by_step.txt
bash scripts/generate_perturbation.sh --prompt_fp prompts/gen_flowchart.txt
bash scripts/generate_perturbation.sh --prompt_fp prompts/gen_imaginary.txt
```

The spelling `gen_pseducode.txt` follows the source launcher. These cover comment enhancement and
obfuscation, pseudocode, step-by-step solutions, flowcharts, and imaginary language.
The suffix application script normalizes the source launcher's `pseducode` spelling
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

The raw-instruction map must correspond to the exact training dataset. The recreation
script above does not establish historical suffix identity. For comment enhancement
and obfuscation, each generated JSON contains ID-keyed `instruction`, `old_response`,
and `new_response`; the final publication-format conversion was not found.

Implementations for whitespace removal, variable renaming, nonsense/non-English
keyword replacement, comment removal, local comment swapping, and global comment
replacement were not found in either supplied checkout. Their dataset names and
training configurations do not establish the transformation algorithms or seeds.
They have not been replaced with new implementations.

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
0 with 32. Both use the original system message and `.txt` prompt templates.
Judging extracts integer `<score>` and `<reasoning>` tags; missing scores become
-1. The final partial batch now parses each response independently, fixing the
original stale-score bug. Historical result scores need checking against their
raw output before comparison.

The summary reports per-model, per-language mean and sample standard deviation,
including -1 as the original analysis did, and separately counts invalid scores.
It does not claim a final cross-language paper metric: that aggregation and the
mapping from runs to paper tables were not present. API errors are saved as
`[ERROR]` strings by the original generation loops and skipped on resume; inspect
these before accepting a run.

## CodeMMLU

```bash
bash scripts/codemmlu.sh MODEL_OR_CHECKPOINT runs/codemmlu/MODEL_NAME
python third_party/codemmlu/codemmlu_results2csv.py --root runs/codemmlu --strict --write-combined
```

The launcher preserves the nine subsets, vLLM backend, zeroshot prompts, batch size
16, temperature 0, and 128 new tokens. The local backend first extracts A–D using
ordered regexes, then matches normalized option text. Accuracy is computed per
subset; the supplied aggregator averages the nine subset accuracies without
weighting by example count. Use `--strict` to reject incomplete models. The HF
backend is retained upstream but has not been established as equivalent to the
paper's vLLM scoring. Dataset loading uses upstream `Fsoft-AIC/codemmlu`.

Set `DRY_RUN=1` on any new Bash launcher to print commands without execution.

## Remaining paper inputs

The supplied projects contain no notebooks and no identified final paper figure
generator or run-to-table mapping. Training-loss and parameter-analysis scripts
exist, but their relationship to the final figures is unestablished. Raw result
directories and ZIPs were not imported. Still needed: exact final 12K selection
and deduplication, seven rule-based transformations, generative suffix mapping,
final model/dataset revisions, final table/figure inputs, and confirmation of the
training batch-size discrepancy described in the README. Code, dataset, and
checkpoint licenses remain separate; no blanket grant has been added.

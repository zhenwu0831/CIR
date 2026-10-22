# On Code-Induced Reasoning in LLMs

Code and artifacts for our ICLR 2026 paper **On Code-Induced Reasoning in LLMs**.

**Abdul Waheed\*, Zhen Wu\*, Carolyn Rosé, and Daphne Ippolito**  
Carnegie Mellon University  
\*Equal contribution; co-first authors.

[Code](https://github.com/zhenwu0831/CIR) · [Datasets and models](https://huggingface.co/CodeAtCMU)

We study how code affects LLM reasoning through parallel instruction datasets in
10 programming languages, controlled structural and semantic perturbations, and
fine-tuning across five model families. This repository provides data preparation,
LLaMA-Factory training configurations and launch scripts, evaluation tools, and prompts.

## Installation

Use Python 3.10 or newer. Install training and evaluation dependencies in separate
environments.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m pip install -r requirements/train.txt
```

Training uses LLaMA-Factory with DeepSpeed ZeRO-3. The paper uses four A100 80GB GPUs.
Install a CUDA-compatible PyTorch build for your machine before training.

## Data preparation

Prepare the code and natural-language instruction datasets:

```bash
cir-prepare --source CodeAtCMU/CodeInstructions-120K --name code_data_120K
cir-prepare --source CodeAtCMU/NaturalLanguageInstructions-120K --name natural_language_data_120K
```

These commands write Alpaca-format JSON files and `data/dataset_info.json` for
LLaMA-Factory. Use `--revision COMMIT` to select a dataset snapshot. You can also
provide `--local-file FILE.jsonl` with `instruction`, optional `input`, and
`response` or `output` fields.

Prepare the 50/50 code and natural-language baseline:

```bash
python -m cir.mixtures --nl-percent 50 --baseline --name combined_data_120K
```

For configs using the dataset key `mixed_data_120K`, prepare that name instead.

Prepare a single-language subset:

```bash
cir-prepare --source CodeAtCMU/CodeInstructions-120K --name Python_data_12K --language Python
```

Prepare a mixture-ratio experiment:

```bash
python -m cir.mixtures --nl-percent 2.5 --name code_natural_language_mix_nl_2p5_data_120K
```

`--nl-percent` specifies the natural-language fraction. The sweep uses 2.5, 5, 10,
20, 40, 60, 80, 90, 95, and 97.5 percent natural language. Baseline sampling uses
seed 42; ratio experiments sample with `42 + int(percent * 10)` and shuffle the
combined examples with seed 42.

For perturbation datasets, use `cir-prepare --source DATASET_ID --name DATASET_KEY`.
The dataset key must match `dataset` in the training YAML. See
[the dataset catalog](configs/datasets.json) for dataset keys and source IDs.

## Training

### A single experiment

```bash
bash scripts/train.sh configs/train/Qwen3-0.6B-Base/code_data_120K.yml
```

The launcher runs `llamafactory-cli train` with Torchrun on four GPUs by default.
Set `CUDA_VISIBLE_DEVICES` to select GPUs and `NPROC_PER_NODE` to change the process
count. Pass LLaMA-Factory overrides after the config:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 bash scripts/train.sh \
    configs/train/Qwen3-0.6B-Base/code_data_120K.yml \
    gradient_accumulation_steps=2 \
    output_dir=runs/qwen3-0.6b/code
```

The paper uses full SFT in BF16, sequence length 2048, two epochs, learning rate
1e-5, cosine decay, and warmup ratio 0.1. Total batch size is 64 for full-dataset
experiments and 32 for single-language experiments. Check the selected YAML:

```text
effective batch size = per_device_train_batch_size × gradient_accumulation_steps × GPU count
```

For a microbatch of 8 on four GPUs, use accumulation 2 for total batch size 64 and
accumulation 1 for total batch size 32. Configs include experiment-specific batch
settings; the launcher uses them unless overridden.

Model checkpoints go to the config's `output_dir`. Each launch also saves its
console log, config, command-line overrides, GPU process settings, package versions,
and Git revision under `runs/launches/`.

### Baselines

Train the Qwen3-0.6B-Base code-only, natural-language-only, and mixed baselines:

```bash
bash scripts/train_baselines.sh gradient_accumulation_steps=2
```

Prepare all three datasets first. Additional model configs are under `configs/train/`.

### Perturbations, language subsets, and mixture sweeps

Run all YAMLs directly inside a config directory sequentially:

```bash
# Rule-based perturbations.
bash scripts/train_sweep.sh configs/train/Qwen3-0.6B-Base-CorruptedComments

# Generative perturbations.
bash scripts/train_sweep.sh configs/train/Qwen3-0.6B-Base-GenerativePerturbations

# Code/NL mixture ratios.
bash scripts/train_sweep.sh configs/train/Qwen3-0.6B-Base-DataMix
```

Launch a language-specific experiment:

```bash
bash scripts/train.sh configs/train/Qwen3-0.6B-Base/Python_data_12K.yml
```

Sweep arguments after the directory are forwarded to every training run. A sweep
stops if a run fails. Inspect commands without starting training using `DRY_RUN=1`:

```bash
DRY_RUN=1 bash scripts/train_baselines.sh gradient_accumulation_steps=2
```

## Evaluation

In an evaluation environment, install the package and harness:

```bash
python -m pip install -e .
python -m pip install -r requirements/eval.txt
bash scripts/evaluate.sh \
    --model MODEL_OR_LOCAL_CHECKPOINT \
    --tasks arc_easy,arc_challenge,piqa,openbookqa,logiqa \
    --output-dir runs/eval/example
```

Use `--dry-run` to inspect the command and `--num-fewshot` to override task defaults.
Evaluation saves the invocation and package versions alongside the results.
Math benchmarks include GSM8K, HRM8K, arithmetic tasks, and math subsets of MMLU.
HRM8K task definitions are also available under `third_party/tasks/hrm8k/`.

CodeMMLU and HumanEvalX generation, rubric generation, GPT-4o-mini judging,
and aggregation commands are in [WORKFLOWS.md](WORKFLOWS.md). That guide also
covers instruction collection, natural-language construction, generative
perturbations, exact settings, and remaining experimental inputs.

## Repository contents

| Path | Contents |
| --- | --- |
| `scripts/` | Bash launchers for training, sweeps, and evaluation |
| `configs/train/` | Model and dataset training configurations |
| `configs/deepspeed/` | DeepSpeed ZeRO-3 configuration |
| `src/cir/` | Dataset preparation, mixture sampling, and evaluation tools |
| `prompts/` | Generation and perturbation prompts, all in `.txt` format |
| `configs/datasets.json` | Dataset catalog |
| `workflows/` | Instruction construction, generative perturbations, and HumanEvalX |
| `third_party/codemmlu/` | CodeMMLU evaluator and aggregation |

Additional data-generation code, dataset revisions, model links, and final aggregate
results will be added as the release is completed.

## Citation and attribution

If you find our work useful please cite using:

```bibtex
@inproceedings{waheed2026code,
  title={On code-induced reasoning in llms},
  author={Waheed, Abdul and Wu, Zhen and Ros{\'e}, Carolyn and Ippolito, Daphne},
  booktitle={International Conference on Learning Representations},
  volume={2026},
  pages={150435--150476},
  year={2026}
}
```

Training uses
[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory); evaluation uses
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).
Upstream notices are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
Code, dataset, and model licensing will be specified separately.

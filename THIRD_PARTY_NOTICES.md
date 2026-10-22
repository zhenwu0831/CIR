# Third-party notices

The CIR-specific tooling is a cleanup of the author-supplied experiment workflow.
Its license is pending author selection. No repository-wide license is imposed on
third-party data or model weights.

## LLaMA-Factory

Repository: https://github.com/hiyouga/LLaMA-Factory

Training framework fork: https://github.com/macabdul9/LLaMA-Factory

Commit: `d230f23e74fb1656dfeb07f782e96d7ec2632814`.

The ZeRO-3 configuration is copied from that checkout. Training YAMLs are from the CIR experiments, with local paths replaced by repository-relative paths.
The original Apache-2.0 license is retained at
`third_party/licenses/LLaMA-Factory.txt`.

## lm-evaluation-harness

Repository: https://github.com/EleutherAI/lm-evaluation-harness

Commit: `4439847887ea0481f4f1eb335d39f6f5207904b6`.

The HRM8K task directory is copied without modification from that checkout. Its
license is retained at `third_party/licenses/lm-evaluation-harness.txt`. Dataset
terms apply independently; refer to each task's source dataset and README.

## Author-supplied paper and prompts

The prompts in `prompts/paper/` are extracted from `CIR.zip`, with LaTeX figure
wrappers and typesetting commands removed. The other text prompt templates come
from `CodeInstruction/prompts/`. They are research artifacts supplied by the paper's
authors. Their publication license still needs to be selected.

## CodeMMLU

The evaluator under `third_party/codemmlu/` comes from the supplied CodeMMLU
checkout, based on https://github.com/FSoft-AI4Code/CodeMMLU. Its MIT license
and README are retained. The local vLLM answer-extraction and metric implementation
is preserved. The CLI uses importlib.metadata instead of deprecated pkg_resources.
CodeMMLU prompt templates remain verbatim in `.txt` files.

## CIR workflows

The scripts under `workflows/` originate in the author-supplied CodeInstruction
checkout. Local dataset writes replace upload calls; API clients use environment
configuration. HumanEvalX judging fixes the stale score in the final partial batch.
Prompts used by those scripts are preserved from that checkout as plain text.
See WORKFLOWS.md for unresolved experimental details.

# Third-party notices

Third-party code retains its upstream licenses. Dataset and model terms apply
independently.

## LLaMA-Factory

Repository: https://github.com/hiyouga/LLaMA-Factory

Training framework fork: https://github.com/macabdul9/LLaMA-Factory

Commit: `d230f23e74fb1656dfeb07f782e96d7ec2632814`.

The ZeRO-3 configuration comes from that checkout. Training YAMLs specify the CIR experiments.
The original Apache-2.0 license is retained at
`third_party/licenses/LLaMA-Factory.txt`.

## lm-evaluation-harness

Repository: https://github.com/EleutherAI/lm-evaluation-harness

Commit: `4439847887ea0481f4f1eb335d39f6f5207904b6`.

The HRM8K task directory is copied without modification from that checkout. Its
license is retained at `third_party/licenses/lm-evaluation-harness.txt`. Dataset
terms apply independently; refer to each task's source dataset and README.

## Paper and prompts

The prompts in `prompts/paper/` accompany the paper. The other text prompt templates come
from `CodeInstruction/prompts/` and are provided by the paper's authors.

## CodeMMLU

The evaluator under `third_party/codemmlu/` comes from the CodeMMLU
checkout, based on https://github.com/FSoft-AI4Code/CodeMMLU. Its MIT license
and README are retained. The local vLLM answer-extraction and metric implementation
is preserved. The CLI uses importlib.metadata instead of deprecated pkg_resources.
CodeMMLU prompt templates remain verbatim in `.txt` files.

## CIR workflows

The scripts under `workflows/` originate in the authors' CodeInstruction
checkout. Dataset construction writes local files; API clients use environment
configuration.
Prompts used by those scripts are preserved from that checkout as plain text.
See WORKFLOWS.md for commands and experimental settings.

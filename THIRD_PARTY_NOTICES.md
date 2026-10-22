# Third-party notices

The CIR-specific tooling is a cleanup of the author-supplied experiment workflow.
Its license is pending author selection. No repository-wide license is imposed on
third-party data or model weights.

## LLaMA-Factory

Repository: https://github.com/hiyouga/LLaMA-Factory

Training framework fork: https://github.com/macabdul9/LLaMA-Factory

Commit: `d230f23e74fb1656dfeb07f782e96d7ec2632814`.

The ZeRO-3 configuration is copied from that checkout. Training YAMLs are from the CIR experiments; portability changes are listed in
the source manifest.
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
authors. Their publication license still needs to be selected. Source package and
config hashes are recorded in `artifacts/source_manifest.json`.

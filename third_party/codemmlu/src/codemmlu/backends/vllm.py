import os
import re
import json
from tqdm import tqdm
from typing import Dict

import torch
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from codemmlu.backends.base import Backend

CHOICE_SET = ("A", "B", "C", "D")

def _norm_choice(s: str):
    if not s:
        return None
    s = s.strip().upper()
    return s if s in CHOICE_SET else None

_EXTRACT_PATTERNS = [
    r"(?:answer|final answer|final|prediction|choice)\s*[:\-]?\s*([A-D])\b",
    r"option\s*([A-D])\b",
    r"\(\s*([A-D])\s*\)",       # (C)
    r"\b([A-D])[.)]\s",         # C)  or  C.
    r"\b([A-D])\b",             # fallback: lone letter
]

def extract_choice(text: str):
    if not text:
        return None
    for pat in _EXTRACT_PATTERNS:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return _norm_choice(m.group(1))
    return None

def _normalize_for_match(s: str) -> str:
    s = re.sub(r'[^A-Za-z0-9 \n]', "", s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s.lower()

def _parse_choice_map_from_prompt(prompt_text: str) -> Dict[str, str]:
    """
    Extract (A) ... (B) ... (C) ... (D) ... from the prompt block itself.
    Supports multi-line options until the next (X) marker or end.
    """
    mapping: Dict[str, str] = {}
    cur = None
    for line in prompt_text.splitlines():
        m = re.match(r'\(\s*([A-D])\s*\)\s*(.*)', line)
        if m:
            cur = m.group(1).upper()
            mapping[cur] = m.group(2).strip()
        elif cur is not None:
            # continuation of the current option
            mapping[cur] = (mapping[cur] + " " + line.strip()).strip()
    return mapping

def extract_choice_with_fallback(response_text: str, prompt_text: str):
    """
    1) Try explicit A/B/C/D via regex (extract_choice).
    2) If not found, parse options from the prompt and return the letter
       whose option text appears as a substring in the response (case-insensitive).
    """
    # Step 1: regex letter extraction (A–D only)
    letter = extract_choice(response_text)
    if letter:
        return letter

    # Step 2: option-text substring match using what’s inside the prompt
    choice_map = _parse_choice_map_from_prompt(prompt_text)
    if not choice_map:
        return None

    norm_resp = _normalize_for_match(response_text)
    for letter in ("A", "B", "C", "D"):  # determinism: check in order
        norm_opt = _normalize_for_match(choice_map.get(letter, ""))
        if norm_opt and norm_opt in norm_resp:
            return letter

    return None

class VllmEngine(Backend):
    def __init__(self, model_name: str, **kwargs):
        super().__init__(model_name=model_name, **kwargs)
        ngpus = torch.cuda.device_count()
        backend_kwargs = dict(
            disable_log_stats=True,
            tensor_parallel_size=ngpus,
            download_dir=self.cache_dir,
            trust_remote_code=self.trust_remote_code,
        )
        
        self.model = LLM(self.model_name, 
            enable_lora=True if self.peft_model else None,
            **backend_kwargs)
        
        self.lora_request = None
        if self.peft_model:
            self.lora_request=LoRARequest("lora", 1, self.peft_model)
            
        self.sampling_params = SamplingParams(
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )

    def generate(self):
        ds_loader = [self.dataset[i:i+self.batch_size] 
                    for i in range(0, len(self.dataset), self.batch_size)]
        
        # result = []
        gens, tasks, prompts, golds = [], [], [], []
        for batch in tqdm(ds_loader, total=len(ds_loader), desc="Generating"):
            outputs = self.model.generate(batch['question'], 
                                          self.sampling_params, 
                                          lora_request=self.lora_request)

            # batch['generation'] = [output.outputs[0].text for output in outputs]
            # result.extend(batch['generation'])
            # self._save_result(batch)
            batch_gens = [o.outputs[0].text for o in outputs]
            # batch_preds = [extract_choice(x) for x in batch_gens]
            batch_preds = [
                extract_choice_with_fallback(resp, batch['question'][i])
                for i, resp in enumerate(batch_gens)
            ]
            batch_golds = [_norm_choice(g) for g in batch['answer']]

            gens.extend(batch_gens)
            tasks.extend(batch['task_id'])
            prompts.extend(batch['question'])
            golds.extend(batch_golds)

            # stream to disk for this batch
            self._save_result({
                'task_id': batch['task_id'],
                'question': batch['question'],
                'generation': batch_gens,
                'pred': batch_preds,
                'gold': batch_golds,
            })
            
        # self.dataset = self.dataset.add_column('generation', result)
        # TODO: process response and extract answer

        # preds_all = [extract_choice(x) for x in gens]
        preds_all = [
            extract_choice_with_fallback(resp, prmpt)
            for resp, prmpt in zip(gens, prompts)
        ]

        correct_flags = [(p == g) for p, g in zip(preds_all, golds)]
        correct = sum(bool(x) for x in correct_flags)
        total = len(correct_flags)
        accuracy = correct / total if total else 0.0

        self.dataset = (self.dataset
                        .add_column('generation', gens)
                        .add_column('pred', preds_all)
                        .add_column('gold', golds)
                        .add_column('correct', correct_flags))

        os.makedirs(self.output_dir, exist_ok=True)
        with open(os.path.join(self.output_dir, f"{self.subset}.{self.split}.metrics.json"), "w") as f:
            json.dump({
                "subset": self.subset,
                "split": self.split,
                "accuracy": accuracy,
                "correct": correct,
                "total": total,
            }, f, indent=2)

        # optional consolidated JSONL (handy for quick inspection)
        merged_path = os.path.join(self.output_dir, f"{self.subset}.{self.split}.generated.with_preds.jsonl")
        with open(merged_path, "w") as w:
            for t, q, resp, p, g, c in zip(tasks, prompts, gens, preds_all, golds, correct_flags):
                json.dump({
                    "task_id": t,
                    "prompt": q,
                    "response": resp,
                    "pred": p,
                    "gold": g,
                    "correct": bool(c),
                }, w); w.write("\n")

        return self.dataset

    def _save_result(self, batched_outputs: Dict):
        """Append per-row outputs; include pred/gold/correct."""
        assert 'question' in batched_outputs and 'generation' in batched_outputs

        save_path = os.path.join(self.output_dir, f"{self.subset}.final.generated.jsonl")
        with open(save_path, "a") as writer:
            for idx, gen in enumerate(batched_outputs['generation']):
                res = dict(
                    task_id=batched_outputs['task_id'][idx],
                    prompt=batched_outputs['question'][idx],
                    response=gen,
                    pred=batched_outputs.get('pred', [None])[idx],
                    gold=batched_outputs.get('gold', [None])[idx],
                    correct=(batched_outputs.get('pred', [None])[idx]
                            == batched_outputs.get('gold', [None])[idx]),
                )
                json.dump(res, writer); writer.write("\n")

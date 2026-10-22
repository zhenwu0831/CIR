import json
from pathlib import Path

import yaml

from cir.evaluate import evaluation_command

ROOT = Path(__file__).resolve().parents[1]


def test_configs_are_portable_and_match_catalog():
    datasets = {
        item["training_dataset_key"]
        for item in json.loads((ROOT / "configs/datasets.json").read_text())
    }
    configs = list((ROOT / "configs/train").rglob("*.yml"))
    assert configs
    for path in configs:
        config = yaml.safe_load(path.read_text())
        assert config["dataset"] in datasets
        assert config["report_to"] == "none"
        assert config["overwrite_output_dir"] is False
        assert not config.get("resume_from_checkpoint")
        assert not any(isinstance(v, str) and v.startswith("/") for v in config.values())
        if config.get("deepspeed"):
            assert (ROOT / config["deepspeed"]).is_file()


def test_eval_preserves_task_defaults_and_paths_with_spaces():
    command = evaluation_command("model path/weights", "arc_easy", Path("run path"), "auto", None)
    assert "pretrained=model path/weights" in command
    assert command[-1] == "run path"
    assert "--num_fewshot" not in command
    assert "--confirm_run_unsafe_code" not in command
    assert evaluation_command("model", "gsm8k", Path("run"), "4", 0)[-2:] == ["--num_fewshot", "0"]

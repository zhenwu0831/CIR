from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def evaluation_command(
    model: str, tasks: str, output_dir: Path, batch_size: str, fewshot: int | None
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "lm_eval",
        "--model",
        "hf",
        "--model_args",
        f"pretrained={model}",
        "--tasks",
        tasks,
        "--batch_size",
        batch_size,
        "--output_path",
        str(output_dir),
    ]
    if fewshot is not None:
        command.extend(["--num_fewshot", str(fewshot)])
    return command


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run lm-evaluation-harness with recorded arguments"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--num-fewshot", type=int, help="Omit to retain each task's default")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.num_fewshot is not None and args.num_fewshot < 0:
        parser.error("--num-fewshot must be nonnegative")
    command = evaluation_command(
        args.model, args.tasks, args.output_dir, args.batch_size, args.num_fewshot
    )
    if args.dry_run:
        print(json.dumps(command, indent=2))
        return
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    environment = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (args.output_dir / "requirements.txt").write_text(environment)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

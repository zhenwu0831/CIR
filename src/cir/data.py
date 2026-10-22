from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


def alpaca_record(example: Mapping[str, Any]) -> dict[str, str]:
    response = example.get("response", example.get("output"))
    instruction = example.get("instruction")
    context = example.get("input", "")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Each record needs a nonempty instruction string")
    if not isinstance(response, str) or not response.strip():
        raise ValueError("Each record needs a nonempty response or output string")
    if not isinstance(context, str):
        raise ValueError("The input field must be a string")
    return {"instruction": instruction, "input": context, "output": response}


def write_dataset(
    rows: Iterable[Mapping[str, Any]], directory: Path, name: str, provenance: dict[str, Any]
) -> int:
    if not re.fullmatch(r"[\w#+.-]+", name) or name in {".", ".."}:
        raise ValueError("Dataset names must be filenames without path separators")
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{name}.json"
    registry = directory / "dataset_info.json"
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite {target}")
    info = json.loads(registry.read_text()) if registry.exists() else {}
    if name in info:
        raise ValueError(f"Dataset {name} is already registered")
    records = [alpaca_record(row) for row in rows]
    if not records:
        raise ValueError("Cannot register an empty dataset")
    content = json.dumps(records, ensure_ascii=False, indent=2) + "\n"
    target.write_text(content, encoding="utf-8")
    info[name] = {"file_name": target.name}
    registry.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    metadata = {
        **provenance,
        "num_samples": len(records),
        "sha256": hashlib.sha256(content.encode()).hexdigest(),
    }
    (directory / f"{name}.provenance.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    return len(records)

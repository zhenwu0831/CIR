
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Consolidate per-model CodeMMLU results into CSVs.

For each subfolder under a given ROOT (each representing one model configuration),
this script looks for files named like:
  <subtask>.test.metrics.json
extracts the "accuracy" field, computes "overall" as the mean across the found subtasks,
and writes a single-row CSV in that subfolder.

It can also write a combined CSV over all models.

Usage:
    python consolidate_results.py \
        --root /path/to/results \
        --write-combined                 # optional: writes all_models.csv at root
        --combined-name all_models.csv   # optional: custom name for combined CSV
        --strict                         # optional: if set, error on missing subtask files

Notes:
- The column order is fixed as:
    overall, api_frameworks, programming_syntax, software_principles,
    dbms_sql, others, code_completion, fill_in_the_middle,
    code_repair, execution_prediction
- "overall" is computed as the average of the available (non-missing) accuracies
  among the 9 subtasks. If --strict is used, all 9 must be present.
"""

import os
import json
import csv
import argparse
from typing import Dict, List, Optional

SUBTASKS = [
    "api_frameworks",
    "programming_syntax",
    "software_principles",
    "dbms_sql",
    "others",
    "code_completion",
    "fill_in_the_middle",
    "code_repair",
    "execution_prediction",
]

COLUMN_ORDER = ["overall"] + SUBTASKS


def read_accuracy(metrics_path: str) -> Optional[float]:
    """Read accuracy from a *.test.metrics.json file."""
    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Expect "accuracy" in the json
        acc = data.get("accuracy", None)
        if isinstance(acc, (int, float)):
            return float(acc)
        # Sometimes accuracy might be a string; try to cast
        if isinstance(acc, str):
            try:
                return float(acc)
            except ValueError:
                return None
        return None
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"[WARN] Failed reading {metrics_path}: {e}")
        return None


def consolidate_one_model(model_dir: str, strict: bool = False) -> Dict[str, Optional[float]]:
    """
    For a given model directory, collect accuracies for each subtask and compute overall.
    Returns a dict with keys matching COLUMN_ORDER.
    """
    row: Dict[str, Optional[float]] = {}

    # Read each subtask accuracy
    accuracies = []
    for task in SUBTASKS:
        filename = f"{task}.test.metrics.json"
        path = os.path.join(model_dir, filename)
        acc = read_accuracy(path)
        row[task] = acc
        if acc is not None:
            accuracies.append(acc)
        elif strict:
            raise FileNotFoundError(f"Missing or unreadable metrics for subtask '{task}' at: {path}")

    # Compute overall as mean of available accuracies
    if strict and len(accuracies) != len(SUBTASKS):
        raise ValueError(f"Strict mode: not all subtasks found in {model_dir}")
    row["overall"] = sum(accuracies) / len(accuracies) if accuracies else None
    return row


def write_single_csv(model_dir: str, model_name: str, row: Dict[str, Optional[float]], outfile_name: str = "summary.csv") -> str:
    """
    Write a one-row CSV in the model directory with accuracies in COLUMN_ORDER.
    Returns the path to the written CSV.
    """
    out_path = os.path.join(model_dir, outfile_name)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model"] + COLUMN_ORDER)
        writer.writerow([model_name] + [row.get(col, "") if row.get(col, None) is not None else "" for col in COLUMN_ORDER])
    return out_path


def write_combined_csv(root: str, table: List[Dict[str, Optional[float]]], model_names: List[str], combined_name: str) -> str:
    """
    Write a multi-row CSV at the root folder combining all models.
    """
    out_path = os.path.join(root, combined_name)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model"] + COLUMN_ORDER)
        for model_name, row in zip(model_names, table):
            writer.writerow([model_name] + [row.get(col, "") if row.get(col, None) is not None else "" for col in COLUMN_ORDER])
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Consolidate per-model CodeMMLU results into CSVs.")
    parser.add_argument("--root", required=True, help="Path containing subfolders, each a model configuration.")
    parser.add_argument("--write-combined", action="store_true", help="Also write a combined CSV at ROOT.")
    parser.add_argument("--combined-name", default="all_models.csv", help="Filename for the combined CSV (at ROOT).")
    parser.add_argument("--strict", action="store_true", help="Require all subtasks to exist; otherwise error.")
    parser.add_argument("--per-model-name", default="summary.csv", help="Filename for per-model CSV written inside each model folder.")
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        raise NotADirectoryError(f"Root path does not exist or is not a directory: {root}")

    # Find immediate subdirectories (model configurations)
    subdirs = sorted([
        os.path.join(root, d) for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    ])

    if not subdirs:
        print(f"[WARN] No subdirectories found under: {root}")

    combined_rows: List[Dict[str, Optional[float]]] = []
    model_names: List[str] = []

    for model_dir in subdirs:
        model_name = os.path.basename(model_dir.rstrip(os.sep))

        row = consolidate_one_model(model_dir, strict=args.strict)
        csv_path = write_single_csv(model_dir, model_name, row, outfile_name=args.per_model_name)
        print(f"[OK] Wrote {csv_path}")

        combined_rows.append(row)
        model_names.append(model_name)

    if args.write_combined and combined_rows:
        combined_path = write_combined_csv(root, combined_rows, model_names, args.combined_name)
        print(f"[OK] Wrote combined CSV: {combined_path}")


if __name__ == "__main__":
    main()

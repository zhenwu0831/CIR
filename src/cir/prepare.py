from __future__ import annotations

import argparse
import json
from pathlib import Path

from cir.data import write_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Register a dataset for the released training configs"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source", help="Hugging Face dataset ID")
    source.add_argument(
        "--local-file", type=Path, help="JSON array or JSONL of instruction/response pairs"
    )
    parser.add_argument("--name", required=True, help="Exact dataset key used in the training YAML")
    parser.add_argument("--revision", help="Hub commit or tag; pin a commit for reproducibility")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--language", help="Filter the prefix of the original id, e.g. Python or C++"
    )
    parser.add_argument("--shuffle-count", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.shuffle_count < 0:
        parser.error("--shuffle-count must be nonnegative")
    if args.local_file:
        if args.language or args.shuffle_count:
            parser.error("Language filtering and shuffling require --source")
        content = args.local_file.read_text(encoding="utf-8")
        rows = (
            [json.loads(line) for line in content.splitlines() if line.strip()]
            if args.local_file.suffix == ".jsonl"
            else json.loads(content)
        )
        provenance = {"source": args.local_file.name, "format": "local"}
    else:
        from datasets import load_dataset

        rows = load_dataset(args.source, split=args.split, revision=args.revision)
        for _ in range(args.shuffle_count):
            rows = rows.shuffle(seed=args.seed)
        if args.language:
            rows = rows.filter(lambda row: row["id"].split("_", 1)[0] == args.language)
        provenance = {
            "source": args.source,
            "revision": args.revision,
            "split": args.split,
            "fingerprint": rows._fingerprint,
            "language": args.language,
            "shuffle_count": args.shuffle_count,
            "seed": args.seed,
        }
    count = write_dataset(rows, args.output_dir, args.name, provenance)
    print(f"Registered {args.name}: {count} examples")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

from cir.data import write_dataset


def mixture_counts(total: int, nl_percent: float) -> tuple[int, int]:
    if total <= 0 or not 0 <= nl_percent <= 100:
        raise ValueError("Total must be positive and NL percent must be between 0 and 100")
    nl_count = int(round(total * nl_percent / 100))
    return total - nl_count, nl_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare code/NL instruction mixtures")
    parser.add_argument("--code-source", default="CodeAtCMU/CodeInstructions-120K")
    parser.add_argument("--nl-source", default="CodeAtCMU/NaturalLanguageInstructions-120K")
    parser.add_argument("--code-revision")
    parser.add_argument("--nl-revision")
    parser.add_argument("--nl-percent", type=float, required=True)
    parser.add_argument("--total", type=int, default=120000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Use format_code_data.py's 50/50 seed and ordering instead of data_mixing.py",
    )
    args = parser.parse_args()
    code_count, nl_count = mixture_counts(args.total, args.nl_percent)
    if args.baseline and args.nl_percent != 50:
        parser.error("--baseline requires --nl-percent 50")
    from datasets import concatenate_datasets, load_dataset

    code = load_dataset(args.code_source, revision=args.code_revision, split="train")
    nl = load_dataset(args.nl_source, revision=args.nl_revision, split="train")
    if code_count > len(code) or nl_count > len(nl):
        parser.error("The requested mixture exceeds the available source examples")
    seed = args.seed if args.baseline else args.seed + int(args.nl_percent * 10)

    # Mixture training uses instruction/response pairs without auxiliary input fields.
    def normalize(row: dict) -> dict[str, str]:
        return {"instruction": row["instruction"], "response": row["response"]}

    code_part = code.shuffle(seed=seed).select(range(code_count))
    nl_part = nl.shuffle(seed=seed).select(range(nl_count))
    code_part = code_part.map(normalize, remove_columns=code_part.column_names)
    nl_part = nl_part.map(normalize, remove_columns=nl_part.column_names)
    mixed = concatenate_datasets([code_part, nl_part])
    if not args.baseline:
        mixed = mixed.shuffle(seed=args.seed)
    write_dataset(
        mixed,
        args.output_dir,
        args.name,
        {
            "code_source": args.code_source,
            "nl_source": args.nl_source,
            "code_revision": args.code_revision,
            "nl_revision": args.nl_revision,
            "code_fingerprint": code._fingerprint,
            "nl_fingerprint": nl._fingerprint,
            "nl_percent": args.nl_percent,
            "seed": args.seed,
            "sampling_seed": seed,
            "baseline": args.baseline,
        },
    )


if __name__ == "__main__":
    main()

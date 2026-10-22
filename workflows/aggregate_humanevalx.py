import csv
import json
import statistics
from pathlib import Path


def summarize(records: dict) -> dict:
    scores = [record["score"] for record in records.values()]
    return {
        "count": len(scores),
        "invalid": sum(score == -1 for score in scores),
        "mean": statistics.mean(scores) if scores else "",
        "std": statistics.stdev(scores) if len(scores) > 1 else 0.0,
    }


def main(args) -> None:
    rows = []
    for path in sorted(Path(args.input_dir).glob("*/*.json")):
        rows.append(
            {
                "model": path.parent.name,
                "language": path.stem,
                **summarize(json.loads(path.read_text())),
            }
        )
    with open(args.output_file, "w", newline="") as output:
        writer = csv.DictWriter(
            output, fieldnames=["model", "language", "count", "invalid", "mean", "std"]
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Summarize instance scores, including legacy -1 failures."
    )
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_file", required=True)
    main(parser.parse_args())

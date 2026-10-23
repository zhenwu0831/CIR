from __future__ import annotations

import argparse
import json
import random
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from cir.data import write_dataset

IDENTIFIER = r"\b([a-zA-Z_][a-zA-Z0-9_]*)\b"
C_LIKE = {"single": "//", "multi": ("/*", "*/"), "var": IDENTIFIER}

# Example ids start with the language, e.g. "Python_6819". The original scripts keyed
# "C#" as "CSharp" and had no entry for "C++", so C++ responses pass through every
# perturbation unchanged, as in the released datasets.
C_KEYWORDS = "int float char void return if else for while switch case break continue"
LANGUAGES: dict[str, dict[str, Any]] = {
    "Python": {
        "single": "#",
        "multi": ("'''", "'''"),
        "var": IDENTIFIER,
        "keywords": (
            "def return if else elif while for try except import from class with as pass "
            "break continue"
        ).split(),
    },
    "C": {**C_LIKE, "keywords": C_KEYWORDS.split()},
    "Java": {
        **C_LIKE,
        "keywords": (
            f"{C_KEYWORDS} package import class interface public private protected"
        ).split(),
    },
    "CSharp": {
        **C_LIKE,
        "keywords": f"{C_KEYWORDS} namespace class public private protected".split(),
    },
    "JavaScript": {
        **C_LIKE,
        "keywords": (
            "function return if else for while var let const switch case break continue"
        ).split(),
    },
    "TypeScript": {
        **C_LIKE,
        "keywords": (
            "function return if else for while var let const interface type implements extends"
        ).split(),
    },
    "PHP": {
        **C_LIKE,
        "var": r"\$[a-zA-Z_][a-zA-Z0-9_]*",
        "keywords": "function if else elseif foreach while return echo isset unset".split(),
    },
    "Rust": {
        **C_LIKE,
        "keywords": "fn let mut pub impl match if else while loop return".split(),
    },
    "Go": {
        **C_LIKE,
        "keywords": "func package import return if else for switch case break continue".split(),
    },
}

NONSENSE_EN = "foo bar baz qux quux corge grault".split()
NONSENSE_NONEN = (
    "amigo makan kissa wasser fleur xiexie chleb hund sol nuit vino fjord honua lune tårta "
    "selva tokka amici cielo brot"
).split()

# Identifiers not renamed by replace_variables even when they look like declarations.
RESERVED = set(
    (
        "int float char return if else for while void long double true false const static "
        "public private protected class struct switch case break continue do new try catch "
        "import package namespace using include sizeof main def print None let var function "
        "async await match fn pub impl use self this"
    ).split()
)

FENCE = re.compile(r"(```(?:\w*\n)?[\s\S]*?```)")
FENCE_CONTENT = re.compile(r"```(\w*\n)?([\s\S]*?)```")
# The keyword and comment perturbations were generated with doubly escaped patterns.
# This fence never matches ordinary code blocks, so those perturbations treated the
# whole response as code, and a response containing a degenerate fence such as
# `````` was left unchanged. Preserved so that the released datasets reproduce.
DEGENERATE_FENCE = re.compile(r"```(?:\\w*\\n)?[\\s\\S]*?```")
STRING = re.compile(r'"[^"\\]*(?:\\.[^"\\]*)*"|\'[^\'\\]*(?:\\.[^\'\\]*)*\'')
PYTHON_ASSIGNMENT = re.compile(r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=", re.MULTILINE)
C_DECLARATION = re.compile(
    r"^\s*(?:struct|const)?\s*[\w_<>]+\s*(\*+)?\s*([a-zA-Z_][a-zA-Z0-9_]*)(\s*=[^;]*)?;",
    re.MULTILINE,
)

PERTURBATIONS = (
    "remove_comments",
    "remove_whitespace",
    "replace_variables",
    "replace_keywords_en",
    "replace_keywords_nonen",
    "swap_comments_local",
    "replace_comments_global",
)
RANDOM_PERTURBATIONS = PERTURBATIONS[3:]


def language_of(example_id: str) -> str:
    return example_id.split("_")[0].replace("C#", "CSharp")


def comment_spans(code: str, rules: Mapping[str, Any]) -> list[tuple[int, int]]:
    pattern = re.escape(rules["single"]) + ".*"
    return [(match.start(), match.end()) for match in re.finditer(pattern, code)]


def extract_comments(code: str, rules: Mapping[str, Any]) -> list[str]:
    return re.findall(re.escape(rules["single"]) + ".*", code)


def remove_comments(code: str, rules: Mapping[str, Any]) -> str:
    code = re.sub(re.escape(rules["single"]) + r".*?$", "", code, flags=re.MULTILINE)
    start, end = (re.escape(marker) for marker in rules["multi"])
    return re.sub(rf"{start}.*?{end}", "", code, flags=re.DOTALL)


def remove_whitespace(code: str) -> str:
    return re.sub(r"\s+", "", code)


def replace_variables(code: str, rules: Mapping[str, Any]) -> str:
    strings = [(match.start(), match.end()) for match in STRING.finditer(code)]

    def in_string(position: int) -> bool:
        return any(start <= position < end for start, end in strings)

    if rules["single"] == "#":
        declared = {
            match.group(1)
            for match in PYTHON_ASSIGNMENT.finditer(code)
            if not in_string(match.start(1))
        }
    else:
        declared = {match.group(2) for match in C_DECLARATION.finditer(code)}
    mapping = {name: f"var_{i}" for i, name in enumerate(sorted(declared - RESERVED))}

    def rename(match: re.Match[str]) -> str:
        name = match.group(0)
        if name in mapping and not in_string(match.start()):
            return mapping[name]
        return name

    return re.sub(rf"\b{rules['var']}\b", rename, code)


def replace_keywords(
    code: str, rules: Mapping[str, Any], words: list[str], rng: random.Random
) -> str:
    # Spans are measured once on the input; later substitutions shift positions. The
    # lookarounds exclude a literal "\w" rather than word characters, so keywords also
    # match inside identifiers and prose, case-insensitively. Both match the release.
    spans = comment_spans(code, rules)

    def in_comment(position: int) -> bool:
        return any(start <= position < end for start, end in spans)

    for keyword in rules["keywords"]:

        def replacement(match: re.Match[str]) -> str:
            if in_comment(match.start()):
                return match.group(0)
            return rng.choice(words)

        code = re.sub(rf"(?<!\\w){keyword}(?!\\w)", replacement, code, flags=re.IGNORECASE)
    return code


def swap_comments(code: str, rules: Mapping[str, Any], rng: random.Random) -> str:
    comments = extract_comments(code, rules)
    if len(set(comments)) < 2:
        return code
    shuffled = comments[:]
    for _ in range(5):
        rng.shuffle(shuffled)
        if shuffled != comments:
            break
    else:
        return code
    for original, new in zip(comments, shuffled, strict=True):
        code = code.replace(original, new, 1)
    return code


def replace_comments_global(
    code: str, rules: Mapping[str, Any], pool: list[str], rng: random.Random
) -> str:
    if not pool:
        return code
    for comment in extract_comments(code, rules):
        code = code.replace(comment, rng.choice(pool), 1)
    return code


def comment_pool(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[str]]:
    pool = {}
    for row in rows:
        rules = LANGUAGES.get(language_of(row["id"]))
        if rules is None:
            continue
        comments = extract_comments(row["response"], rules)
        if comments:
            pool[row["id"]] = comments
    return pool


def perturb_code_blocks(response: str, rules: Mapping[str, Any], perturbation: str) -> str:
    parts = []
    for part in FENCE.split(response):
        match = FENCE_CONTENT.match(part) if part.startswith("```") else None
        if match and part.endswith("```"):
            prefix, code = match.group(1) or "", match.group(2)
            if perturbation == "remove_comments":
                code = remove_comments(code, rules)
            elif perturbation == "remove_whitespace":
                code = remove_whitespace(code)
            else:
                code = replace_variables(code, rules)
            part = f"```{prefix}{code}```"
        parts.append(part)
    return "".join(parts)


def perturb_response(
    response: str,
    language: str,
    perturbation: str,
    rng: random.Random | None = None,
    pool: list[str] | None = None,
) -> str:
    if perturbation not in PERTURBATIONS:
        raise ValueError(f"Unknown perturbation {perturbation}")
    rules = LANGUAGES.get(language)
    if rules is None:
        return response
    if perturbation not in RANDOM_PERTURBATIONS:
        return perturb_code_blocks(response, rules, perturbation)
    if rng is None:
        raise ValueError(f"{perturbation} needs a random generator")
    if DEGENERATE_FENCE.search(response):
        return response
    if perturbation == "replace_keywords_en":
        return replace_keywords(response, rules, NONSENSE_EN, rng)
    if perturbation == "replace_keywords_nonen":
        return replace_keywords(response, rules, NONSENSE_NONEN, rng)
    if perturbation == "swap_comments_local":
        return swap_comments(response, rules, rng)
    return replace_comments_global(response, rules, pool or [], rng)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply a rule-based perturbation to code instruction responses"
    )
    parser.add_argument("--source", default="CodeAtCMU/CodeInstructions-120K")
    parser.add_argument("--revision", help="Hub commit or tag; pin a commit for reproducibility")
    parser.add_argument("--split", default="train")
    parser.add_argument("--perturbation", choices=PERTURBATIONS, required=True)
    parser.add_argument("--name", required=True, help="Exact dataset key used in the training YAML")
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    from datasets import load_dataset

    rows = load_dataset(args.source, split=args.split, revision=args.revision)
    rng = random.Random(args.seed)
    pool: list[str] = []
    if args.perturbation == "replace_comments_global":
        comments = comment_pool(rows)
        pool = [comment for found in comments.values() for comment in found]
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / f"{args.name}.comments.json").write_text(
            json.dumps(comments, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    changed = 0
    records = []
    for row in rows:
        response = perturb_response(
            row["response"], language_of(row["id"]), args.perturbation, rng, pool
        )
        changed += response != row["response"]
        records.append({**row, "response": response})
    count = write_dataset(
        records,
        args.output_dir,
        args.name,
        {
            "source": args.source,
            "revision": args.revision,
            "split": args.split,
            "fingerprint": rows._fingerprint,
            "perturbation": args.perturbation,
            "seed": args.seed if args.perturbation in RANDOM_PERTURBATIONS else None,
            "num_changed": changed,
        },
    )
    print(f"Registered {args.name}: {count} examples, {changed} changed")


if __name__ == "__main__":
    main()

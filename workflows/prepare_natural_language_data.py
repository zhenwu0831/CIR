import argparse
from pathlib import Path

import datasets
from tqdm import tqdm


def main(args) -> None:
    dataset = datasets.load_dataset("teknium/OpenHermes-2.5", split="train")

    dataset = dataset.filter(
        lambda example: example["category"] not in ["coding", "agent", "summarization"], num_proc=16
    )

    non_english_languages = [
        "Chinese",
        "Hawaiian",
        "German",
        "Luxembourgish",
        "Slovak",
        "Ukrainian",
        "Vietnamese",
        "Serbian",
        "Indonesian",
        "Slovenian",
        "Esperanto",
        "Aymara",
        "Korean",
        "Latvian",
        "Croatian",
        "unknown",
        "Spanish",
        "Galician",
        "Dutch",
        "Swedish",
        "Catalan",
        "Japanese",
        "Hebrew",
        "Polish",
        "Czech",
        "Persian",
        "Hmong",
        "Italian",
        "Guarani",
        "Romanian",
        "Latin",
        "Norwegian",
        "Hungarian",
        "Finnish",
        "French",
        "Quechua",
        "Shona",
        "Turkish",
        "Thai",
        "Scots",
        "Russian",
        "Portuguese",
        "Danish",
    ]

    dataset = dataset.filter(
        lambda example: example["language"] not in non_english_languages, num_proc=16
    )

    print(dataset)

    examples = []

    for idx, example in tqdm(enumerate(dataset), total=len(dataset)):
        # example is formatted as = [{""from": "human/gpt", value: "text"}, {"from": "gpt/human", value: "text"}]
        data = {}

        for item in example["conversations"]:
            if item["from"] == "human":
                data["instruction"] = item["value"]
            elif item["from"] == "gpt":
                data["response"] = item["value"]

        data["id"] = (
            "teknium_OpenHermes-2.5_"
            + str(example["source"])
            + "_"
            + str(example["category"])
            + "_"
            + str(idx)
        )
        examples.append(data)

    ds = datasets.Dataset.from_list(examples)

    ds = ds.filter(
        lambda example: (
            "instruction" in example
            and "response" in example
            and example["instruction"]
            and example["response"]
        ),
        num_proc=16,
    )

    ds = ds.shuffle(seed=42).select(range(120000))

    dataset = datasets.DatasetDict({"train": ds.shuffle(seed=42)})

    dataset.save_to_disk(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--languages_fp", default=str(Path(__file__).with_name("languages.txt")))
    main(parser.parse_args())

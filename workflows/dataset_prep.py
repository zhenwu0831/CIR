import argparse
import hashlib
from functools import partial
from pathlib import Path

import datasets


def main(args) -> None:
    def generate_6_digit_hash(text):
        # Encode the text as a byte string
        text_bytes = text.encode("utf-8")

        # Use SHA-256 to hash the text
        hash_object = hashlib.sha256(text_bytes)

        # Get the hexadecimal representation of the hash
        hex_hash = hash_object.hexdigest()

        # Take the first 6 digits of the hash
        six_digit_hash = hex_hash[:6]

        return six_digit_hash

    programming_language = open(args.languages_fp).readlines()

    programming_language = [lang.strip() for lang in programming_language]

    programming_language_lower = [lang.lower() for lang in programming_language]

    programming_language_upper = [lang.upper() for lang in programming_language]

    addition_keywords = ["webpage", "web page", "site", "website", "SQL", "HTML"]

    def update_ids(prefix, example):

        example["id"] = prefix + "_" + example["id"]
        return example

    def get_instruction(example):
        example["instruction"] = "# Problem\n" + "\n".join(example["prompt"].split("# Problem")[1:])
        return example

    code_forces_dataset_name = "open-r1/codeforces-cots"

    update_id_fn = partial(update_ids, code_forces_dataset_name)

    code_forces = datasets.load_dataset(code_forces_dataset_name, split="train")

    code_forces = code_forces.map(update_id_fn)

    code_forces = code_forces.filter(lambda example: len(example["prompt"].split()) > 10)

    code_forces = code_forces.map(get_instruction)

    code_forces = code_forces.remove_columns(
        [col for col in code_forces.column_names if col not in ["id", "instruction"]]
    )

    other_dataset = {
        "TokenBender/code_instructions_122k_alpaca_style": {
            "instruction_column": "instruction",
            "id_column": None,
            "input_column": "input",
        },
        "nickrosh/Evol-Instruct-Code-80k-v1": {
            "instruction_column": "instruction",
            "id_column": None,
            "input_column": None,
        },
        "red1xe/code_instructions": {
            "instruction_column": "instruction",
            "id_column": None,
            "input_column": "input",
        },
        "AtlasUnified/Code-Instruct-Sets": {
            "instruction_column": "instruction",
            "id_column": "id",
            "input_column": None,
        },
        "rombodawg/code_instruct_alpaca_vicuna_wizardlm_56k_backup": {
            "instruction_column": "prompt",
            "id_column": None,
            "input_column": None,
        },
    }

    def add_input_to_instruction(example, instruction_column, input_column):
        if example[input_column]:
            example["instruction"] = (
                example[instruction_column] + "\nInput:\n" + example[input_column]
            )
        return example

    all_other_datasets = []

    for dataset_name in other_dataset:
        dataset = datasets.load_dataset(dataset_name, split="train")

        dataset = dataset.filter(
            lambda example, dataset_name=dataset_name: (
                not any(
                    [
                        lang in example[other_dataset[dataset_name]["instruction_column"]]
                        for lang in programming_language_lower
                    ]
                )
            )
        )
        dataset = dataset.filter(
            lambda example, dataset_name=dataset_name: (
                not any(
                    [
                        lang in example[other_dataset[dataset_name]["instruction_column"]]
                        for lang in programming_language_upper
                    ]
                )
            )
        )
        dataset = dataset.filter(
            lambda example, dataset_name=dataset_name: (
                not any(
                    [
                        lang in example[other_dataset[dataset_name]["instruction_column"]]
                        for lang in programming_language
                    ]
                )
            )
        )
        dataset = dataset.filter(
            lambda example, dataset_name=dataset_name: (
                not any(
                    [
                        lang in example[other_dataset[dataset_name]["instruction_column"]]
                        for lang in addition_keywords
                    ]
                )
            )
        )

        # import pdb; pdb.set_trace()
        # if input column is present then add it to the instruction column
        if other_dataset[dataset_name]["input_column"] is not None:
            add_input_fn = partial(
                add_input_to_instruction,
                instruction_column=other_dataset[dataset_name]["instruction_column"],
                input_column=other_dataset[dataset_name]["input_column"],
            )

            dataset = dataset.map(add_input_fn)

        # id is not present in the dataset - then create a unique id for each example
        if other_dataset[dataset_name]["id_column"] is None:
            ids = [
                f"{dataset_name}_"
                + generate_6_digit_hash(example[other_dataset[dataset_name]["instruction_column"]])
                for example in dataset
            ]
            dataset = dataset.add_column("id", ids)
        else:
            update_ids_fn = partial(update_ids, dataset_name)
            dataset = dataset.map(update_ids_fn)

        # if instruction column name is not instruction then rename it to instruction
        if other_dataset[dataset_name]["instruction_column"] != "instruction":
            dataset = dataset.rename_column(
                other_dataset[dataset_name]["instruction_column"], "instruction"
            )

        # remove all columns except instruction and id
        dataset = dataset.remove_columns(
            [col for col in dataset.column_names if col not in ["id", "instruction"]]
        )

        print(f"Processing {dataset_name}")
        print(f"Columns: {dataset.column_names}")

        if len(all_other_datasets) == 0:
            all_other_datasets.append(dataset)
        else:
            current_dataset = dataset.cast(all_other_datasets[0].features)
            all_other_datasets.append(current_dataset)

    other_datasets = datasets.concatenate_datasets(all_other_datasets)

    other_datasets = other_datasets.filter(lambda example: len(example["instruction"].split()) > 10)

    all_datasets = datasets.concatenate_datasets([code_forces, other_datasets])

    all_datasets = all_datasets.shuffle(seed=42).shuffle(seed=42).shuffle(seed=42)

    all_datasets.save_to_disk(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--languages_fp", default=str(Path(__file__).with_name("languages.txt")))
    main(parser.parse_args())

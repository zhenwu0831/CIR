import argparse
import random
from pathlib import Path

import datasets
import pandas as pd
from tqdm import tqdm


def main(args) -> None:
    dataset = datasets.load_dataset("CodeAtCMU/CodeInstructions-12K")

    print(dataset)

    languages = open(args.languages_fp).read().splitlines()[:10]

    templates = [
        "Generate the code in {language}.",
        "Write the code in {language}.",
        "Create the code using {language}.",
        "Produce a code snippet in {language}.",
        "Generate a solution in {language}.",
        "Implement the code in {language}.",
        "Compose the code in {language}.",
        "Construct the code using {language}.",
        "Write a program in {language}.",
        "Provide code in {language}.",
        "Build the code using {language}.",
        "Draft the code in {language}.",
        "Develop the code using {language}.",
        "Create a script in {language}.",
        "Design the code in {language}.",
        "Format the code in {language}.",
        "Prepare a code snippet in {language}.",
        "Write a function in {language}.",
        "Deliver the code in {language}.",
        "Generate a script using {language}.",
    ]

    all_data_df = pd.DataFrame(columns=["id", "instruction", "response"])

    for i, example in tqdm(enumerate(dataset["train"])):
        # get the code and language

        input = example["instruction"]

        for language in languages:
            response = example[language]

            # select a random template
            template = random.choice(templates)

            # format the prompt
            prompt = template.format(language=language)

            instruction = f"{input} {prompt}"

            idx = f"{language}_{i}"

            # append the data to the dataframe
            all_data_df = all_data_df._append(
                {"id": idx, "instruction": instruction, "response": response}, ignore_index=True
            )

    dataset = datasets.Dataset.from_pandas(all_data_df)

    dataset = dataset.shuffle(seed=42).select(range(120000))

    dataset.save_to_disk(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--languages_fp", default=str(Path(__file__).with_name("languages.txt")))
    main(parser.parse_args())

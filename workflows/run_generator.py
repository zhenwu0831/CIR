# This is to rate problem based on difficulty level on scale of 1-10. 1 is the easiest and 10 is the hardest.
# Similar implementaiton - https://github.com/macabdul9/SkyThought/blob/main/skythought/tools/label_math_difficulty.py
# Maybe we can take this implementation directly.
import json
import os

import datasets
from tqdm import tqdm
from utils import create_llm_payload, format_message, generator


def main(args):

    # Load the dataset
    dataset = datasets.load_dataset(path=args.dataset_name_or_path, split=args.split)

    # create the the solver and verifier object -- please note that verification model is fixed to gpt-4o-mini

    # Create the output directory if it does not exist
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    # Save the output in a JSON file
    filename = f"{args.output_dir}/{args.dataset_name_or_path.split('/')[-1]}_instructions.json"

    outputs = {}

    # if the file already exists, then load the file and start from the last checkpoint
    if os.path.exists(filename):
        with open(filename) as f:
            outputs = json.load(f)

    # Load prompt template from .txt file
    with open(args.prompt_fp) as f:
        prompt_template = f.read()

    # Load programming languages from .txt file
    with open(args.languages_fp) as f:
        languages = f.readlines()

    for idx, example in tqdm(enumerate(dataset), total=len(dataset)):
        # get the prompt

        output = {}

        # store instruction first
        output["instruction"] = example[args.instruction_column]

        # Iterate over each language
        for language in languages:
            prompt = prompt_template.format(
                instruction=example[args.instruction_column], language=language
            )

            payload = create_llm_payload(format_message([], "user", prompt))

            # generate the response
            try:
                response = generator(payload=payload)
            except Exception:
                response = "invalid"

            # if response=="invalid" then break the loop and move to the next example
            if response == "invalid":
                break

            output[language] = response

        # Save the output
        outputs[example[args.id_column]] = output

        # import pdb; pdb.set_trace()
        # checkpoint the output bro -- in case something goes wrong so that we can start from the last checkpoint
        if idx % args.checkpoint_steps == 0:
            with open(filename, "w") as f:
                json.dump(outputs, f, indent=4)

    # Save the final experiment output
    with open(filename, "w") as f:
        json.dump(outputs, f, indent=4)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Rate the problems based on difficulty level.")

    # dataset
    parser.add_argument(
        "--dataset_name_or_path",
        type=str,
        default="CodeAtCMU/CodeInstructions",
        help="Dataset name or path to use for solving the problems. Default is macabdul9/hle_text_only.",
    )
    parser.add_argument(
        "--prompt_fp",
        type=str,
        default="prompts/gen_prompt.txt",
        help="File path to the prompt template. Default is prompt.txt.",
    )
    parser.add_argument(
        "--languages_fp",
        type=str,
        default="languages.txt",
        help="File path to the programming languages. Default is languages.txt.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use for solving the problems. Default is test.",
    )
    parser.add_argument(
        "--id_column", type=str, default="id", help="Column name for the id. Default is None."
    )
    parser.add_argument(
        "--instruction_column",
        type=str,
        default="instruction",
        help="Column name for the question. Default is None.",
    )

    parser.add_argument(
        "--checkpoint_steps",
        type=int,
        default=10,
        help="Checkpoint step to save the model. Default is 1000.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="generation",
        help="Output directory to save the results. Default is current directory.",
    )

    args = parser.parse_args()

    main(args)

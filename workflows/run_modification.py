import asyncio
import json
import os

import datasets
from openai import AsyncOpenAI
from tqdm import tqdm

client = None

semaphore = asyncio.Semaphore(500)  # Max concurrent requests


async def generate_response(prompt, retries=3, backoff=2):
    async with semaphore:
        for attempt in range(retries):
            try:
                response = await client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.6,
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                if attempt < retries - 1:
                    await asyncio.sleep(backoff * (2**attempt))
                else:
                    return f"[ERROR] {str(e)}"


async def main(args):
    global client
    client = AsyncOpenAI()
    # Load the dataset
    dataset = datasets.load_dataset(path=args.dataset_name_or_path, split=args.split)

    # Create the output directory if it does not exist
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    # Save the output in a JSON file
    filename = f"{args.output_dir}/{args.dataset_name_or_path.split('/')[-1]}_instructions_{args.prompt_fp.split('/')[-1].split('.')[0]}.json"

    # if the file already exists, then load the file and start from the last checkpoint
    if os.path.exists(filename):
        with open(filename) as f:
            outputs = json.load(f)
    else:
        outputs = {}

    # Load prompt template from .txt file
    with open(args.prompt_fp) as f:
        prompt_template = f.read()

    tasks = []
    id_to_example = {}
    for _idx, example in tqdm(enumerate(dataset), total=len(dataset)):
        example_id = example[args.id_column]
        if example_id in outputs:
            continue

        # prompt = prompt_template.format(
        #     instruction=example[args.instruction_column],
        #     response=example[args.response_column],
        # )
        # instead of format use replace to avoid issues with curly braces
        prompt = prompt_template.replace("{instruction}", example[args.instruction_column]).replace(
            "{response}", example[args.response_column]
        )

        task = asyncio.create_task(generate_response(prompt))
        tasks.append(task)
        id_to_example[example_id] = example

        if len(tasks) == 500:
            results = await asyncio.gather(*tasks)
            for i, (example_id, ex) in enumerate(id_to_example.items()):
                outputs[example_id] = {
                    "instruction": ex[args.instruction_column],
                    "old_response": ex[args.response_column],
                    "new_response": results[i],
                    "setting": args.prompt_fp.split("/")[-1].split(".")[0],
                }
            with open(filename, "w") as f:
                json.dump(outputs, f, indent=4)
            tasks = []
            id_to_example = {}

    # Final batch
    if tasks:
        results = await asyncio.gather(*tasks)
        for i, (example_id, ex) in enumerate(id_to_example.items()):
            outputs[example_id] = {
                "instruction": ex[args.instruction_column],
                "old_response": ex[args.response_column],
                "new_response": results[i],
                "setting": args.prompt_fp.split("/")[-1].split(".")[0],
            }
        with open(filename, "w") as f:
            json.dump(outputs, f, indent=4)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Rate the problems based on difficulty level.")

    # dataset
    parser.add_argument(
        "--dataset_name_or_path",
        type=str,
        default="CodeAtCMU/CodeInstructions-120K",
        help="Dataset name or path to use for solving the problems. Default is macabdul9/hle_text_only.",
    )
    parser.add_argument(
        "--prompt_fp",
        type=str,
        default="prompts/good_code.txt",
        help="File path to the prompt template. Default is prompt.txt.",
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
        "--response_column",
        type=str,
        default="response",
        help="Column name for the response. Default is None.",
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
        default="code-modification",
        help="Output directory to save the results. Default is current directory.",
    )

    args = parser.parse_args()

    # Run the main function
    asyncio.run(main(args))

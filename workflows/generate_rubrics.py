import asyncio
import json
import os

import datasets
from openai import AsyncOpenAI
from tqdm import tqdm

client = None

CUNCURRENT_REQUESTS = 164  # Max concurrent requests

semaphore = asyncio.Semaphore(CUNCURRENT_REQUESTS)  # Max concurrent requests


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

    # Create the output directory if it does not exist
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    languages = ["python", "cpp", "go", "java", "js"]

    # iterate over languages
    for lang in languages:
        # Load the dataset
        dataset = datasets.load_dataset(args.dataset_name_or_path, lang, split=args.split)

        # Save the output in a JSON file
        filename = f"{args.output_dir}/{lang}.json"

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

            prompt = prompt_template.format(
                code_prompt=example[args.code_prompt_column],
                test_case=example[args.test_case_column],
                canonical_solution=example.get(args.canonical_solution_column, "N/A"),
            )

            task = asyncio.create_task(generate_response(prompt))
            tasks.append(task)
            id_to_example[example_id] = example

            if len(tasks) == CUNCURRENT_REQUESTS:
                results = await asyncio.gather(*tasks)
                for i, (example_id, ex) in enumerate(id_to_example.items()):
                    # import pdb; pdb.set_trace()
                    outputs[example_id] = {
                        "code_prompt": ex[args.code_prompt_column],
                        "test_cases": ex[args.test_case_column],
                        "canonical_solution": ex[args.canonical_solution_column],
                        "rubric": results[i],
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
                    "code_prompt": ex[args.code_prompt_column],
                    "test_cases": ex[args.test_case_column],
                    "canonical_solution": ex[args.canonical_solution_column],
                    "rubric": results[i],
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
        default="zai-org/humaneval-x",
        help="Dataset name or path to use for solving the problems. Default is macabdul9/hle_text_only.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to use for solving the problems. Default is test.",
    )

    parser.add_argument(
        "--prompt_fp",
        type=str,
        default="prompts/gen_rubric.txt",
        help="File path to the prompt template. Default is prompt.txt.",
    )
    parser.add_argument(
        "--id_column", type=str, default="task_id", help="Column name for the id. Default is None."
    )
    parser.add_argument(
        "--code_prompt_column",
        type=str,
        default="prompt",
        help="Column name for the question. Default is None.",
    )
    parser.add_argument(
        "--test_case_column",
        type=str,
        default="test",
        help="Column name for the response. Default is None.",
    )
    parser.add_argument(
        "--canonical_solution_column",
        type=str,
        default="canonical_solution",
        help="Column name for the canonical solution. Default is None.",
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
        default="rubrics",
        help="Output directory to save the results. Default is current directory.",
    )

    args = parser.parse_args()

    # Run the main function
    asyncio.run(main(args))

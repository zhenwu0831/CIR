import asyncio
import json
import os
import re

import datasets
from openai import AsyncOpenAI
from tqdm import tqdm

client = None

CUNCURRENT_REQUESTS = 32  # Max concurrent requests

semaphore = asyncio.Semaphore(CUNCURRENT_REQUESTS)  # Max concurrent requests


def extract_reasoning(output):
    reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", output, re.DOTALL)
    score_match = re.search(r"<score>(\d+)</score>", output)

    reasoning = reasoning_match.group(1).strip() if reasoning_match else "No reasoning found."
    score = int(score_match.group(1)) if score_match else -1

    return {"reasoning": reasoning, "score": score}


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
                    temperature=0.0,
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

    # load the rubrics
    rubrics = {}
    for lang in languages:
        rubric_fp = os.path.join(args.rubrics_dir, f"{lang}.json")
        with open(rubric_fp) as f:
            rubrics[lang] = json.load(f)

    # load the prompt template
    # Load prompt template from .txt file
    with open(args.prompt_fp) as f:
        prompt_template = f.read()

    # read the subdirs -- each subdir is a model
    models = os.listdir(args.model_output_dir)

    # iterate over models
    for model in models:
        # each model_output_dir/model has 5 csv files -- one for each language that contains the model outputs
        model_dir = os.path.join(args.model_output_dir, model)

        # if the file already exists, then load the file and start from the last checkpoint
        output_dir = os.path.join(args.output_dir, model)
        os.makedirs(output_dir, exist_ok=True)

        # iterate over languages
        for lang in languages:
            # output file path
            output_fp = os.path.join(model_dir, f"{lang}.csv")
            if not os.path.exists(output_fp):
                print(f"File {output_fp} does not exist. Skipping...")
                continue
            # load the model outputs as datasets
            dataset = datasets.load_dataset("csv", data_files=output_fp, split="train")

            # get the rubrics for the language
            dataset = dataset.map(
                lambda x, lang=lang: {"rubric": rubrics[lang][x[args.id_column]]["rubric"]}
            )

            filename = f"{output_dir}/{lang}.json"

            if os.path.exists(filename):
                with open(filename) as f:
                    outputs = json.load(f)
            else:
                outputs = {}

            tasks = []
            id_to_example = {}
            for _idx, example in tqdm(enumerate(dataset), total=len(dataset)):
                example_id = example[args.id_column]

                if example_id in outputs:
                    continue

                prompt = prompt_template.format(
                    code_prompt=example[args.code_prompt_column],
                    rubric=example["rubric"],
                    model_response=example[args.model_response_column],
                )

                task = asyncio.create_task(generate_response(prompt))
                tasks.append(task)
                id_to_example[example_id] = example

                if len(tasks) == CUNCURRENT_REQUESTS:
                    results = await asyncio.gather(*tasks)
                    for i, (example_id, ex) in enumerate(id_to_example.items()):
                        score_reasoning = extract_reasoning(results[i])
                        # import pdb; pdb.set_trace()

                        # import pdb; pdb.set_trace()
                        outputs[example_id] = {
                            "prompt": ex[args.code_prompt_column],
                            "model_response": ex[args.model_response_column],
                            "score": score_reasoning["score"],
                            "reasoning": score_reasoning["reasoning"],
                            "raw_output": results[i],
                        }
                    with open(filename, "w") as f:
                        json.dump(outputs, f, indent=4)
                    tasks = []
                    id_to_example = {}

            # Final batch
            if tasks:
                results = await asyncio.gather(*tasks)
                for i, (example_id, ex) in enumerate(id_to_example.items()):
                    score_reasoning = extract_reasoning(results[i])
                    outputs[example_id] = {
                        "prompt": ex[args.code_prompt_column],
                        "model_response": ex[args.model_response_column],
                        "score": score_reasoning["score"],
                        "reasoning": score_reasoning["reasoning"],
                        "raw_output": results[i],
                    }
                with open(filename, "w") as f:
                    json.dump(outputs, f, indent=4)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Rate the problems based on difficulty level.")

    # dataset
    parser.add_argument(
        "--model_output_dir",
        type=str,
        default="llm_as_judge_generation",
        help="Directory where the model outputs are stored. Default is llm_as_judge_generation.",
    )
    parser.add_argument(
        "--rubrics_dir",
        type=str,
        default="rubrics",
        help="Directory where the rubrics are stored. Default is rubrics.",
    )

    parser.add_argument(
        "--prompt_fp",
        type=str,
        default="prompts/llm_as_judge_evaluation.txt",
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
        "--model_response_column",
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
        default="llm_as_judge_results",
        help="Output directory to save the results. Default is current directory.",
    )

    args = parser.parse_args()

    # Run the main function
    asyncio.run(main(args))

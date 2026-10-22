import os
from functools import partial


def load_model(model_name_or_path):
    from vllm import LLM

    llm = LLM(model=model_name_or_path, generation_config="auto", gpu_memory_utilization=0.5)

    sampling_params = llm.get_default_sampling_params()

    return {"llm": llm, "sampling_params": sampling_params}


def generate_respnose(llm, prompts, sampling_params=None):

    outputs = llm.generate(prompts, sampling_params)

    responses = []
    for output in outputs:
        response = output.outputs[0].text
        responses.append(response)

    return responses


def batch_generation(batch, vllm, sampling_params):

    prompts = batch["prompt"]

    responses = generate_respnose(vllm, prompts, sampling_params)

    batch["response"] = responses

    return batch


def main(args):

    # 1. load datasets
    from datasets import load_dataset

    model_dict = load_model(args.model_name_or_path)

    llm = model_dict["llm"]
    sampling_params = model_dict["sampling_params"]

    # update sampling params
    sampling_params.temperature = args.temperature
    sampling_params.max_tokens = args.max_new_tokens

    # ensure that output directory exists
    output_dir = f"{args.output_dir}/{args.model_name_or_path.split('/')[-1]}"
    os.makedirs(output_dir, exist_ok=True)

    # programming languages
    programming_languages = ["cpp", "go", "java", "js", "python"]

    for lang in programming_languages:
        # load the datasets
        # import pdb; pdb.set_trace()
        # save the generated dataset as csv file
        output_file = f"{output_dir}/{lang}.csv"

        # if file exists skip
        if os.path.exists(output_file):
            print(f"File {output_file} already exists, skipping...")
            continue

        dataset = load_dataset(args.dataset_name_or_path, lang, trust_remote_code=True)

        print(f"Loaded {len(dataset['test'])} examples for language: {lang}")

        # iterate through the dataset and generate responses in batches
        # there is clever way of batching basically we need to ensure that the prompts
        # in a batch are of similar length to minimize the padding

        generation_fn = partial(batch_generation, vllm=llm, sampling_params=sampling_params)

        generated_dataset = dataset["test"].map(
            generation_fn, batched=True, batch_size=args.batch_size, num_proc=args.num_proc
        )

        generated_dataset.to_csv(output_file, index=False)

        # exit()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="vLLM Generation Script")
    parser.add_argument(
        "--dataset_name_or_path",
        type=str,
        default="zai-org/humaneval-x",
        help="Input prompt for generation",
    )
    parser.add_argument(
        "--model_name_or_path", type=str, default="Qwen/Qwen3-0.6B", help="Model name or path"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="Qwen3-1.7B_humanevalx.json",
        help="Output file to save generations",
    )
    parser.add_argument("--temperature", type=float, default=0.5, help="Temperature for generation")
    parser.add_argument(
        "--max_new_tokens", type=int, default=2048, help="Maximum new tokens to generate"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="llm_as_judge_generation",
        help="Directory to save outputs",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for generation")
    parser.add_argument(
        "--num_proc", type=int, default=1, help="Number of processes for dataset mapping"
    )

    args = parser.parse_args()
    main(args)

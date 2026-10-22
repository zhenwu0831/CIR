import argparse
import json
import random
from pathlib import Path

import datasets
import pandas as pd
from suffixes import suffixes

random.seed(42)

root_dir = "code-modification"


parser = argparse.ArgumentParser()
parser.add_argument("--input_files", nargs="+", required=True)
parser.add_argument("--raw_instructions_fp", required=True)
parser.add_argument("--output_dir", required=True)
args = parser.parse_args()
files = args.input_files
Path(args.output_dir).mkdir(parents=True, exist_ok=True)


def load_json(file_path):
    with open(file_path) as file:
        return json.load(file)


raw_instructions = load_json(args.raw_instructions_fp)

for file in files:
    data = load_json(file)

    # suffix code
    suffix_code = file.split("/")[-1].split(".")[0].split("_gen_")[-1]
    if suffix_code == "pseducode":
        suffix_code = "pseudocode"

    updated_data = {}

    # iterate over all keys in the data and replace the instruction with raw_instruction + random suffix
    for id, values in data.items():
        raw_instruction = raw_instructions[id]["raw_instruction"]

        language = id.split("_")[0]
        current_instruction = values["instruction"]

        # randoomly select a suffix from list of suffixes for the given suffix_code -- with seed 42
        random_suffix = random.choice(suffixes[suffix_code])

        # sanity check -- raw_instruction must be a substring of current_instruction
        if raw_instruction in current_instruction:
            # replace the instruction with raw_instruction + random suffix
            values["instruction"] = raw_instruction + " " + random_suffix.format(language=language)

            # update the id to be the raw_instruction + random suffix
            updated_id = f"{suffix_code}_{id}"
            updated_data[updated_id] = values

            # import pdb;pdb.set_trace()
        else:
            print(f"Warning: Raw instruction not found in current instruction for id '{id}'")
            # import pdb;pdb.set_trace()
            exit()

    # convert the updated data to a pandas dataframe
    df = pd.DataFrame.from_dict(updated_data, orient="index")
    df.reset_index(inplace=True)
    df.rename(columns={"index": "id", "new_response": "response"}, inplace=True)

    # convert the dataframe to a dataset
    ds = datasets.Dataset.from_pandas(df)

    dataset = datasets.DatasetDict({"train": ds.shuffle(seed=42)})

    dataset.save_to_disk(str(Path(args.output_dir) / suffix_code))

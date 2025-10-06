import argparse
from datasets import load_dataset, DatasetDict
import json
import os
import pickle

REMOVE_PREFIX = "# YOUR CODE HERE\n"


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--evaluate_dir", type=str, required=True)
    parser.add_argument("--output_dataset", type=str, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    config = get_config()
    dataset = load_dataset(config.dataset, split="train")

    with open(os.path.join(config.evaluate_dir, "state_history.pkl"), "rb") as f:
        state_history = pickle.load(f)

    code = [state.code for state in state_history]
    code = list(
        map(
            lambda x: x[len(REMOVE_PREFIX) :] if x.startswith(REMOVE_PREFIX) else x,
            code,
        )
    )
    if "code" in dataset.column_names:
        dataset = dataset.remove_columns("code")
    dataset = dataset.add_column("code", code)
    dataset_dict = DatasetDict({"train": dataset})
    dataset_dict.push_to_hub(config.output_dataset)

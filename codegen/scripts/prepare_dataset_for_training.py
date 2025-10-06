import argparse

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
)

from codegen.training.utils import (
    choose_best_completions,
    format_dataset,
    tokenize_dataset,
)

"""
Requires a dataset that has the empowerment values precomputed.
Will select the completions to train on, tokenize the dataset, and then upload to huggingface
"""


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
    )
    parser.add_argument(
        "--completion_chooser",
        type=str,
        default="random",
        help="Options are 'random', 'empowerment', 'longest', 'longest_random', or 'shortest'",
    )
    parser.add_argument(
        "--model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct"
    )
    parser.add_argument(
        "--empowerment_model", type=str, default="meta-llama/Llama-3.1-8B"
    )
    parser.add_argument(
        "--k_r_high", type=int, default=100, help="Maximum length of robot completion"
    )
    parser.add_argument(
        "--k_r_low", type=int, default=1, help="Minimum length of robot completion"
    )
    parser.add_argument(
        "--accept_prob",
        type=float,
        default=0.5,
        help="Accept probability to weight empowerment by",
    )
    parser.add_argument(
        "--train_size",
        type=int,
        default=0,
        help="Number of problems to start from when creating the dataset",
    )
    parser.add_argument(
        "--test_frac", type=float, default=0.2, help="Fractional size of the test set"
    )
    args = parser.parse_args()

    return vars(args)


if __name__ == "__main__":
    config = get_config()
    dataset = load_dataset(config["dataset"])
    dataset["train"] = (
        dataset["train"].select(range(config["train_size"]))
        if config["train_size"] > 0
        else dataset["train"]
    )
    dataset["test"] = (
        dataset["test"].select(range(int(config["train_size"] * config["test_frac"])))
        if config["train_size"] > 0
        else dataset["test"]
    )
    tokenizer = AutoTokenizer.from_pretrained(config["model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = 128001

    emp_tokenizer = AutoTokenizer.from_pretrained(config["empowerment_model"])
    if emp_tokenizer.pad_token is None:
        emp_tokenizer.pad_token = emp_tokenizer.eos_token

    ds_suffix = f"f_{config['completion_chooser']}"
    if not config["dataset"].endswith(ds_suffix):
        print("Choosing best completions for dataset")
        dataset = choose_best_completions(
            dataset, config, emp_tokenizer, config["completion_chooser"]
        )

        print("Formatting dataset")
        formatted_dataset = format_dataset(dataset, tokenizer)

        print("Tokenizing dataset")
        tokenized_dataset = tokenize_dataset(formatted_dataset, tokenizer)

        tokenized_dataset.push_to_hub(f"{config['dataset']}_{ds_suffix}")
    else:
        print(
            f"Dataset {config['dataset']} already exists with suffix {ds_suffix}. Skipping."
        )

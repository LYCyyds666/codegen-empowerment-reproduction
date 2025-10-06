import argparse
from datasets import load_dataset
from transformers import AutoTokenizer
from codegen.training.utils import tokenize_dataset


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    config = get_config()
    dataset = load_dataset(config.dataset)
    if config.debug:
        dataset["train"] = dataset["train"].select(range(100))
        dataset["test"] = dataset["test"].select(range(30))

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)

    tokenizer_suffix = config.tokenizer.split("/")[-1]
    tokenized_dataset.push_to_hub(config.dataset + "_" + tokenizer_suffix)

from datasets import DatasetDict, load_dataset
import argparse
import torch
from transformers import AutoTokenizer
from codegen.training.utils import (
    format_and_tokenize_dataset,
)
from codegen.utils.utils import (
    setup_tokenizer_special_tokens,
)
import random

"""
Construct a baseline dataset for finetune_agent.py.
For each code, we take 20 tokens after the state for training.
That is used for training.
"""


def tokenize(example, tokenizer, max_length):
    def tokenize_list(text: list[str]):
        iids = []
        attn_mask = []
        for string in text:
            enc = tokenizer(
                string, max_length=max_length, truncation=True, padding="max_length"
            )
            iids.append(enc["input_ids"])
            attn_mask.append(enc["attention_mask"])
        return iids, attn_mask

    code_text = example["code"]
    state_text = example["state"]

    # Tokenize each
    code_enc = tokenizer(
        code_text, max_length=max_length, truncation=True, padding="max_length"
    )

    state_enc = tokenizer(
        state_text, max_length=max_length, truncation=True, padding="max_length"
    )
    return {
        "input_ids_code": code_enc["input_ids"],
        "attention_mask_code": code_enc["attention_mask"],
        "input_ids_state": state_enc["input_ids"],
        "attention_mask_state": state_enc["attention_mask"],
    }


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
    )
    parser.add_argument("--assistant_model", type=str, required=True)
    parser.add_argument(
        "-T",
        "--tokens_after_state",
        type=int,
        default=20,
        help="The number of tokens after the state to take for the completion",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Whether to choose the completion length randomly between 1 and T",
    )
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def choose_agreement_completion(
    iids_state: list[int],
    iids_code: list[int],
    tokens_after_state: int,
    sample: bool,
) -> dict[str, list[int]]:
    """
    Choose the completion to be T tokens after the state.
    :param iids_state: The state
    :param iids_code: The code
    :param tokens_after_state: The number of tokens after the state to take for the completion
    :param sample: Whether to choose the completion length randomly between 1 and T
    :return: The correct completion
    """
    if sample:
        tokens_after_state = random.randint(1, tokens_after_state + 1)
    correct_completion = iids_code[: len(iids_state) + tokens_after_state]

    return {"iids_correct_completion": correct_completion}


if __name__ == "__main__":
    config = get_config()
    dataset = load_dataset(config.dataset, split="train")
    if config.debug:
        dataset = dataset.select(range(100))
        # dataset["test"] = dataset["test"].select(range(40))

    assistant_tokenizer = AutoTokenizer.from_pretrained(config.assistant_model)
    setup_tokenizer_special_tokens(assistant_tokenizer)

    dataset = dataset.map(
        lambda code: {"iids_code": assistant_tokenizer(code)["input_ids"]},
        batched=True,
        input_columns=["code"],
    )

    print("Choosing fixed length completion")
    dataset = dataset.map(
        choose_agreement_completion,
        batched=False,
        input_columns=[
            "iids_state",
            "iids_code",
        ],
        fn_kwargs={
            "tokens_after_state": config.tokens_after_state,
            "sample": config.sample,
        },
    )

    # Delete the outdated columns
    for col in [
        "input_ids_state",
        "attention_mask_state",
        "input_ids_state_ar",
        "attention_mask_state_ar",
        "input_ids_next_state",
        "attention_mask_next_state",
        "completion",
    ]:
        if col in dataset.column_names:
            dataset = dataset.remove_columns([col])

    # Format the dataset
    print("Formatting dataset")
    tokenized_ds = format_and_tokenize_dataset(dataset, assistant_tokenizer)
    tokenized_ds = DatasetDict({"train": tokenized_ds})

    print("Pushing to hub")
    tokenized_ds.push_to_hub(
        f"{config.dataset}_{config.tokens_after_state}_tok_after_state{'_sample' if config.sample else ''}"
    )

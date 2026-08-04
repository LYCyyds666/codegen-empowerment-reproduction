import pickle
from pathlib import Path

from datasets import Dataset
from transformers import AutoTokenizer


MODEL_NAME = "/scratch/hhu49/97e911/models/Qwen3-0.6B"
OUTPUT_PATH = Path("/scratch/hhu49/97e911/data/codegen_smoke.pkl")
MAX_LENGTH = 4096


def make_example(tokenizer, index: int) -> dict:
    prompt = (
        "You are a coding assistant. Return only Python code.\n"
        f"Problem {index}: Write a Python function add(a, b) "
        "that returns the sum of a and b.\n"
        "Answer:\n"
    )
    answer = "def add(a, b):\n    return a + b\n"

    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=True,
    )["input_ids"]

    answer_ids = tokenizer(
        answer,
        add_special_tokens=False,
    )["input_ids"]

    if tokenizer.eos_token_id is not None:
        answer_ids.append(tokenizer.eos_token_id)

    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids
    attention_mask = [1] * len(input_ids)

    if len(input_ids) > MAX_LENGTH:
        raise ValueError(f"Example too long: {len(input_ids)}")

    pad_length = MAX_LENGTH - len(input_ids)

    input_ids += [tokenizer.pad_token_id] * pad_length
    labels += [-100] * pad_length
    attention_mask += [0] * pad_length

    assert len(input_ids) == MAX_LENGTH
    assert len(labels) == MAX_LENGTH
    assert len(attention_mask) == MAX_LENGTH
    assert any(label != -100 for label in labels)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("Tokenizer has no pad or EOS token.")
        tokenizer.pad_token = tokenizer.eos_token

    examples = [make_example(tokenizer, i) for i in range(20)]
    dataset = Dataset.from_list(examples)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("wb") as file:
        pickle.dump(dataset, file)

    print(dataset)
    print(f"Saved smoke dataset: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

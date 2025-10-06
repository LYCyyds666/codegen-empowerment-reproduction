from datasets import DatasetDict, load_dataset
import argparse
from transformers import AutoTokenizer
from codegen.training.utils import (
    format_and_tokenize_dataset,
)
from codegen.utils.utils import (
    setup_tokenizer_special_tokens,
)
from transformers import AutoModelForCausalLM
from accelerate import Accelerator
from torch.utils.data import DataLoader
from codegen.policies.human import format_prompt_blind_null_from_code
import torch
from tqdm import tqdm
from codegen.utils.utils import accelerator_breakpoint

"""
Construct a dataset for finetune_agent.py.
For each state, we take the longest completion which is agreed upon by both the true completion and the nulls. 
That is used for training.
"""

PAD_ID = -100
PAD_LENGTH = 5000


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
    )
    parser.add_argument(
        "--save_name",
        type=str,
    )
    parser.add_argument("--assistant_model", type=str, required=True)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Cumulative probability lower bound for the completion.",  # "Bound on the one-sample cumulative entropy of the completion.",
    )
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def choose_agreement_completion(
    iids_state: list[int],
    iids_code: list[int],
    iids_null_code: list[list[int]],
) -> dict[str, list[int]]:
    """
    Choose the longest completion which is agreed upon by both the true completion and the nulls.
    :param iids_state: The state
    :param iids_code: The code
    :param null_code: The null code
    :return: The correct completion
    """
    correct_completion = iids_state
    for i in range(len(iids_state), len(iids_code)):
        must_match = iids_code[i]

        for nc in iids_null_code:
            if i >= len(nc) or nc[i] != must_match:
                correct_completion = iids_code[:i]

                null_says = nc[max(0, i - 4) : i + 1]
                code_says = iids_code[max(0, i - 4) : i + 1]
                print(
                    f"Moment of disagreement {i - len(iids_state)} tokens after state: null says {null_says}, code says {code_says}"
                )

                return {"iids_correct_completion": correct_completion}

    return {"iids_correct_completion": correct_completion}


def get_prompt(
    iids_state: list[int], iids_code: list[int], tokenizer: AutoTokenizer
) -> dict[str, torch.Tensor]:
    state = tokenizer.decode(iids_state, skip_special_tokens=True)
    code = tokenizer.decode(iids_code, skip_special_tokens=True)
    prompt = format_prompt_blind_null_from_code(state)
    prompt.append(
        {
            "role": "assistant",
            "content": f"""```python
{code}""",
        }
    )
    tokenized = tokenizer.apply_chat_template(
        prompt,
        add_generation_prompt=False,
        truncation=False,
        continue_final_message=True,
        return_tensors="pt",
        return_dict=True,
    )

    return {
        "input_ids": tokenized["input_ids"][0],
        "attention_mask": tokenized["attention_mask"][0],
    }


def collate(batch):
    input_ids = torch.tensor([x["input_ids"] for x in batch])
    attention_mask = torch.tensor([x["attention_mask"] for x in batch])
    iids_state = torch.tensor([x["iids_state"] for x in batch])
    iids_code = torch.tensor([x["iids_code"] for x in batch])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "iids_state": iids_state,
        "iids_code": iids_code,
    }


if __name__ == "__main__":
    config = get_config()
    dataset = load_dataset(config.dataset, split="train")
    if config.debug:
        dataset = dataset.select(range(100))

    assistant_tokenizer = AutoTokenizer.from_pretrained(config.assistant_model)
    setup_tokenizer_special_tokens(assistant_tokenizer)

    print("Choosing agreement completion")

    dataset = dataset.map(
        lambda code: {"iids_code": assistant_tokenizer(code)["input_ids"]},
        batched=True,
        input_columns=["code"],
    )
    dataset = dataset.map(
        get_prompt,
        batched=False,
        fn_kwargs={"tokenizer": assistant_tokenizer},
        input_columns=["iids_state", "iids_code"],
    )
    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "iids_state", "iids_code"],
    )

    assistant_model = AutoModelForCausalLM.from_pretrained(config.assistant_model)

    accelerator = Accelerator()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    assistant_model, dataloader = accelerator.prepare(assistant_model, dataloader)

    all_iids_correct_completions_padded = []
    for batch in tqdm(dataloader):
        iids_state = batch["iids_state"]
        iids_code = batch["iids_code"]
        iids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        B, C = iids.shape
        with torch.no_grad():
            output = assistant_model(
                input_ids=iids,
                attention_mask=attention_mask,
            )
        shifted_logits = output.logits[:, :-1, :]
        shifted_iids = iids[:, 1:]
        probs = torch.softmax(shifted_logits, dim=-1)

        state_len = len(batch["iids_state"][0])
        tokens_remaining = len(batch["iids_code"][0]) - state_len

        compl_probs = probs.take_along_dim(shifted_iids[..., None], dim=-1)[..., 0]
        cum_probs = torch.cumprod(compl_probs[:, -tokens_remaining:], dim=-1)

        completion_mask = cum_probs >= config.threshold
        iids_correct_completion = shifted_iids[:, -tokens_remaining:][completion_mask]
        # Prepend the state
        iids_correct_completion = torch.cat([iids_state[0], iids_correct_completion])

        iids_correct_completions_padded = torch.cat(
            [
                iids_correct_completion,
                torch.full((PAD_LENGTH - len(iids_correct_completion),), PAD_ID).to(
                    iids_correct_completion.device
                ),
            ],
        )
        iids_correct_completions_padded = accelerator.gather(
            iids_correct_completions_padded[None]
        )
        all_iids_correct_completions_padded.append(
            iids_correct_completions_padded.to("cpu")
        )
        del (
            iids,
            attention_mask,
            output,
            shifted_logits,
            shifted_iids,
            probs,
            compl_probs,
            cum_probs,
            completion_mask,
            iids_correct_completion,
            iids_correct_completions_padded,
        )

    iids_correct_completions_padded = torch.cat(
        all_iids_correct_completions_padded, dim=0
    )

    if accelerator.is_main_process:
        if "iids_correct_completion" in dataset.column_names:
            dataset = dataset.remove_columns(["iids_correct_completion"])
        iids_correct_completions_padded = iids_correct_completions_padded[
            : len(dataset)
        ]

        # Remove the PAD tokens
        iids_correct_completions = []
        for iid in iids_correct_completions_padded:
            iids_correct_completions.append(iid[iid != PAD_ID].tolist())
        dataset = dataset.add_column(
            "iids_correct_completion", iids_correct_completions
        )

        mean_completion_length = sum(
            [
                len(iid) - len(state)
                for iid, state in zip(iids_correct_completions, dataset["iids_state"])
            ]
        ) / len(iids_correct_completions)
        print(f"Mean completion length: {mean_completion_length}")

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
        print("Formatting and tokenizing dataset")
        dataset = format_and_tokenize_dataset(dataset, assistant_tokenizer)

        tokenized_ds = DatasetDict({"train": dataset})
        print("Pushing to hub")
        tokenized_ds.push_to_hub(config.save_name + str(config.threshold))

    accelerator.wait_for_everyone()

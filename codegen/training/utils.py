import random

import numpy as np
import torch
from datasets import Dataset, DatasetDict, concatenate_datasets
from transformers import AutoTokenizer

from codegen.empowerment import get_last_non_eos_index
from codegen.policies.assistant import format_assistant_prompts
import re
import os

PAD_LENGTH = 4096


def choose_best_completions(
    dataset: DatasetDict,
    config: dict[str, any],
    tokenizer: AutoTokenizer,
    completion_chooser: str,
) -> DatasetDict:
    def choose_best_helper(empowerment_q, iids_state, iids_code, iids_null_code, idx=0):
        t = get_last_non_eos_index(iids_state, tokenizer)
        s_eos_idx = t + 1
        assert (t >= 0).all().item(), "Could not find state EOS token"

        if idx == 0:
            iids_completion = iids_code
        else:
            iids_completion = iids_null_code[idx - 1]

        empowerment_v = empowerment_q.mean()
        if completion_chooser == "empowerment":
            max_emp_idx = empowerment_q[idx].argmax()
            iids_completion = iids_completion[: s_eos_idx + max_emp_idx + 1]
            empowerment_q = empowerment_q[idx][max_emp_idx]
        elif completion_chooser == "longest":
            iids_completion = iids_completion[: s_eos_idx + config["k_r_high"]]
            empowerment_q = torch.tensor(0, dtype=torch.float32)
        elif completion_chooser == "random":
            rand_idx = random.randint(0, config["k_r_high"] - 1)
            iids_completion = iids_completion[: s_eos_idx + rand_idx + 1]
            empowerment_q = empowerment_q[idx][rand_idx]
        elif completion_chooser == "longest_random":
            rand_idx = random.randint(0, config["k_r_high"] - 1)
            iids_completion = iids_completion[: s_eos_idx + rand_idx + 1]
            empowerment_q = -torch.tensor(rand_idx, dtype=torch.float32)
            empowerment_v = torch.tensor(0, dtype=torch.float32)
        elif completion_chooser == "shortest":
            iids_completion = iids_completion[: s_eos_idx + config["k_r_low"]]
            empowerment_q = torch.tensor(0, dtype=torch.float32)

        completion = tokenizer.decode(iids_completion, skip_special_tokens=True) + "```"

        return {
            "correct_completion": completion,
            "empowerment_q": empowerment_q,
            "empowerment_a": empowerment_q - empowerment_v,
            "empowerment_v": empowerment_v,
        }

    dataset.set_format("torch", columns=["empowerment"])
    ds2 = dataset.map(
        lambda emp, accept_prob: {
            "empowerment_q": compute_empowerment_q(emp, accept_prob)
        },
        input_columns="empowerment",
        fn_kwargs={"accept_prob": config["accept_prob"]},
        batched=True,
        batch_size=256,
        keep_in_memory=True,
    )

    # Unravel all of the positives into their own entries
    ds_best_completions = []
    for i in range(ds2["train"][0]["empowerment"].shape[0]):
        ith_ds = ds2.map(
            choose_best_helper,
            input_columns=[
                "empowerment_q",
                "input_ids_state",
                "input_ids_code",
                "input_ids_null_code",
            ],
            batched=False,
            fn_kwargs={"idx": i},
            keep_in_memory=True,
        )
        ds_best_completions.append(ith_ds)

    ds = {}
    for split in dataset.keys():
        ds[split] = concatenate_datasets(
            [d[split] for d in ds_best_completions], split=split
        )
    ds = DatasetDict(ds)
    return ds


def tokenize_dataset(dataset, tokenizer: AutoTokenizer):
    def tokenize_function(examples):
        enc = tokenizer(
            examples["assistant_prompt_response"],
            padding="max_length",
            truncation=True,
            max_length=4096,
            return_offsets_mapping=True,  # <-- the crucial flag
            return_tensors="pt",
        )

        labels = enc["input_ids"].clone()

        # We need to throw out the examples where the tokenized state is the same length as the tokenized response
        throwout = []
        correct_states = []
        for i, offsets in enumerate(enc["offset_mapping"]):
            cutoff = (
                len(examples["chat_template_state"][i]) - 10
            )  # We need to subtract 10 because that is the length of <|eot_id|>
            keep_mask = torch.tensor(
                [cutoff < end - 1 for start, end in offsets],
                dtype=torch.bool,
            )
            labels[i][~keep_mask] = -100  # everything before `cutoff` is prompt

            # Remove the part of the state that is in the labels so that the empowerment calculation is correct
            if keep_mask.nonzero().numel() == 0:
                throwout.append(True)
                correct_states.append(examples["chat_template_state"][i])
                continue

            correct_chat_template_state_iids = enc["input_ids"][i][
                : keep_mask.nonzero()[0, 0]
            ]

            correct_chat_template_state = tokenizer.decode(
                correct_chat_template_state_iids
            )
            correct_state_str = re.match(
                r"(.|\n)*```python\n((.|\n)*)", correct_chat_template_state
            ).group(2)
            correct_states.append(correct_state_str)
            throwout.append(False)

        enc["labels"] = labels
        enc["state"] = correct_states
        enc["throwout"] = throwout

        # parse = lambda x: tokenizer.decode(x[x != -100], skip_special_tokens=True)
        # breakpoint()

        del enc["offset_mapping"]  # save RAM / not needed downstream
        return enc

    tokenized_datasets = dataset.map(
        tokenize_function, batched=True, keep_in_memory=True
    )
    # tokenized_datasets = tokenized_datasets.remove_columns(["code"])
    tokenized_datasets = tokenized_datasets.filter(
        lambda throwout: not throwout, input_columns="throwout"
    )
    tokenized_datasets = tokenized_datasets.filter(
        lambda labels: (torch.tensor(labels) != -100).sum() > 0, input_columns="labels"
    )

    return tokenized_datasets


def format_and_tokenize_dataset(dataset, tokenizer: AutoTokenizer) -> Dataset:
    def format_for_assistant_sft(entry):
        state_text = tokenizer.decode(entry["iids_state"], skip_special_tokens=True)
        completion_message = format_assistant_prompts(
            state_text, assistant_message=""
        )  # We'll add in the assistant message later
        input_ids = tokenizer.apply_chat_template(
            completion_message, tokenize=True, continue_final_message=True
        )

        # Add in the correct completion tokens
        prompt_length = len(input_ids) + len(entry["iids_state"])
        input_ids += entry["iids_correct_completion"]

        # Add in a ``` and a eos_token
        input_ids += tokenizer("```", add_special_tokens=False)["input_ids"] + [
            tokenizer.eos_token_id
        ]
        labels = input_ids[prompt_length:]
        labels = [-100] * prompt_length + labels
        attention_mask = [1] * len(input_ids)

        # Pad out to 4096
        pad_length = PAD_LENGTH - len(input_ids)
        input_ids += [tokenizer.pad_token_id] * pad_length
        labels += [-100] * pad_length
        attention_mask += [0] * pad_length

        # Triple check that the labels are correct
        label_tensor = torch.tensor(labels)
        label_mask = label_tensor != -100

        assert label_mask.any()
        assert (label_tensor[label_mask] == torch.tensor(input_ids)[label_mask]).all()
        assert (label_tensor[:prompt_length] == -100).all()

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    return dataset.map(format_for_assistant_sft, batched=False)


def compute_weights(dataset, strategy: str, temperature: float) -> Dataset:
    def compute_weight(adv):
        adv = torch.tensor(adv)
        if strategy == "awr":
            weight = np.exp((adv - adv_mean) / (adv_std * temperature))

            return {"weight": torch.tensor(np.clip(weight, 0.01, 100))}
        elif strategy == "wbc":
            weight = (adv - adv_mean) / (adv_std * temperature)

            return {
                "weight": weight,
            }
        else:
            return {"weight": torch.ones_like(adv)}

    if strategy in ["awr", "wbc"]:
        train_adv = torch.tensor(dataset["train"]["empowerment_a"])
        adv_mean = train_adv.mean()
        adv_std = train_adv.std()

        return dataset.map(
            compute_weight,
            input_columns="empowerment_a",
            keep_in_memory=True,
            batched=True,
            batch_size=2048,
        )
    else:
        dataset = dataset.map(
            lambda iids_code: {"weight": torch.ones((len(iids_code),))},
            input_columns=["input_ids"],
            batched=True,
            batch_size=2048,
        )
        return dataset


def format_dataset(dataset, tokenizer: AutoTokenizer) -> Dataset:
    def format_for_assistant_sft(entry):
        completion_message = format_assistant_prompts(
            entry["state"], assistant_message=entry["correct_completion"]
        )
        entry["assistant_prompt_response"] = tokenizer.apply_chat_template(
            completion_message, tokenize=False
        )
        state_message = format_assistant_prompts(entry["state"])
        entry["chat_template_state"] = tokenizer.apply_chat_template(
            state_message, tokenize=False
        )

        return entry

    return dataset.map(format_for_assistant_sft, batched=False, num_proc=20)

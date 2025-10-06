import argparse
import gc
import os
import pickle

import numpy as np
import pandas as pd
import torch
import yaml
from accelerate import Accelerator, DistributedDataParallelKwargs
from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import set_seed

from codegen.empowerment import (
    empowerment_from_state_and_action,
    initialize_empowerment_from_config,
    tokenize,
)
from codegen.empowerment.base import EmpowermentConfig
from codegen.environment import LCBEnvironment, State
from codegen.evaluate.evaluate_with_assistant import run_phase
from codegen.evaluate.base import (
    RolloutWithAssistantConfig,
    EvaluateWithAssistantConfig,
    EvaluateConfig,
)
from codegen.generators import initialize_generator
from codegen.policies import Assistant, HumanAcceptor, HumanAppender
from codegen.utils import download_model_if_needed
import hydra
from transformers import AutoTokenizer
from codegen.training.utils import format_dataset, tokenize_dataset
from lcb_runner.benchmarks import CodeGenerationProblem
from codegen.environment.benchmark import ds_row_to_code_generation_problem
from datetime import datetime
from dataclasses import dataclass
from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics
from vllm.distributed.parallel_state import destroy_model_parallel

SAVE_ROOT = os.environ["CODEGEN_ROOT"]


def run_test_cases(
    state_histories: list[list[list[State]]],
    num_process_evaluate: int = 12,
    timeout: int = 6,
) -> list[dict[int, float]]:
    test_case_results = []
    for i in range(len(state_histories)):
        eval_samples = [
            sh[0].problem.get_evaluation_sample() for sh in state_histories[i]
        ]

        eval_results = codegen_metrics(
            eval_samples,
            [[sh[-1].code] for sh in state_histories[i]],
            num_process_evaluate=num_process_evaluate,
            timeout=timeout,
        )
        test_case_results.append(eval_results[0]["detail"]["pass@1"])

    return test_case_results


@hydra.main(config_path="../configs", config_name="rollout_on_train", version_base=None)
def roll_out_on_dataset(
    config: RolloutWithAssistantConfig,
) -> tuple[str, dict]:
    set_seed(config.seed)

    if config.dataset.endswith("pkl"):
        dataset = pickle.load(open(config.dataset, "rb"))
    else:
        dataset_dict = load_dataset(config.dataset)
        assert isinstance(dataset_dict, DatasetDict)

        if config.num_problems > 0:
            dataset = dataset_dict["train"].shuffle().select(range(config.num_problems))
        else:
            dataset = dataset_dict["train"]

    assistant_generator = initialize_generator(config.assistant.generator)
    tokenizer = AutoTokenizer.from_pretrained(config.assistant.generator.tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token

    assistant = Assistant(assistant_generator, config.assistant)
    env = LCBEnvironment([])

    if config.resample_state:
        state_text = []
        for d in dataset:
            iids = tokenizer(d["code"])["input_ids"]
            length = np.random.randint(0, len(iids))

            iids = iids[:length]
            state_text.append(tokenizer.decode(iids, skip_special_tokens=True))
        dataset = (
            dataset.remove_columns("state")
            if "state" in dataset.column_names
            else dataset
        )
        dataset = dataset.add_column("state", state_text)

    states = []
    for d in dataset:
        states.append(
            State(
                code=d["state"],
                suggested_completion="",
                problem=ds_row_to_code_generation_problem(d),
            )
        )

    N = len(states)
    K = config.num_futures  # Number of possible futures to generate per state

    # Initialize history lists for the first future
    state_histories = [[[state] for state in states]]
    info_histories: list[list[list[dict]]] = [[[] for _ in range(N)]]
    active = [[True] * N]

    # Run assistant phase to get suggestions
    run_phase(
        active[0],
        state_histories[0],
        info_histories[0],
        assistant,
        env.step_assistant,
        check_done=False,
    )

    # Now initialize the remaining futures with the assistant's suggestions
    for _ in range(K - 1):
        state_histories.append([sh.copy() for sh in state_histories[0]])
        info_histories.append([[[{}]] for ih in info_histories[0]])
        active.append([True] * N)

    del assistant_generator.llm
    gc.collect()
    torch.cuda.empty_cache()

    human_generator = initialize_generator(config.human.generator)
    human_acceptor = HumanAcceptor(human_generator, config.human)
    human_appender = HumanAppender(human_generator, config.human)

    # For each future path
    for k in range(K):
        # Run acceptor and appender for this future
        run_phase(
            active[k],
            state_histories[k],
            info_histories[k],
            human_acceptor,
            env.step_human,
            check_done=False,
        )
        run_phase(
            active[k],
            state_histories[k],
            info_histories[k],
            human_appender,
            env.step_human,
            check_done=True,
        )

    pass_at_1 = 0
    if config.run_test_cases:
        assert isinstance(config, EvaluateWithAssistantConfig) or isinstance(
            config, EvaluateConfig
        )
        test_case_results = run_test_cases(
            state_histories,
            num_process_evaluate=config.num_process_evaluate,
            timeout=config.timeout,
        )

        pass_at_1 = np.mean(list(test_case_results[0].values()))
        if config.filter_correct:
            filtered_state_histories: list[list[list[State]]] = []
            filtered_info_histories: list[list[list[dict]]] = []
            for i in range(len(state_histories)):
                filtered_state_histories.append([])
                filtered_info_histories.append([])
                for j in range(len(state_histories[i])):
                    if test_case_results[i][j] == 1:
                        filtered_state_histories[i].append(state_histories[i][j])
                        filtered_info_histories[i].append(info_histories[i][j])
            state_histories = filtered_state_histories
            info_histories = filtered_info_histories

            def filter_dataset_correct(example, idx, test_case_results):
                return test_case_results[idx] == 1

            dataset = dataset.filter(
                filter_dataset_correct,
                with_indices=True,
                fn_kwargs={"test_case_results": test_case_results[0]},
            )

    save_dir = os.path.join(
        SAVE_ROOT, config.assistant.generator.model_name.split("/")[-1]
    )
    os.makedirs(save_dir, exist_ok=True)

    # Save the original rollout dataset
    with open(os.path.join(save_dir, "training_results.pkl"), "wb") as fb:
        pickle.dump(state_histories[0], fb)
    with open(os.path.join(save_dir, "training_results_config.pkl"), "wb") as fb:
        pickle.dump(config, fb)

    if config.save_as_dataset:
        # Create the original rollout dataset
        def modify_example(example, idx, state_histories, info_histories):
            # The correct completion is always the assistant's suggestion
            example["correct_completion"] = (
                state_histories[0][idx][1].suggested_completion + "```"
            )

            # The final code is from the first future
            example["code"] = state_histories[0][idx][-1].code
            example["accepted"] = info_histories[0][idx][1]["action"] == "accept"
            return example

        modified_dataset = dataset.map(
            modify_example,
            with_indices=True,
            fn_kwargs={
                "state_histories": state_histories,
                "info_histories": info_histories,
            },
        )
        formatted_rollout_dataset = format_dataset(modified_dataset, tokenizer)
        formatted_rollout_dataset = tokenize_dataset(
            formatted_rollout_dataset, tokenizer
        )

        with open(os.path.join(save_dir, "rollout_dataset.pkl"), "wb") as fb:
            pickle.dump(formatted_rollout_dataset, fb)

        # Create the paired dataset for binary NCE
        paired_dataset = []
        for idx in range(len(dataset)):
            # Create example with next states and final codes for all futures
            example = dataset[idx].copy()

            # Delete unneeded columns
            for col in list(example.keys()):
                if (
                    "input_ids" in col
                    or "attention_mask" in col
                    or col == "labels"
                    or "empowerment" in col
                    or col == "hn_code"
                    or col == "completion"
                    or col == "null_code"
                    or col == "final_codes"
                    or col == "next_states"
                ):
                    del example[col]

            example["correct_completion"] = state_histories[0][idx][
                1
            ].suggested_completion

            # Store next states and final codes for each future
            next_states = []
            final_codes = []
            for k in range(K):
                # The next state is 20 tokens after the code at state_histories[k][idx][-2]
                tokenized_code = tokenizer(state_histories[k][idx][-1].code)[
                    "input_ids"
                ]
                tokenized_after_acceptor = tokenizer(state_histories[k][idx][-2].code)[
                    "input_ids"
                ]
                tokenized_next_state = tokenized_code[
                    : len(tokenized_after_acceptor)
                    + 10  # The human takes an action of length 10
                ]
                next_states.append(
                    tokenizer.decode(tokenized_next_state, skip_special_tokens=True)
                )
                final_codes.append(state_histories[k][idx][-1].code)

            example["correct_completion"] = state_histories[0][idx][
                1
            ].suggested_completion
            example["next_state"] = next_states
            example["code"] = final_codes

            paired_dataset.append(example)

        # Format and save the paired dataset
        formatted_paired_dataset = format_dataset(
            Dataset.from_list(paired_dataset), tokenizer
        )
        formatted_paired_dataset = tokenize_dataset(formatted_paired_dataset, tokenizer)
        with open(os.path.join(save_dir, "paired_dataset.pkl"), "wb") as fb:
            pickle.dump(formatted_paired_dataset, fb)

    # Calculate and print metrics
    action_lengths = [
        len(tokenizer(sh[1].suggested_completion)["input_ids"])
        - len(tokenizer(sh[1].code)["input_ids"])
        for sh in state_histories[0]
    ]
    print(
        f"Mean action length: {np.mean(action_lengths)}, SEM: {np.std(action_lengths) / np.sqrt(len(action_lengths))}, N= {len(action_lengths)}"
    )

    # Compute accept ratio across all futures
    accept_ratios = []
    for k in range(K):
        accept_ratios.extend([ih[1]["action"] == "accept" for ih in info_histories[k]])
    accept_ratio = np.mean(accept_ratios)
    print(
        f"Mean accept ratio: {accept_ratio}, SEM: {np.std(accept_ratios) / np.sqrt(len(accept_ratios))}, N= {len(accept_ratios)}"
    )

    info = {
        "accept_ratio": accept_ratio,
        "mean_action_length": np.mean(action_lengths),
        "sem_action_length": np.std(action_lengths) / np.sqrt(len(action_lengths)),
        "pass@1": pass_at_1,
    }

    destroy_model_parallel()
    del human_generator.llm
    gc.collect()
    torch.cuda.empty_cache()

    return save_dir, info


if __name__ == "__main__":
    roll_out_on_dataset()

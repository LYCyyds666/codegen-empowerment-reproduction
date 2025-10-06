import argparse
import json
import os
import pickle
import random

import numpy as np
import wandb
import yaml
from lcb_runner.evaluation import codegen_metrics

from codegen.environment import LCBEnvironment
from codegen.environment.benchmark import build_prompt_benchmark
from codegen.generators import initialize_generator
from codegen.policies import Assistant, HumanAcceptor, HumanAppender
from codegen.evaluate.base import EvaluateWithAssistantConfig
import hydra
from omegaconf import OmegaConf
from codegen.evaluate.compute_metrics_from_results import assistant_metric
from transformers import AutoTokenizer


SAVE_ROOT = os.environ["CODEGEN_ROOT"]


def run_phase(
    active, state_history, info_history, compute_fn, step_fn, check_done=False
):
    """
    Execute a single phase of the environment update.

    Args:
        states (list): Current states for each environment instance.
        active (list of bool): Flags indicating which environments are still active.
        state_history (list of lists): History of states for each environment.
        info_history (list of lists): History of info for each environment.
        compute_fn (callable): Function that computes the action and phase info.
        step_fn (callable): Function that steps the environment.
        check_done (bool): Whether to check and update the active flag based on done flags.
    """
    # Get the action and associated info for this phase.
    active_idx = [i for i, a in enumerate(active) if a]
    active_states = [state_history[i][-1] for i in active_idx]
    action, phase_info = compute_fn(active_states)

    # Take a step in the environment.
    new_states, done_flags = step_fn(active_states, action)

    # Update each environment instance if it's still active.
    for i, a_idx in enumerate(active_idx):
        state_history[a_idx].append(new_states[i])
        info_history[a_idx].append(phase_info[i])
        if check_done and done_flags[i]:
            active[a_idx] = False


def get_action_ratio(info: list[list[dict]], action: str) -> float:
    num_matching = 0
    tot = 0

    for i in range(len(info)):
        for j in range(len(info[i])):
            if "action" not in info[i][j]:
                continue
            if info[i][j]["action"] == action:
                num_matching += 1
            tot += 1

    return num_matching / tot


def get_all_action_ratios(info: list[list[dict]]) -> dict:
    ratios = {}
    for action in ["accept", "append", "finish"]:
        ratios.update({f"{action}_ratio": get_action_ratio(info, action)})
    return ratios


def dict_mean(info: list[list[dict]]) -> dict:
    vals = {}

    for i in range(len(info)):
        for j in range(len(info[i])):
            for k in info[i][j]:
                if not (type(k) == int or type(k) == float):
                    continue

                if k not in vals:
                    vals[k] = []
                vals[k].append(info[i][j][k])

    means = {k: np.mean(v) for k, v in vals.items()}
    return means


def most_recent(data: list[list[any]]) -> list[any]:
    return [[d[-1]] for d in data]


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as fb:
        config = yaml.load(fb, Loader=yaml.FullLoader)
    return config


@hydra.main(
    config_path="../configs", config_name="evaluate_with_assistant", version_base=None
)
def main(config: EvaluateWithAssistantConfig):
    N = config.num_problems

    random.seed(config.seed)
    np.random.seed(config.seed)

    run = wandb.init(
        project="codegen", group="lcb_eval", config=OmegaConf.to_container(config)
    )

    assistant_generator = initialize_generator(config.assistant.generator)
    human_generator = initialize_generator(config.human.generator)

    assistant = Assistant(assistant_generator, config.assistant)
    human_acceptor = HumanAcceptor(human_generator, config.human)
    human_appender = HumanAppender(human_generator, config.human)

    benchmark = build_prompt_benchmark(config.benchmark)
    random.shuffle(benchmark)
    if N == 0:
        N = len(benchmark)
    benchmark = benchmark[:N]
    env = LCBEnvironment(benchmark)

    states = env.reset()

    # Initialize per-environment history lists.
    state_history = [[state] for state in states]
    info_history = [[] for _ in range(N)]
    active = [True] * N  # Tracks which environments are still active

    wandb.log({"active": N})
    for t in range(config.max_steps):
        if not any(active):
            print("All inactive, finishing")
            break

        print(f"At step {t}")
        wandb_info = {}

        # Assistant phase
        run_phase(
            active,
            state_history,
            info_history,
            assistant,
            env.step_assistant,
            check_done=False,
        )
        wandb_info.update(dict_mean(most_recent(info_history)))

        # Get metrics for logging
        code_before_phase = []
        suggested_completions = []
        for i in range(N):
            if active[i]:
                code_before_phase.append(state_history[i][-1].code)
                suggested_completions.append(state_history[i][-1].suggested_completion)

        # Human acceptor phase
        run_phase(
            active,
            state_history,
            info_history,
            human_acceptor,
            env.step_human,
            check_done=False,
        )
        acceptor_info = most_recent(info_history)
        wandb_info.update(dict_mean(acceptor_info))
        wandb_info.update(get_all_action_ratios(acceptor_info))

        # If the acceptor wrote to finish, then set active to False
        for i in range(N):
            active[i] = active[i] and not info_history[i][-1]["action"] == "finish"

        # Human appender phase (check done)
        run_phase(
            active,
            state_history,
            info_history,
            human_appender,
            env.step_human,
            check_done=True,
        )

        # # If the stripped code is the same as three iterations ago, then set active to False
        # for i in range(N):
        #     if active[i] and len(state_history[i]) >= 1 + 3 * 3:
        #         if (
        #             state_history[i][-1].code.strip()
        #             == state_history[i][-1 - 3 * 3].code.strip()
        #         ):
        #             breakpoint()
        #             active[i] = False

        wandb_info.update(dict_mean(most_recent(info_history)))

        # Log implicit acceptance rate
        implicit_acceptance = []
        for i in range(N):
            if not active[i]:
                continue
            sg = state_history[i][-3].suggested_completion
            prev_state = state_history[i][-3].code
            cur_code = state_history[i][-1].code
            action = info_history[i][-2]["action"]
            if (
                action != "accept"
                and cur_code.startswith(sg)
                and len(sg) > len(prev_state)
            ):
                implicit_acceptance.append(1)
            else:
                implicit_acceptance.append(0)
        wandb_info.update({"implicit_acceptance_rate": np.mean(implicit_acceptance)})

        # Log data for the table
        code_before_phase = []
        suggested_completions = []
        question_contents = []
        human_acceptor_action = []
        human_acceptor_reasoning = []
        human_appender_actions = []
        for i in range(N):
            if active[i]:
                code_before_phase.append(state_history[i][-3].code)
                suggested_completions.append(state_history[i][-3].suggested_completion)
                question_contents.append(state_history[i][-1].problem.question_content)
                human_acceptor_action.append(info_history[i][-2]["action"])
                human_acceptor_reasoning.append(info_history[i][-2]["reasoning"])
                human_appender_actions.append(state_history[i][-1].code)

        table_data = list(
            zip(
                question_contents,
                code_before_phase,
                suggested_completions,
                human_acceptor_action,
                human_acceptor_reasoning,
                human_appender_actions,
            )
        )
        assistant_table = wandb.Table(
            columns=[
                "Problem",
                "Code Before Phase",
                "Suggested Completion",
                "Human Acceptor Action",
                "Human Acceptor Reasoning",
                "Human Appender Action",
            ],
            data=table_data,
        )
        wandb_info.update({"assistant_table": assistant_table})
        wandb_info["active"] = sum(active)
        wandb.run.log(wandb_info)

    output_code = [[states[-1].code] for states in state_history]
    eval_samples = [instance.get_evaluation_sample() for instance in benchmark]
    eval_results = codegen_metrics(
        eval_samples,
        output_code,
        num_process_evaluate=config.num_process_evaluate,
        timeout=config.timeout,
        debug=config.debug,
    )

    pass_at_1 = eval_results[0]["pass@1"]
    print(f"Pass@1: {pass_at_1}")

    output_dir = os.path.join(SAVE_ROOT, "evaluate", f"{run.id}")
    os.makedirs(output_dir)
    print(f"Saving to {output_dir}")

    save_results = []
    for i in range(N):
        data = benchmark[i].insert_output_evaluation(
            output_list=output_code[i],
            code_list=output_code[i],
            graded_list=eval_results[1][i][0],
            errors=json.loads(eval_results[2][i][0]),
        )
        save_results.append(data)

    asst_metric, asst_metric_std_err = assistant_metric(
        state_history, save_results, info_history
    )

    # Get the lengths of the suggested completions
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    completion_lengths = []
    for i in range(len(state_history)):
        for j in range(1, len(state_history[i]), 3):
            completion_only = state_history[i][j].suggested_completion[
                len(state_history[i][j].code) :
            ]
            completion_lengths.append(len(tokenizer(completion_only)["input_ids"]))
    mean_comp_length = np.mean(completion_lengths)
    std_err = np.std(completion_lengths) / np.sqrt(len(completion_lengths))
    print(f"Mean completion length in tokens: {mean_comp_length} \nStd err: {std_err}")

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(save_results, f, indent=4)
    with open(os.path.join(output_dir, "info_history.json"), "w") as f:
        json.dump(info_history, f, indent=4)
    with open(os.path.join(output_dir, "state_history.pkl"), "wb") as f:
        pickle.dump(state_history, f)
    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        OmegaConf.save(config, f)

    wandb_summary = {"pass@1": pass_at_1, "output_dir": output_dir}
    wandb_summary.update(get_all_action_ratios(info_history))
    wandb_summary.update(dict_mean(info_history))
    wandb_summary.update({"mean_comp_length": mean_comp_length, "std_err": std_err})
    wandb_summary.update(
        {
            "assistant_metric": asst_metric,
            "assistant_metric_std_err": asst_metric_std_err,
        }
    )
    run.summary.update(wandb_summary)

    breakpoint()


if __name__ == "__main__":
    main()

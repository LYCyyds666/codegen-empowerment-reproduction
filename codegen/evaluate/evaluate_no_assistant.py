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
from codegen.evaluate.base import EvaluateConfig
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


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as fb:
        config = yaml.load(fb, Loader=yaml.FullLoader)
    return config


@hydra.main(
    config_path="../configs", config_name="evaluate_no_assistant", version_base=None
)
def main(config: EvaluateConfig):
    N = config.num_problems

    random.seed(config.seed)
    np.random.seed(config.seed)

    run = wandb.init(
        project="codegen", group="lcb_eval", config=OmegaConf.to_container(config)
    )

    human_generator = initialize_generator(config.human.generator)
    human_appender = HumanAppender(human_generator, config.human)

    benchmark = build_prompt_benchmark(config.benchmark)

    if N > 0:
        random.shuffle(benchmark)
        benchmark = benchmark[:N]
    else:
        N = len(benchmark)

    env = LCBEnvironment(benchmark)

    states = env.reset()

    # Human appender
    action, phase_info = human_appender(states)
    new_states, _ = env.step_human(states, action)

    breakpoint()
    wandb.log(dict_mean(phase_info))
    output_code = [[state.code] for state in new_states]

    breakpoint()
    if not config.benchmark.override_benchmark_path:
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

        save_results = []
        for i in range(N):
            data = benchmark[i].insert_output_evaluation(
                output_list=output_code[i],
                code_list=output_code[i],
                graded_list=eval_results[1][i][0],
                errors=json.loads(eval_results[2][i][0]),
            )
            save_results.append(data)
    else:
        save_results = []
        pass_at_1 = 0

    output_dir = os.path.join(SAVE_ROOT, "evaluate", f"{run.id}")
    os.makedirs(output_dir)
    print(f"Saving to {output_dir}")
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(save_results, f, indent=4)
    with open(os.path.join(output_dir, "info_history.json"), "w") as f:
        json.dump(phase_info, f, indent=4)
    with open(os.path.join(output_dir, "state_history.pkl"), "wb") as f:
        pickle.dump(new_states, f)
    with open(os.path.join(output_dir, "config.yaml"), "w") as f:
        OmegaConf.save(config, f)

    # Mean code length
    code_lengths = [len(state.code) for state in new_states]
    mean_code_length = np.mean(code_lengths)

    wandb_summary = {
        "pass@1": pass_at_1,
        "output_dir": output_dir,
        "mean_code_length": mean_code_length,
        "std_code_length": np.std(code_lengths),
    }
    wandb.run.summary.update(wandb_summary)
    breakpoint()


if __name__ == "__main__":
    main()

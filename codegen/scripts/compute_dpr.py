import argparse
import pickle
import os
import json
import numpy as np

WRITING_CHAR_WEIGHT = 0.5
READING_CHAR_WEIGHT = 0.1


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--discount_per_token", type=float, default=0.999)
    args = parser.parse_args()
    return args


def main():
    args = get_config()

    with open(os.path.join(args.save_dir, "state_history.pkl"), "rb") as f:
        state_histories = pickle.load(f)

    with open(os.path.join(args.save_dir, "results.json"), "r") as f:
        results = json.load(f)

    characters_suggested: list[list[int]] = []
    characters_appended: list[list[int]] = []
    for sh in state_histories:
        characters_suggested.append([])
        characters_appended.append([])
        for i in range(len(sh)):
            # We just finished an append step
            if i % 3 == 0 and i > 0:
                characters_appended[-1].append(len(sh[i].code) - len(sh[i - 1].code))
            # We are at a suggestion stage
            elif i % 3 == 1:
                characters_suggested[-1].append(
                    len(sh[i].suggested_completion) - len(sh[i].code)
                )

    pass_at_1 = [r["pass@1"] for r in results]

    for sugg, app, sh in zip(
        characters_suggested, characters_appended, state_histories
    ):
        assert sum(app) <= len(sh[-1].code)

    # Compute the DPR
    discounted_rewards = []
    for sugg, app, passing_tests in zip(
        characters_suggested, characters_appended, pass_at_1
    ):
        weighted_characters = (
            sum(sugg) * READING_CHAR_WEIGHT + sum(app) * WRITING_CHAR_WEIGHT
        )
        passed = passing_tests == 1
        discounted_reward = args.discount_per_token**weighted_characters * passed
        discounted_rewards.append(discounted_reward)

    # Compute the DPR
    print(f"DPR: {sum(discounted_rewards) / len(discounted_rewards)}")
    print(f"Stderr: {np.std(discounted_rewards) / np.sqrt(len(discounted_rewards))}")
    breakpoint()

    print(args)


if __name__ == "__main__":
    main()

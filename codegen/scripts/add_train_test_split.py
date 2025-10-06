import argparse

from datasets import load_dataset


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    args = parser.parse_args()

    return args


def main():
    config = get_config()
    dataset = load_dataset(config.dataset, split="train")

    dataset = dataset.train_test_split(test_size=0.2)
    dataset.push_to_hub(config.dataset + "_f")


if __name__ == "__main__":
    main()

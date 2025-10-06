import datetime
import argparse
from datasets import Dataset, concatenate_datasets, load_dataset


def difficulty(rating: int) -> str:
    if rating < 1000:
        return "easy"
    elif rating < 2000:
        return "medium"
    else:
        return "hard"


def format_as_problems(df):
    df.rename(
        {"name": "question_title", "index": "question_id", "contestId": "contest_id"},
        inplace=True,
        axis=1,
    )
    df = df.assign(
        platform="codeforces",
        contest_date=str(datetime.datetime.now()),
        starter_code="",
        public_test_cases="[]",
        private_test_cases="[]",
        metadata="[]",
    )
    df["difficulty"] = df["rating"].map(difficulty)
    df["question_content"] = df[
        ["problem-description", "input-specification", "output-specification"]
    ].apply(lambda x: "".join(x), axis=1)
    return df[
        [
            "question_title",
            "question_content",
            "platform",
            "question_id",
            "contest_id",
            "contest_date",
            "starter_code",
            "difficulty",
            "public_test_cases",
            "private_test_cases",
            "metadata",
        ]
    ]


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="MatrixStudio/Codeforces-Python-Submissions"
    )
    parser.add_argument("--output_dataset", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    config = get_config()
    ds = load_dataset(config.dataset)
    ds = concatenate_datasets([ds["train"], ds["test"]])

    df = ds.to_pandas()
    print("De-duping")
    df.drop_duplicates(subset=["contestId", "index"], inplace=True, ignore_index=True)
    print("Formatting")
    df = format_as_problems(df)
    ds = Dataset.from_pandas(df)
    ds.push_to_hub(config.output_dataset, private=True)

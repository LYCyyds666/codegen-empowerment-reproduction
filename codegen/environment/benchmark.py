from lcb_runner.benchmarks import (
    CodeGenerationProblem,
    load_code_generation_dataset,
    load_code_generation_dataset_not_fast,
)
from codegen.evaluate.base import BenchmarkConfig
import pickle
from datasets import load_dataset
from datasets import DatasetDict


def ds_row_to_code_generation_problem(row: dict) -> CodeGenerationProblem:
    return CodeGenerationProblem(
        question_title=row["question_title"],
        question_content=row["question_content"],
        platform=row["platform"],
        question_id=row["question_id"],
        contest_id=row["contest_id"],
        contest_date=row["contest_date"],
        starter_code=row["starter_code"],
        difficulty=row["difficulty"],
        public_test_cases=(
            row["public_test_cases"] if "public_test_cases" in row else "[]"
        ),
        private_test_cases=(
            row["private_test_cases"] if "private_test_cases" in row else "[]"
        ),
        metadata=row["metadata"] if "metadata" in row else "{}",
    )


def build_prompt_benchmark(config: BenchmarkConfig) -> list[CodeGenerationProblem]:
    if config.override_benchmark_path:
        if config.override_benchmark_path.endswith(".pkl"):
            dataset = pickle.load(open(config.override_benchmark_path, "rb"))
        else:
            dataset = load_dataset(config.override_benchmark_path)

        benchmark = []

        # Flatten if DatasetDict
        if isinstance(dataset, DatasetDict):
            flattened_dataset = []
            for split in dataset.keys():
                flattened_dataset.extend(dataset[split])
            dataset = flattened_dataset

        for row in dataset:
            benchmark.append(ds_row_to_code_generation_problem(row))

    elif config.not_fast:
        benchmark = load_code_generation_dataset_not_fast(config.release_version)
    else:
        benchmark = load_code_generation_dataset(
            config.release_version,
            start_date=config.start_date,
            end_date=None,
        )
    benchmark = sorted(benchmark, key=lambda x: x.question_id)

    return benchmark

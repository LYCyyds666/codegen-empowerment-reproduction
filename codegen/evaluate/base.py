from dataclasses import dataclass, field
from omegaconf import MISSING

from codegen.policies.base import PolicyConfig
from hydra.core.config_store import ConfigStore


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark."""

    not_fast: bool = False
    release_version: str = "release_v5"
    start_date: str = "2024-01-01"
    override_benchmark_path: str | None = (
        None  # If provided, use this benchmark instead of the one in the config. Will not evaluate on it.
    )


@dataclass
class RolloutConfig:
    """Configuration for rollout on training dataset."""

    seed: int = 100
    max_steps: int = 30
    num_problems: int = 2000
    dataset: str = MISSING
    human: PolicyConfig = MISSING
    benchmark: BenchmarkConfig = MISSING
    save_as_dataset: bool = False
    save_dir: str = MISSING
    num_futures: int = MISSING
    run_test_cases: bool = False


@dataclass
class RolloutWithAssistantConfig(RolloutConfig):
    """Configuration for rollout on training dataset with assistant."""

    assistant: PolicyConfig = MISSING
    resample_state: bool = False
    filter_correct: bool = False


@dataclass
class EvaluateWithAssistantConfig(RolloutWithAssistantConfig):
    """Configuration for evaluating a model with assistant."""

    num_process_evaluate: int = 12
    timeout: int = 6
    debug: bool = False
    run_test_cases: bool = True


@dataclass
class EvaluateConfig(RolloutConfig):
    """Configuration for evaluating a model without assistant."""

    num_process_evaluate: int = 12
    timeout: int = 6
    debug: bool = False
    run_test_cases: bool = True


cs = ConfigStore.instance()
cs.store(name="rollout_no_assistant", node=RolloutConfig)
cs.store(group="assistant", name="policy", node=PolicyConfig)
cs.store(group="human", name="policy", node=PolicyConfig)
cs.store(group="benchmark", name="config", node=BenchmarkConfig)
cs.store(name="rollout_with_assistant", node=RolloutWithAssistantConfig)
cs.store(name="evaluate_with_assistant", node=EvaluateWithAssistantConfig)
cs.store(name="evaluate_no_assistant", node=EvaluateConfig)

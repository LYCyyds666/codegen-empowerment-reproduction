from dataclasses import dataclass, field
from omegaconf import MISSING
from typing import Optional
from hydra.core.config_store import ConfigStore


@dataclass
class LLMConfig:
    """Configuration for the LLM serving."""

    tensor_parallel_size: int = 4  # Number of tensor parallel workers
    pipeline_parallel_size: int = 1  # Number of pipeline parallel workers
    max_model_len: int = 8000  # Maximum model context length
    enable_prefix_caching: bool = True  # Whether to enable prefix caching


@dataclass
class SamplingConfig:
    """Configuration for text generation sampling."""

    temperature: float = 0.6  # Sampling temperature
    top_p: float = 0.9  # Nucleus sampling parameter
    max_tokens: int = 2000  # Maximum tokens to generate
    min_tokens: int = 0  # Minimum tokens to generate
    logprobs: int = 1  # Number of logprobs to return
    detokenize: bool = False  # Whether to detokenize output


@dataclass
class GeneratorConfig:
    """Configuration for the generator component."""

    model_name: str = MISSING  # Name/path of the model to use (required)
    served: bool = False  # Whether the model is served
    llm: LLMConfig = field(default_factory=LLMConfig)  # LLM serving configuration
    sampling: SamplingConfig = MISSING  # Sampling configuration
    lora_path: Optional[str] = field(default=None)  # Path to the LORA weights
    tokenizer_path: Optional[str] = field(default=None)  # Path to the tokenizer
    ip: str = "localhost"  # IP to use for the served model
    ports: Optional[list[int]] = field(
        default=None
    )  # Ports to use for the served model
    min_likelihood_threshold: Optional[float] = field(
        default=None
    )  # Minimum likelihood threshold. Will truncate the completion at the first token with likelihood below this threshold.


cs = ConfigStore.instance()
cs.store(name="generator", node=GeneratorConfig)
cs.store(name="sampling", node=SamplingConfig)
cs.store(name="llm", node=LLMConfig)

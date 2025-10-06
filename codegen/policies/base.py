from dataclasses import dataclass
from omegaconf import MISSING
from codegen.generators.base import GeneratorConfig
from hydra.core.config_store import ConfigStore


@dataclass
class PolicyConfig:
    """Configuration for the assistant component."""

    generator: GeneratorConfig = MISSING  # Generator configuration
    min_tokens: int = 1  # Minimum tokens
    max_tokens: int = 20  # Maximum tokens for policy decisions
    reasoning_length: int = 100  # Maximum tokens for reasoning for the human policy

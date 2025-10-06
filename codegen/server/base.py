from dataclasses import dataclass, field
from omegaconf import MISSING

from codegen.policies.base import PolicyConfig
from hydra.core.config_store import ConfigStore


@dataclass
class ServerConfig:
    """Configuration for server."""

    seed: int = 100
    head_port: int = 9000
    assistant: PolicyConfig = MISSING
    assistant_name: str = MISSING


@dataclass
class NamedPolicyConfig(PolicyConfig):
    name: str = MISSING


@dataclass
class StudyServerConfig:
    """Configuration for the study server."""

    head_port: int = 9000
    assistants: list[NamedPolicyConfig] = MISSING


cs = ConfigStore.instance()
cs.store(name="eta1", node=ServerConfig)
cs.store(name="eta4", node=ServerConfig)
cs.store(name="untrained", node=ServerConfig)
cs.store(name="sft10", node=ServerConfig)
cs.store(name="untrained10", node=ServerConfig)
cs.store(name="untrained20", node=ServerConfig)
cs.store(name="study_server", node=StudyServerConfig)

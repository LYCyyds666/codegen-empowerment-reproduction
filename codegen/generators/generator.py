from dataclasses import dataclass

from codegen.environment import ChatPrompt
from codegen.generators.base import GeneratorConfig


@dataclass
class LLMOutput:
    logprobs: list[dict[int, float]]
    text: str


class Generator:
    def __init__(self, config: GeneratorConfig):
        self.config = config

    def __call__(
        self, prompts: list[ChatPrompt], continue_final_message: bool = False, **kwargs
    ) -> list[LLMOutput]:
        raise NotImplementedError()

from codegen.environment import Action, State
from codegen.generators import Generator
from codegen.policies.base import PolicyConfig


class Policy:
    def __init__(self, generator: Generator, config: PolicyConfig):
        self.generator = generator
        self.config = config

    def __call__(self, states: list[State]) -> tuple[list[Action], list[dict]]:
        raise NotImplementedError()

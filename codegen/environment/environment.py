from dataclasses import dataclass
from typing import NewType, Optional

from lcb_runner.benchmarks import CodeGenerationProblem

ChatPrompt = NewType("ChatPrompt", list[dict[str, str]])
Benchmark = list[CodeGenerationProblem]


@dataclass
class Action:
    code: str
    code_iids: Optional[list[int]] = None
    # info: dict[str, any]


@dataclass
class State:
    code: str
    suggested_completion: str | None
    problem: CodeGenerationProblem
    code_iids: Optional[list[int]] = None


class LCBEnvironment:
    def __init__(self, benchmark: Benchmark):
        self.benchmark = benchmark

    def reset(self) -> list[State]:
        states = []
        for problem in self.benchmark:
            state = State(
                code=problem.starter_code,
                suggested_completion=None,
                problem=problem,
            )
            states.append(state)
        return states

    def step_human(
        self, states: list[State], actions: list[Action]
    ) -> tuple[list[State], list[bool]]:
        next_states = []
        done = []
        for state, action in zip(states, actions):
            next_state = State(
                code=action.code,
                suggested_completion=None,
                problem=state.problem,
            )
            next_states.append(next_state)
            done.append(action.code == state.code)

        return next_states, done

    def step_assistant(
        self, states: list[State], actions: list[Action]
    ) -> tuple[list[State], list[bool]]:
        next_states = []
        done = []
        for state, action in zip(states, actions):
            next_state = State(
                code=state.code,
                suggested_completion=action.code,
                problem=state.problem,
            )
            next_states.append(next_state)
            done.append(False)

        return next_states, done

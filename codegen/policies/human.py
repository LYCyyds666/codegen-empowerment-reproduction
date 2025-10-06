import numpy as np

from codegen.environment import Action, ChatPrompt, State
from codegen.generators import Generator
from codegen.policies.policy import Policy
from codegen.policies.base import PolicyConfig
from codegen.utils import clean_snippet, git_diff_string
from lcb_runner.benchmarks import CodeGenerationProblem

EXAMPLE_PROGRAMS = [
    """
def twoSum(self, nums: List[int], target: int) -> List[int]:
    numMap = {{}}
    n = len(nums)
    k =

    # Build the hash table
    for i in range(n:
        numMap[nums[i]] = i

    # Find the complement
    for i in range(n)
""",
    """
# Python program for Dijkstra's single
# source shortest path algorithm. The program is
# for adjacency matrix representation of the graph

# Library for INT_MAX
import sys


class Graph():

    def __init__(self, vertices):
        self.V = vertices
        self.graph = [[0 for column in range(vertices)]
                      for row in range(vertices)]

    def printSolution(self, dist):
        print("Vertex \tDistance from Source")
        for node in range(self.V):
""",
    """
import asyncio

async def chatter(text: str):
    await asyncio.sleep(3)
    print(text)

async def main(messages: list[str]):
    with asyncio.TaskGroup() as tg:
        for message in messages:
""",
    """
def is_multiple_of_two(num: int) -> bool:

""",
]
EXAMPLE_SUGGESTIONS = [
    """
def twoSum(self, nums: List[int], target: int) -> List[int]:
    numMap = {{}}
    n = len(nums)
    k =

    # Build the hash table
    for i in range(n:
        numMap[nums[i]] = i

    # Find the complement
    for i in range(n):
        complement = target - nums[i]
""",
    """
# Python program for Dijkstra's single
# source shortest path algorithm. The program is
# for adjacency matrix representation of the graph

# Library for INT_MAX
import sys


class Graph():

    def __init__(self, vertices):
        self.V = vertices
        self.graph = [[0 for column in range(vertices)]
                      for row in range(vertices)]

    def printSolution(self, dist):
        print("Vertex \tDistance from Source")
        for node in range(self.V):
            print(node, "\t", dist[node])

    def minDistance(self, dist, sptSet):
        # Initialize minimum distance for next node
        min = sys.maxsize
""",
    """
import asyncio

async def chatter(text: str):
   await asyncio.sleep(3)
   print(text)

async def main(messages: list[str]):
   with asyncio.TaskGroup() as tg: 
       for message in messages:
           tg.create_task(chatter(message))
""",
    """
def is_multiple_of_two(num: int) -> bool:
    return num % 2 == 0
""",
]
EXAMPLE_RESPONSES = [
    "The suggestion is not correct. The complement should be calculated as -(nums[i] - target) instead. Therefore, I would like to reject",
    "The suggestion correctly begins a minDistance function, which will be useful for Djikstra. Therefore, I would like to accept",
    "The suggestion correctly creates a task in asyncio. Therefore, I would like to accept",
    "The suggestion correctly infers the question, which is to find if a number is a multiple of two. Therefore, I would like to accept",
]


def format_code_and_suggestion(code: str, suggestion: str, json: bool = True) -> str:
    return f"""
## You have written the following code:
```python
{code}
```

## Suggested Completion
Here is what your code would look like with a suggested completion: 
```python
{suggestion}
```

## Suggested Completion diff
For clarity, here is the diff between your current code and the suggested completion code:
{git_diff_string(code, suggestion)} 

## Instructions:
What do you think of the suggested completion? Do you think it is solving the question correctly, or does it introduce a bug or error?
Please write down your thoughts. You are not allowed to write any new code in your response, only your thoughts on whether the suggested completion helps you on your way to solving the problem, or otherwise improves the quality of your code.
It is ok if the the suggested completion is incomplete, because you will be prompted to append to it later.
You are also not able to take any actions at this stage.
"""


SYSTEM_MESSAGE_GENERIC = """You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. 
*Please do not provide any sample outputs or testcases in your response. Additionally, you are only allowed to solve the problem *ONCE*. 
Do not attempt to retry your solution if you are unhappy with it.
For example, if your solution is in a function called `solve`, you should only define one function called `solve`. DO NOT try to retry it if you think it has a bug.
For example, you should not write `solve2` if you think `solve` has a bug. Only the first solution will be counted, so simply stop writing once the first solution is finished--even if it is not correct."""

FORMATTING_MESSAGE_WITH_STARTER_CODE = "You will use the following starter code to write the solution to the problem and enclose your code within delimiters."
FORMATTING_WITHOUT_STARTER_CODE = "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT."

SYSTEM_MESSAGE_ACCEPTOR = f"""You are an expert Python programming agent. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. 

The way you will compose your program is by interacting with the file editing interface through tools. 
In addition, you will be assisted by an automated assistant that will sometimes propose suggested completions to whatever you are typing. 
At every step you will first write your thoughts on the suggestion and whether it correctly continues your solution, or instead introduces bugs.
Please *only consider the assistant suggestion* in your decision to accept the suggestion, reject the suggestion, or finish and submit your code for testing.
Previous code that you have written is not able to be changed, so you should only look at the changes the assistant is proposing.
At the very end of your thought process, you will write one word to specify which action you are taking.
Unless you take the "finish" action, you will be prompted afterwards to write what you would like to append to your program.
Therefore, please accept suggestions as long as they do not introduce bugs, and either help you solve the problem or improve the quality of your code.
It's ok if the the suggested completion is incomplete, because you can always append to it later.

You will first be prompted to write your thoughts on the suggestion.
Afterwords, you will be prompted to write which action you would like to take.

Here are the actions you have access to:
### "accept"
    description: Accepts a suggested completion given by an intelligent assistant. The suggested completion will then be incorporated into the code you have written.
### "reject"
    description: Rejects a suggested completion given by an intelligent assistant. The suggested completion will not be incorporated into the code you have written.
### "finish"
    description: Tells the editor that you have finished writing the program and to run the testcases. Only call this tool if you are confident that your program is finished. You will not be prompted to write any more code after calling this tool.

Remember, you will be able to continue writing your program regardless of whether you accept the suggested completion or not.
As long as the suggested completion does not introduce bugs, and either helps you solve the problem or improves the quality of your code, you should accept it.
DO NOT reject a suggestion because it is "minor" or "short". Only reject the suggestion if it is wrong, introduces a bug, or otherwise sets you back.
"""


def get_generic_question_template_answer(problem: CodeGenerationProblem):
    prompt = f"### Question:\n{problem.question_content}\n\n"
    if problem.starter_code:
        prompt += f"### Format: {FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
        prompt += f"```python\n{problem.starter_code}\n```\n\n"
    else:
        prompt += f"### Format: {FORMATTING_WITHOUT_STARTER_CODE}\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += f"### Answer: (use the provided format with backticks)\n\n"
    return prompt


def format_prompt_generation(state: State) -> ChatPrompt:
    system_message = SYSTEM_MESSAGE_GENERIC
    user_message = get_generic_question_template_answer(state.problem)

    chat_messages = [
        {"role": "system", "content": system_message},
        {
            "role": "user",
            "content": user_message,
        },
    ]
    return ChatPrompt(chat_messages)


def format_prompt_blind_null_from_code(code: str) -> ChatPrompt:
    chat_messages = [
        {
            "role": "system",
            "content": """You are an expert Python programmer. You will be provided with a code snippet and will be asked to finish the program.
You will not be able to see the question--instead, you should just finish the program however you see fit.
""",
        },
        {
            "role": "user",
            "content": f"""Please finish the code below:
```python
{code}
```""",
        },
    ]

    return ChatPrompt(chat_messages)


def format_prompt_blind_null(state: State) -> ChatPrompt:
    return format_prompt_blind_null_from_code(state.code)


def format_prompt_acceptor(state: State) -> ChatPrompt:
    assert state.suggested_completion is not None

    chat_messages = [
        {
            "role": "system",
            "content": SYSTEM_MESSAGE_ACCEPTOR,
        },
        {
            "role": "user",
            "content": get_generic_question_template_answer(state.problem),
        },
        {
            "role": "assistant",
            "content": f"""```python
        {state.code}
        ```""",
        },
        {
            "role": "user",
            "content": f"""{format_code_and_suggestion(state.code, state.suggested_completion, json=False)}""",
        },
    ]
    return ChatPrompt(chat_messages)


class HumanAcceptor(Policy):
    def __init__(self, generator: Generator, config: PolicyConfig):
        super().__init__(generator, config)
        self.reasoning_length = config.reasoning_length

    def __call__(self, states: list[State]) -> tuple[list[Action], list[dict]]:
        prompts = [
            format_prompt_acceptor(
                state,
            )
            for state in states
        ]

        # First, get the thought process
        reasoning_outputs = self.generator(
            prompts,
            continue_final_message=False,
            sampling_overrides={"max_tokens": self.reasoning_length},
        )

        infos = []
        for i in range(len(prompts)):
            prompts[i].extend(
                [
                    {
                        "role": "assistant",
                        "content": reasoning_outputs[i].text,
                    },
                    {
                        "role": "user",
                        "content": f"""Now, please write which action you would like to take. 
Remember, the actions available are "accept" to accept the suggested completion, "reject" to reject the suggested completion, and "finish" to finish writing your code and run the tests.
Please only call "finish" if you are confident that your code is correct and you are ready to run the tests.
""",
                    },
                    {"role": "assistant", "content": "Therefore, I would like to "},
                ]
            )
            infos.append({"reasoning": reasoning_outputs[i].text})

        action_outputs = self.generator(
            prompts,
            continue_final_message=True,
            sampling_overrides={
                "max_tokens": 1,
                "logprobs": 20,
                "top_p": 1,
                "top_k": 20,
            },
        )

        actions = []
        for i in range(len(prompts)):
            logprobs: dict[int, float] = action_outputs[i].logprobs[0]

            # Sum of the probabilities of "accept", "finish", and "append"
            accept_prob = 0
            reject_prob = 0
            finish_prob = 0

            for token, lp in logprobs.items():
                try:
                    token_text = (
                        self.generator.tokenizer.decode([token]).lower().strip()
                    )
                except Exception as e:
                    breakpoint()
                    raise e

                if token_text == "accept":
                    accept_prob += np.exp(lp)
                elif token_text == "reject":
                    reject_prob += np.exp(lp)
                elif token_text == "finish":
                    finish_prob += np.exp(lp)

            if max(accept_prob, reject_prob, finish_prob) == 0:
                reject_prob = 1  # Reject by default

            h_a, h_a_p = max(
                zip(
                    ["accept", "reject", "finish"],
                    [accept_prob, reject_prob, finish_prob],
                ),
                key=lambda x: x[1],
            )
            h_a_lp = np.log(h_a_p)

            if h_a is None:
                h_a = "reject"

            if h_a == "accept":
                code = states[i].suggested_completion
            else:
                code = states[i].code

            info = {
                "accept_prob": accept_prob / (accept_prob + reject_prob + finish_prob),
                "action": h_a,
                "logprob": h_a_lp,
                "all_logprobs": logprobs,
                "action_text_raw": action_outputs[i].text,
            }

            actions.append(Action(code=code))
            infos[i].update(info)

        return actions, infos


class HumanAppender(Policy):
    def __init__(self, generator: Generator, config: PolicyConfig):
        super().__init__(generator, config)
        self.max_tokens = config.max_tokens

    def __call__(self, states: list[State]) -> tuple[list[Action], list[dict]]:
        prompts = [format_prompt_generation(state) for state in states]

        for state, prompt in zip(states, prompts):
            prompt.append(
                {
                    "role": "assistant",
                    "content": f"""
```python
{state.code}""",
                }
            )

        append_outputs = self.generator(
            prompts,
            continue_final_message=True,
            sampling_overrides={"max_tokens": self.max_tokens},
        )

        actions = []
        for state, append in zip(states, append_outputs):
            appended = clean_snippet(state.code + append.text)
            actions.append(Action(code=appended))
        infos = [{} for _ in states]

        return actions, infos


class HumanBlindNull(Policy):
    """
    A human null policy that does not see the question, and instead just appends to the code.
    """

    def __init__(self, generator: Generator, config: PolicyConfig):
        super().__init__(generator, config)
        self.max_tokens = config.max_tokens

    def __call__(self, states: list[State]) -> tuple[list[Action], list[dict]]:
        # We require each state to have a code_iids field
        for state in states:
            assert (
                state.code_iids is not None
            ), "HumanBlindNull requires a code_iids field in the state"

        prompts = [format_prompt_blind_null(state) for state in states]

        for state, prompt in zip(states, prompts):
            prompt.append(
                {
                    "role": "assistant",
                    "content": f"""```python
{state.code}""",
                }
            )

        append_outputs = self.generator(
            prompts,
            continue_final_message=True,
            sampling_overrides={"max_tokens": self.max_tokens, "detokenize": False},
        )

        tokenizer = self.generator.tokenizer
        actions = []

        for state, append in zip(states, append_outputs):
            appended = clean_snippet(state.code + append.text)

            append_tokens = [next(iter(lp.keys())) for lp in append.logprobs]
            code_iids = state.code_iids + append_tokens

            try:
                assert (
                    clean_snippet(tokenizer.decode(code_iids, skip_special_tokens=True))
                    == appended
                )
            except AssertionError as e:
                print(f"AssertionError: {e}")

            actions.append(Action(code=appended, code_iids=code_iids))
        infos = [{} for _ in states]
        return actions, infos

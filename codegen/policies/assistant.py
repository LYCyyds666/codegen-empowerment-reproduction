from codegen.environment import Action, ChatPrompt, State
from codegen.policies.policy import Policy
from codegen.utils import clean_snippet
from codegen.utils.utils import setup_tokenizer_special_tokens
from codegen.policies.base import PolicyConfig

ASSISTANT_SYSTEM_MESSAGE = """
You are assisting a human in a python code generation task. Your role is to provide suggested completions given
what they have already typed. Please try to infer what the human wants the next piece of code to be given the
code they have already written. If they have not written any code, please provide a good start to their program, such as with import statements or function definitions. 

The way you will compose your suggestion is by providing the next version of the code which would replace the current code. 
Please re-type the current code and then add in your suggested completion.
DO NOT output any other text, including no quotation marks.

## Remember to always re-type the code written so far and then add in your suggested completion.
If you don't re-type the code written so far *exactly as it is written* (with all of the functions, comments, import statements, etc)
an error will be raised.
"""


def format_user_prompt(code_to_complete: str) -> str:
    return f"""Now it's your turn! Please provide a completion for the following code:
```python
{code_to_complete}
```"""


def format_assistant_response(assistant_message: str) -> str:
    return f"""Here is my suggested completion:
```python
{assistant_message}"""


ASSISTANT_FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "content": format_user_prompt(
            """def twoSum(self, nums: List[int], target: int) -> List[int]:
    numMap = {}
    n = len(nums)

    # Build the hash table
    for i in range(n):
        numMap[nums[i]] = i

    # Find the complement
    for i in range(n):
"""
        ),
    },
    {
        "role": "assistant",
        "content": format_assistant_response(
            """def twoSum(self, nums: List[int], target: int) -> List[int]:
    numMap = {}
    n = len(nums)

    # Build the hash table
    for i in range(n):
        numMap[nums[i]] = i

    # Find the complement
    for i in range(n):
        complement = target - nums[i]```"""
        ),
    },
    {
        "role": "user",
        "content": format_user_prompt("""def whoami(name: """),
    },
    {
        "role": "assistant",
        "content": format_assistant_response(
            """def whoami(name: str, age: int) -> str:
```"""
        ),
    },
]


def format_assistant_prompts(
    code_to_complete: str, assistant_message: str = None
) -> ChatPrompt:
    """
    Takes the code written so far by the human and returns an assistant prompt to give a suggestion.
    """

    prompts = [
        {
            "role": "system",
            "content": ASSISTANT_SYSTEM_MESSAGE,
        }
    ]
    prompts.extend(ASSISTANT_FEW_SHOT_EXAMPLES)
    prompts.extend(
        [
            {
                "role": "user",
                "content": format_user_prompt(code_to_complete),
            },
            {
                "role": "assistant",
                "content": format_assistant_response(
                    code_to_complete if assistant_message is None else assistant_message
                ),
            },
        ]
    )
    return ChatPrompt(prompts)


class Assistant(Policy):
    def __init__(self, generator, config: PolicyConfig):
        super().__init__(generator, config)
        self.max_tokens = config.max_tokens
        self.min_tokens = config.min_tokens

    def __call__(self, states: list[State]) -> tuple[list[Action], list[dict]]:
        codes = [state.code for state in states]

        # Now add the assistant to the prompts.
        assistant_prompts = [format_assistant_prompts(code) for code in codes]

        raw_suggested_completions = self.generator(
            assistant_prompts,
            continue_final_message=True,
            sampling_overrides={
                "min_tokens": self.min_tokens,
                "max_tokens": self.max_tokens,
                "stop_token_ids": [self.generator.tokenizer.eos_token_id],
                "include_stop_str_in_output": True,
                "skip_special_tokens": False,
            },
        )

        suggested_completions = []
        for code, sg in zip(codes, raw_suggested_completions):
            sg_text = code + sg.text
            sg_text = clean_snippet(sg_text)
            suggested_completions.append(sg_text)

        actions = [Action(code=sg_text) for sg_text in suggested_completions]
        infos: list[dict] = [{} for _ in states]

        return actions, infos

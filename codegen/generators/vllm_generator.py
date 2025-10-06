import asyncio
import os
from typing import Optional
import math
from openai import AsyncOpenAI
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from dataclasses import asdict
from codegen.generators.generator import *
from codegen.generators.base import GeneratorConfig
from omegaconf import OmegaConf
import numpy as np

os.environ["OPENAI_API_KEY"] = "asdf"


class VLLM(Generator):
    def __init__(self, config: GeneratorConfig):
        super().__init__(config)

        self.model_name = config.model_name
        if config.tokenizer_path is None:
            config.tokenizer_path = config.model_name

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.tokenizer_path, padding_side="left"  # , use_fast=False
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(
        self, prompts: list[str], sampling_params: SamplingParams
    ) -> list[LLMOutput]:
        raise NotImplementedError()

    def merge_sampling_params(self, sampling_overrides: Optional[dict] = None) -> dict:
        sampling_config = OmegaConf.to_container(self.config.sampling)

        if sampling_overrides is not None:
            sampling_config.update(sampling_overrides)
        return sampling_config

    def __call__(
        self,
        prompts: list[ChatPrompt],
        continue_final_message: bool = False,
        sampling_overrides: Optional[dict] = None,
        **kwargs,
    ) -> list[LLMOutput]:
        """
        prompts: list[ChatPrompt]
        continue_final_message: bool
        sampling_overrides: Optional[dict]
        """

        sampling_params = SamplingParams.from_optional(
            **self.merge_sampling_params(sampling_overrides)
        )

        # We need to convert from chat format to string
        converted_prompts: list[str] = []
        for i in range(len(prompts)):
            tokenized = self.tokenizer.apply_chat_template(
                prompts[i],
                add_generation_prompt=not continue_final_message,
                truncation=False,
                padding=False,
                continue_final_message=continue_final_message,
            )

            chat_prompt = self.tokenizer.decode(tokenized)
            if continue_final_message:
                # Sometimes the tokenizer removes trailing spaces. We still need these.
                j = -1
                while prompts[i][-1]["content"][j] != chat_prompt[-1]:
                    j -= 1

                if j != -1:
                    chat_prompt += prompts[i][-1]["content"][j + 1 :]

            converted_prompts.append(chat_prompt)

            if continue_final_message and converted_prompts[-1].endswith(
                self.tokenizer.eos_token
            ):
                converted_prompts[-1] = converted_prompts[-1][
                    : -len(self.tokenizer.eos_token)
                ]

        vllm_outputs = self.generate(converted_prompts, sampling_params)

        return vllm_outputs


class VLLMLocal(VLLM):
    def __init__(self, config: GeneratorConfig):
        super().__init__(config)

        enable_lora = config.lora_path is not None

        self.llm = LLM(
            model=config.model_name,
            tokenizer=config.tokenizer_path,
            enable_lora=enable_lora,
            enforce_eager=True,
            disable_custom_all_reduce=True,
            trust_remote_code=True,
            tensor_parallel_size=config.llm.tensor_parallel_size,
            pipeline_parallel_size=config.llm.pipeline_parallel_size,
            max_model_len=config.llm.max_model_len,
            enable_prefix_caching=config.llm.enable_prefix_caching,
        )

        self.lora_config = None
        self.enable_lora = enable_lora
        if enable_lora:
            self.lora_request = LoRARequest(
                "lora-rq-1", 1, lora_local_path=config.lora_path
            )
        self.min_likelihood_threshold = config.min_likelihood_threshold

    def generate(self, prompts: list[str], sampling_params: SamplingParams):
        assert sampling_params.n == 1, "Only supporting n=1 for sampling params"

        if self.enable_lora:
            outputs = self.llm.generate(
                prompts, sampling_params, lora_request=self.lora_request
            )
        else:
            outputs = self.llm.generate(prompts, sampling_params)

        vllm_outputs = []
        for output in outputs:
            completion = output.outputs[0]

            formatted_logprobs = []
            for logprobs in completion.logprobs:
                formatted_logprobs.append({})
                for token_id, lp in logprobs.items():
                    # decoded_token = self.tokenizer.decode([token_id])
                    formatted_logprobs[-1][token_id] = lp.logprob
            if self.min_likelihood_threshold is not None:
                logprob_threshold = np.log(self.min_likelihood_threshold)
                for i, (token_id, logprobs) in enumerate(
                    zip(completion.token_ids, formatted_logprobs)
                ):
                    logprob_threshold -= logprobs[token_id]
                    try:
                        if logprob_threshold >= 0:
                            completion.token_ids = completion.token_ids[:i]
                            formatted_logprobs = formatted_logprobs[:i]

                            if (
                                len(completion.token_ids) == 0
                                or completion.token_ids[-1]
                                != self.tokenizer.eos_token_id
                            ):
                                completion.token_ids.append(self.tokenizer.eos_token_id)
                                formatted_logprobs.append(
                                    {self.tokenizer.eos_token_id: 1}
                                )
                            break
                    except Exception as e:
                        breakpoint()
                        raise e

            try:
                vllm_outputs.append(
                    LLMOutput(
                        logprobs=formatted_logprobs,
                        text=self.tokenizer.decode(
                            completion.token_ids, skip_special_tokens=True
                        ),
                    )
                )
            except Exception as e:
                breakpoint()
                raise e

        return vllm_outputs


class VLLMServed(VLLM):
    def __init__(self, config: GeneratorConfig):
        super().__init__(config)

        self.clients = []
        for port in config.ports:
            self.clients.append(
                AsyncOpenAI(base_url=f"http://{config.ip}:{port}/v1", api_key=None)
            )

    async def single_threaded_generate(
        self, prompts: list[str], sampling_params: SamplingParams, client: AsyncOpenAI
    ) -> list[LLMOutput]:
        completions = None
        for retry_count in range(5):
            try:
                logprobs = max(sampling_params.logprobs or 1, 1)
                completions = await client.completions.create(
                    model=self.model_name,
                    prompt=prompts,
                    logprobs=logprobs,
                    max_tokens=sampling_params.max_tokens or 2000,
                    temperature=sampling_params.temperature,
                    top_p=sampling_params.top_p,
                    n=1,
                    timeout=None,
                )
                break
            except Exception as e:
                if retry_count >= 4:
                    raise e

                print(
                    f"Error raised, retrying (retry_count={retry_count}) for base_url: {client.base_url}"
                )

        outputs: list[LLMOutput] = []
        for i in range(len(prompts)):
            completion = completions.choices[i * sampling_params.n]

            logprobs_original = completion.logprobs.top_logprobs
            logprobs_decorrupted = []

            # The logprobs will be returned like "Ġsuggested", so we need to decorrupt it
            for logprob in logprobs_original:
                logprobs_decorrupted.append({})
                for token, lp in logprob.items():
                    logprobs_decorrupted[-1][
                        self.tokenizer.convert_tokens_to_ids(token)
                    ] = lp

            outputs.append(
                LLMOutput(logprobs=logprobs_decorrupted, text=completion.text)
            )

        return outputs

    async def distribute_generation(
        self, prompts: list[str], sampling_params: SamplingParams
    ) -> list[LLMOutput]:
        tasks = []
        batch_size = int(math.ceil(len(prompts) / len(self.clients)))
        for i in range(len(self.clients)):
            batch_prompts = prompts[
                i * batch_size : min((i + 1) * batch_size, len(prompts))
            ]
            if len(batch_prompts) == 0:
                break

            tasks.append(
                asyncio.create_task(
                    self.single_threaded_generate(
                        batch_prompts, sampling_params, self.clients[i]
                    )
                )
            )
        await asyncio.gather(*tasks)

        outputs = []
        for task in tasks:
            outputs.extend(task.result())

        return outputs

    def generate(
        self, prompts: list[str], sampling_params: SamplingParams
    ) -> list[LLMOutput]:
        print(f">> Generating {len(prompts)} responses")
        if len(prompts) == 0:
            return []

        outputs = asyncio.run(self.distribute_generation(prompts, sampling_params))

        return outputs


def initialize_generator(config: GeneratorConfig):
    if config.served:
        return VLLMServed(config)
    else:
        return VLLMLocal(config)

import os

os.environ["WANDB_PROJECT"] = "codegen"
import time
import torch
from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    set_seed,
    DefaultDataCollator,
)
import gc

import os
from trl import SFTTrainer, SFTConfig

import hydra
from codegen.training.base import (
    TrainingArguments,
)
import pickle
from datasets import DatasetDict

import torch, os


SAVE_ROOT = os.environ["CODEGEN_ROOT"]


def collate(batch):
    return {
        "input_ids": torch.tensor([b["input_ids"] for b in batch]),
        "labels": torch.tensor([b["labels"] for b in batch]),
        "attention_mask": torch.tensor([b["attention_mask"] for b in batch]),
    }


@hydra.main(config_path="../configs", config_name="finetune_agent", version_base=None)
def train(config: TrainingArguments):
    sft_config = SFTConfig(
        **config.sft,
        model_init_kwargs={
            "torch_dtype": torch.bfloat16,
            "attn_implementation": (
                "flash_attention_2" if config.model.use_flash_attn else None
            ),
        },
    )
    set_seed(sft_config.seed)

    if config.data.dataset_name.endswith("pkl"):
        with open(config.data.dataset_name, "rb") as fb:
            dataset = pickle.load(fb)
    else:
        dataset = load_dataset(config.data.dataset_name)

    if not isinstance(dataset, DatasetDict):
        dataset = dataset.train_test_split(test_size=config.data.test_split)

    if config.data.num_train_examples > 0:
        num_train = min(config.data.num_train_examples, len(dataset["train"]))
        num_test = int(config.data.test_split * num_train)

        dataset["train"] = dataset["train"].shuffle()
        dataset["test"] = dataset["test"].shuffle()

        dataset["train"] = dataset["train"].select(range(num_train))
        dataset["test"] = dataset["test"].select(range(num_test))

    dataset = dataset.filter(
        lambda attn_mask: len(attn_mask) == 4096, input_columns="attention_mask"
    )
    trainer = SFTTrainer(
        model=config.model.model_name_or_path,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=collate,  # collator
        processing_class=AutoTokenizer.from_pretrained(config.model.tokenizer_path),
    )
    trainer.accelerator.print(f"{trainer.model}")

    checkpoint = None
    if sft_config.resume_from_checkpoint is not None:
        checkpoint = sft_config.resume_from_checkpoint

    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.model.zero_grad(set_to_none=True)
    trainer.optimizer.state.clear()
    trainer.accelerator.free_memory()
    torch.cuda.empty_cache()
    gc.collect()

    t1 = time.time()

    trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")

    fp32_state = trainer.model.state_dict()  # still fp32

    if trainer.accelerator.is_main_process:
        bf16_state = {k: v.to("cpu").to(torch.bfloat16) for k, v in fp32_state.items()}

        base = trainer.model.module
        base.config.torch_dtype = "bfloat16"

        base.save_pretrained(
            sft_config.output_dir, state_dict=bf16_state, safe_serialization=True
        )

    trainer.accelerator.wait_for_everyone()

    print(f"Model saved in {time.time() - t1} seconds")


if __name__ == "__main__":
    train()

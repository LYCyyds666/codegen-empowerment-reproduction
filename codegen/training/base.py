from dataclasses import dataclass, field
from omegaconf import MISSING, OmegaConf
from typing import Optional
from hydra.core.config_store import ConfigStore
import random
import string
from codegen.policies.base import PolicyConfig
from codegen.evaluate.base import BenchmarkConfig
from codegen.empowerment.base import EmpowermentConfig
from codegen.generators.base import GeneratorConfig


# Define and parse arguments.
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    tokenizer_path: str = field(
        metadata={
            "help": "Path to pretrained tokenizer or tokenizer identifier from huggingface.co/models"
        }
    )
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=64)
    lora_target_modules: Optional[str] = field(
        default="q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj",
        metadata={
            "help": "comma separated list of target modules to apply LoRA layers to"
        },
    )
    use_nested_quant: Optional[bool] = field(
        default=False,
        metadata={"help": "Activate nested quantization for 4bit base models"},
    )
    bnb_4bit_compute_dtype: Optional[str] = field(
        default="float16",
        metadata={"help": "Compute dtype for 4bit base models"},
    )
    bnb_4bit_quant_storage_dtype: Optional[str] = field(
        default="uint8",
        metadata={"help": "Quantization storage dtype for 4bit base models"},
    )
    bnb_4bit_quant_type: Optional[str] = field(
        default="nf4",
        metadata={"help": "Quantization type fp4 or nf4"},
    )
    use_flash_attn: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables Flash attention for training."},
    )
    use_peft_lora: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables PEFT LoRA for training."},
    )
    use_8bit_quantization: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables loading model in 8bit."},
    )
    use_4bit_quantization: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables loading model in 4bit."},
    )
    use_reentrant: Optional[bool] = field(
        default=False,
        metadata={"help": "Gradient Checkpointing param. Refer the related docs"},
    )
    use_loftq: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables LoftQ init for the LoRA adapters when using QLoRA."},
    )
    use_loftq_callback: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Enables LoftQ callback comparing logits of base model to the ones from LoftQ init. Provides better init."
        },
    )
    moe_layer_name: Optional[str] = field(
        default=None,
        metadata={"help": "MOE layer name"},
    )


@dataclass
class DataTrainingArguments:
    dataset_name: str = field(
        metadata={"help": "The preference dataset to use."},
    )
    append_concat_token: Optional[bool] = field(
        default=False,
        metadata={
            "help": "If True, appends `eos_token_id` at the end of each sample being packed."
        },
    )
    add_special_tokens: Optional[bool] = field(
        default=False,
        metadata={
            "help": "If True, tokenizers adds special tokens to each sample being packed."
        },
    )
    train_type: str = field(
        default="sft",
        metadata={"help": "How we are training. May be 'awr', 'wbc', or 'sft'"},
    )
    temperature: float = field(
        default=1.0, metadata={"help": "Temperature for weights."}
    )
    num_train_examples: int = field(
        default=0, metadata={"help": "Number of training examples."}
    )
    test_split: float = field(default=0.2, metadata={"help": "Test split."})


@dataclass
class TrainingArguments:
    model: ModelArguments = MISSING
    data: DataTrainingArguments = MISSING
    sft: dict = field(default_factory=dict)


def uuid6() -> str:
    return "".join(
        random.choice(string.ascii_uppercase + string.digits) for _ in range(6)
    )


OmegaConf.register_new_resolver("uuid6", uuid6, use_cache=True)

cs = ConfigStore.instance()
cs.store(name="model", node=ModelArguments)
cs.store(name="data", node=DataTrainingArguments)
cs.store(name="training", node=TrainingArguments)

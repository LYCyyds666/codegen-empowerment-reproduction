import difflib
import json
import os
from pathlib import Path

import torch
from accelerate import Accelerator
from transformers import AutoTokenizer


def dump_jsonl(messages: list, fname: str | Path) -> None:
    if not os.path.exists(os.path.dirname(fname)):
        os.makedirs(os.path.dirname(fname))

    with open(fname, "w") as outfile:
        for message in messages:
            json.dump(message, outfile)
            outfile.write("\n")



def clean_snippet(snippet) -> str:
    snippet = snippet.replace("```python", "")

    if snippet.strip().startswith("```"):
        snippet = snippet[snippet.index("```") + 3 :]

    if "```" in snippet:
        snippet = snippet[: snippet.index("```")]

    return snippet


def git_diff_string(original_code, new_code):
    diff = difflib.unified_diff(
        original_code.splitlines(),
        new_code.splitlines(),
        fromfile="OriginalCode",
        tofile="NewCode",
        lineterm="",  # Prevents additional newline characters in the output
    )
    return "\n".join(diff)


def accelerator_breakpoint():
    accelerator = Accelerator()
    if accelerator.is_main_process:
        breakpoint()
    accelerator.wait_for_everyone()


def download_model_if_needed(model_path: str) -> str:
    if not os.path.exists(model_path):
        # Try loading from huggingface if it isn't a local model
        from huggingface_hub import snapshot_download

        model_path = snapshot_download(model_path)

    return os.path.join(model_path, "model")


def setup_tokenizer_special_tokens(tokenizer, model=None):
    """
    Setup special tokens for a tokenizer in a model-agnostic way.
    
    Args:
        tokenizer: The tokenizer to configure
        model: Optional model to update generation config
    
    Returns:
        dict: Dictionary with 'eos_token_id' and 'pad_token_id' keys
    """
    # Get the actual EOS token ID from the tokenizer
    eos_token_id = tokenizer.eos_token_id
    
    # Set up padding token if not present
    if tokenizer.pad_token is None:
        # Check if there's an unused special token we can use
        # For Llama models, <|end_of_text|> (128001) is often available
        if hasattr(tokenizer, 'convert_tokens_to_ids'):
            # Try common unused tokens
            for token in ['<|end_of_text|>', '<pad>', '<unk>']:
                token_id = tokenizer.convert_tokens_to_ids(token)
                if token_id != tokenizer.unk_token_id:  # Valid token found
                    tokenizer.pad_token = token
                    tokenizer.pad_token_id = token_id
                    break
        
        # If no suitable token found, use eos_token as fallback
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
    
    pad_token_id = tokenizer.pad_token_id
    
    # Update model generation config if provided
    if model is not None:
        model.generation_config.pad_token_id = pad_token_id
        model.generation_config.eos_token_id = eos_token_id
    
    return {
        'eos_token_id': eos_token_id,
        'pad_token_id': pad_token_id
    }

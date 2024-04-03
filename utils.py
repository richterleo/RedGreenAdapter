import configparser
import time
from contextlib import contextmanager
import torch
import typing

from datetime import datetime
from dataclasses import dataclass, field
from datasets import Dataset, load_dataset
from typing import Optional, Union, Callable, List, Dict

from pathlib import Path

from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerBase
)


from trl import (
    PPOConfig
)

from trl.core import LengthSampler

# Below is an example function to build the dataset. In our case, we use the IMDB dataset
# from the `datasets` library. One should customize this function to train the model on
# its own dataset.
def build_dataset(
    config: PPOConfig, 
    dataset_name: Optional[str] = "allenai/real-toxicity-prompts", 
    tokenizer: PreTrainedTokenizerBase = None,
    input_min_text_length: int=5, 
    input_max_text_length: int=10,
    just_train: bool=True,
    tox_thresh: float=0.3,
    test_size:float=0.2
):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """
    if not tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(config.adapter_model_name)
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset(dataset_name, split="train")

    def filter_fn(sample):
        toxicity = sample["prompt"]["toxicity"]
        return toxicity is not None and toxicity > tox_thresh

    ds = ds.filter(filter_fn, batched=False)

    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        prompt = sample["prompt"]["text"]
        continuation = sample["continuation"]["text"]

        sample["input_ids"] = tokenizer.encode(prompt + continuation)[: input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])  
        
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")

    ds = ds.train_test_split(test_size=test_size, shuffle=False)
    if just_train:
        return ds["train"]
    else:
        return ds
    
    
def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


def extract_anthropic_prompt(prompt_and_response):
    """Extract the anthropic prompt from a prompt and response pair."""
    search_term = "\n\nAssistant:"
    search_term_idx = prompt_and_response.rfind(search_term)
    assert search_term_idx != -1, f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[: search_term_idx + len(search_term)]


def get_hh(split: str, sanity_check: bool = False, silent: bool = False, cache_dir: Optional[str] = None) -> Dataset:
    """Load the Anthropic Helpful-Harmless dataset from Hugging Face and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt': List[str],
        'chosen': List[str],
        'rejected': List[str],
    }

    Prompts should be structured as follows:
      \n\nHuman: <prompt>\n\nAssistant:
    Multiple turns are allowed, but the prompt should always start with \n\nHuman: and end with \n\nAssistant:.
    """
    dataset = load_dataset("Anthropic/hh-rlhf", split=split, cache_dir=cache_dir)
    if sanity_check:
        dataset = dataset.select(range(min(len(dataset), 1000)))

    def split_prompt_and_responses(sample) -> Dict[str, str]:
        prompt = extract_anthropic_prompt(sample["chosen"])
        return {
            "prompt": prompt,
            "chosen": sample["chosen"][len(prompt) :],
            "rejected": sample["rejected"][len(prompt) :],
        }

    return dataset.map(split_prompt_and_responses)


def create_run_string():
    # Get current date and time
    current_datetime = datetime.now()
    # Format the datetime to a string
    datetime_str = current_datetime.strftime("%Y-%m-%d_%H:%M:%S")
    # Create and return the string with "run" appended with the current date and time
    return f"run_{datetime_str}"


@contextmanager
def time_block(label):
    start = time.time()
    try:
        yield
    finally:
        end = time.time()
        print(f"{label}: {end - start} seconds")
    

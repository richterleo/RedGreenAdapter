import configparser
import torch
import typing

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


def parse_config(config_path: Union[str, Path], use_wandb):
    '''
    Parses (most parts of) config file and saves down as attributes.

    Args:
        config_path: path to config file.

    Returns: 
        root path (of dataset directory), whether files should be overwritten during preprocessing, path to yaml file
    '''
    
    if not isinstance(config_path, str):
        config_path = config_path.as_posix()
            
    model_args = read_config(config_path, "models")
    logging_args = read_config(config_path, "logging")
    store_args = read_config(config_path, "directories")
    
        

                
    return model_args, logging_args, store_args

def parse_wandb_args(logging_args):
    
    pass
    
    

def read_config(config_file: Union[str, Path], section: str) -> Dict:
    '''
    Reads configuration dict of a section.

    Args:
        config_file: path to config file
        section: section to read

    Returns:
        configuration dict
        
    '''
    config = configparser.ConfigParser()
    config.read(config_file)
    
    return config[section]


if __name__ == "__main__":
    
    config_path = "config.ini"
    if not Path(config_path).exists():
        print("Path does not exist")

    model_args, logging_args, store_args = parse_config(config_path)
    
    print(logging_args)
    

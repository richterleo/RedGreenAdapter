import os
import torch
import json
import time
import logging
import random
import numpy as np
from typing import List
from datetime import datetime
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import get_linear_schedule_with_warmup

from data_pool import DataPool

from transformers import GPT2Tokenizer

# additional imports by me
from transformers import GPT2Model
from datasets import load_dataset


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
log = logging.getLogger(__name__)


class PromptDataset(Dataset):
    def __init__(self, ds_name, tokenizer, split='train'):

        data = load_dataset(ds_name, split=split)
        self.items = [(s["prompt"]["text"], s["continuation"]["text"]) for s in data]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        prompt = self.items[idx][0]
        return prompt



class PromptCollator(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, sequences):
        concepts = [sequence['order'] for sequence in sequences]
        prompts = [sequence['prompt'] for sequence in sequences]
        constraints = [sequence['constraint'] for sequence in sequences]

        encodings_dict = self.tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = encodings_dict['input_ids']
        attention_mask = encodings_dict['attention_mask']

        return input_ids, attention_mask, concepts, constraints


class SequenceDataset(Dataset):
    def __init__(self, data_pool: DataPool):
        self.queries, self.responses, self.cat_tokens = data_pool.get_data()

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        return {'query': self.queries[idx],
                'response': self.responses[idx],
                'cat_tokens': self.cat_tokens[idx]
                }


class SequenceCollator(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, sequences):
        queries = [sequence['query'] for sequence in sequences]
        responses = [sequence['response'] + self.tokenizer.eos_token for sequence in sequences]
        cat_ids = [self.tokenizer.convert_tokens_to_ids(sequence['cat_tokens']) for sequence in sequences]

        query_encodings_dict = self.tokenizer(queries, return_tensors="pt", padding=True)
        query_input_ids = query_encodings_dict['input_ids']
        query_mask = query_encodings_dict['attention_mask']

        query_input_ids = torch.cat([query_input_ids.new(cat_ids)[:, None], query_input_ids], dim=1)
        query_mask = torch.cat([query_mask.new([1] * len(query_mask))[:, None], query_mask], dim=1)

        response_encodings_dict = self.tokenizer(responses, return_tensors="pt", padding=True)
        response_input_ids = response_encodings_dict['input_ids']
        response_mask = response_encodings_dict['attention_mask']

        return query_input_ids, query_mask, response_input_ids, response_mask
    
    
    
if __name__ == "__main__":
    base_model_name = "gpt2-large"
    ds_name = "allenai/real-toxicity-prompts"
    tokenizer = GPT2Tokenizer.from_pretrained(base_model_name, pad_token="<|endoftext|>")
    prompt_ds = PromptDataset(ds_name, tokenizer)
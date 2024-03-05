from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from datasets import load_dataset
from torch.optim import Adam
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    ValueHead,
    AutoTokenizer,
    HfArgumentParser,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    GenerationConfig
)

from transformers.generation.logits_process import LogitsProcessorList

from trl import (AutoModelForCausalLMWithValueHead, 
                 PPOConfig, 
                 PPOTrainer, 
                 create_reference_model, 
                 set_seed, 
                 PreTrainedModelWrapper)

from trl.core import LengthSampler

# my stuff
from utils import build_dataset, collator
from PPO_adapter import PPOwithAdapterConfig, PPOwithAdapterTrainer
from product_of_experts import (BaseModelSumLogitsProcessor, 
                    AdapterModelLogitsProcessor, 
                    logits_processor_wrapper, 
                    update_get_logits_warper)



class ProductAutoModelForCausalLMWithValueHead(AutoModelForCausalLMWithValueHead):
    
    def __init__(self,
                 adapter_model,
                 basis_model,
                 **kwargs):
        
        super().__init__(adapter_model, **kwargs)
        
        self.basis_model = basis_model
        v_head_kwargs, _, _ = self._split_kwargs(kwargs)
        
        if not any(hasattr(self.basis_model, attribute) for attribute in self.lm_head_namings):
            raise ValueError("The model does not have a language model head, please use a model that has one.")
        
        self.v_head = self._get_head_for_concatenated_models(v_head_kwargs)
        
        self._init_weights(**v_head_kwargs)
        
    
    def _get_head_for_concatenated_models(self, v_head_kwargs):
        
        return ProductValueHead(self.adapter_model.config, self.basis_model.config, **v_head_kwargs)
    
        
        
    
class ProductValueHead(ValueHead):
    
    def __init__(self,
                 adapter_config,
                 basis_config,
                 **kwargs):
        
    
        super().__init__(adapter_config, **kwargs)
        
        # get size of hidden layer for basis_model to concatenate
        if hasattr(basis_config, "hidden_size"):
            hidden_size = basis_config.hidden_size
        if hasattr(basis_config, "word_embed_proj_dim"):
            hidden_size = basis_config.word_embed_proj_dim
        elif hasattr(basis_config, "is_encoder_decoder"):
            if basis_config.is_encoder_decoder and hasattr(basis_config, "decoder"):
                if hasattr(basis_config.decoder, "hidden_size"):
                    hidden_size = basis_config.decoder.hidden_size

       
        self.summary = nn.Linear(hidden_size + self.summary.in_features, 1)

        self.flatten = nn.Flatten()
        
        
    def forward(self, hidden_states):
        
        output = self.dropout(hidden_states)
        
        if output.dtype != self.summary.weight.dtype:
            output = output.to(self.summary.weight.dtype)
            
        
        try:
            output = self.summary(output)
        
        except RuntimeError:
            # TODO: make this less hacky
            # Currently, we're just concatenating this with a zeros tensor, so it doesn't throw an error if the input is too small
            zeros_tensor = torch.zeros(output.shape[0], self.summary.in_features-output.shape[1])
            concatenated_output = torch.cat((output, zeros_tensor), dim=1)
            output = self.summary(concatenated_output)
            
        return output
        
    
    
    
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from transformers import PreTrainedModel, PretrainedConfig


from trl import AutoModelForCausalLMWithValueHead


class ProductAutoModelForCausalLMWithValueHead(AutoModelForCausalLMWithValueHead):
    '''
    
    '''
    
    supported_args = (
    "basis_model",
    "summary_dropout_prob",
    "v_head_initializer_range",
    "v_head_init_strategy",
    )
    
    def __init__(self,
                 adapter_model, 
                 basis_model: Optional[AutoModelForCausalLMWithValueHead]=None,
                 **kwargs):
        
        super().__init__(adapter_model, **kwargs)
        
        if basis_model:
            self.basis_model = basis_model
            
        # TODO add default basis_model 
        
        v_head_kwargs, _, _ = self._split_kwargs(kwargs)
        
        if not any(hasattr(self.basis_model.pretrained_model, attribute) for attribute in self.lm_head_namings):
            raise ValueError("The model does not have a language model head, please use a model that has one.")
        
        self.v_head = self._get_head_for_concatenated_models(v_head_kwargs)
        
        self._init_weights(**v_head_kwargs)
        
    
    def _get_head_for_concatenated_models(self, v_head_kwargs):
        
        return ProductValueHead(self.pretrained_model.config, self.basis_model.pretrained_model.config, **v_head_kwargs)
    
        
        
    
class ProductValueHead(nn.Module):
    
    def __init__(self,
                 adapter_config,
                 basis_config,
                 **kwargs):
        
    
        super().__init__()
        
        if not hasattr(adapter_config, "summary_dropout_prob"):
            summary_dropout_prob = kwargs.pop("summary_dropout_prob", 0.1)
        else:
            summary_dropout_prob = adapter_config.summary_dropout_prob     
        
        self.dropout = nn.Dropout(summary_dropout_prob) if summary_dropout_prob else nn.Identity()
        
        # get size of hidden layer for basis_model to concatenate
        adapter_hidden_size = self._get_size_of_hidden_layer(adapter_config)
        basis_hidden_size = self._get_size_of_hidden_layer(basis_config)
 
        self.summary = nn.Linear(adapter_hidden_size + basis_hidden_size, 1)

        self.flatten = nn.Flatten()
        
    
    def _get_size_of_hidden_layer(self, config):
        
        if hasattr(config, "hidden_size"):
            hidden_size = config.hidden_size
        if hasattr(config, "word_embed_proj_dim"):
            hidden_size = config.word_embed_proj_dim
        elif hasattr(config, "is_encoder_decoder"):
            if config.is_encoder_decoder and hasattr(config, "decoder"):
                if hasattr(config.decoder, "hidden_size"):
                    hidden_size = config.decoder.hidden_size
                    
        return hidden_size
        
        
    def forward(self, hidden_states):
        
        output = self.dropout(hidden_states)
        
        if output.dtype != self.summary.weight.dtype:
            output = output.to(self.summary.weight.dtype)
            
        
        try:
            output = self.summary(output)
        
        except RuntimeError:
            # TODO: make this less hacky
            # Currently, we're just outputting a zeros tensor if the dimensions don't match (i.e. only output of adapter is fead to valuehead)
            
            output = torch.zeros(output.shape[0], output.shape[1], self.summary.in_features).to(output.device)
            
        return output
        
    

class ProductModel(PreTrainedModel):
    
    
    def __init__(self, config:PretrainedConfig, *inputs, **kwargs):
        
        super().__init__()
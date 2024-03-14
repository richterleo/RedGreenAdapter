import torch
import typing

from copy import deepcopy
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PretrainedConfig,
    AutoModel, 
    AutoConfig,
    LogitsProcessor,
    TopPLogitsWarper,
    LogitNormalization,
    GenerationConfig
)

from transformers.generation.logits_process import LogitsProcessorList

from torch.nn.functional import softmax

from trl import (
    AutoModelForCausalLMWithValueHead, 
    create_reference_model,
    PreTrainedModelWrapper
)



class BaseModelSumLogitsProcessor(LogitsProcessor):
    
    def __init__(self, 
                 source_model_name, 
                 *args, 
                 **kwargs):
        
        super().__init__(*args, **kwargs)
        self.source_model_name = source_model_name
        self.source_model = AutoModelForCausalLM.from_pretrained(source_model_name)
        self.device = None
        
        
    def _calc_bmodel_next_token_logits(self, 
                          input_ids,
                          scores):
        
        if not self.device:
            self.device = input_ids.device
            self.source_model.to(self.device)
        
        self.source_model.eval() # TODO: is this necessary?
        
        with torch.inference_mode():                           
            outputs = self.source_model(input_ids=input_ids, return_dict=True) # outputs (lm_logits, loss, value) 
            bmodel_next_token_logits = outputs.logits[:, -1, :] # get output for last/next token
                                                #attention_mask=attention_mask)[0]      
        
        return bmodel_next_token_logits
                 
        
    
    def __call__(self, 
                 input_ids, 
                 scores):

        
        # if self.rng is None:
        #     self.rng = torch.Generator(device=input_ids.device)
        
        return self._calc_bmodel_next_token_logits(input_ids, scores) + scores
        

class ModelLogitsProcessor(LogitsProcessor):
    
    def __init__(self,
                 model,
                 *args,
                 generation_config=None,
                 base=False,
                 **kwargs):
        
        super().__init__(*args, **kwargs)
        self.model = model
        
        if generation_config:
            self.generation_config = generation_config
        else:
            self.generation_config = self._get_default_generation_config(base)
            
        self.device = None
            
    def _get_default_generation_config(self, base):
        
        if base:
            return GenerationConfig(
                min_length=-1,
                top_k=0.0,
                top_p=1.0,
                do_sample=True,
                pad_token_id= tokenizer.eos_token_id,
                renormalize_logits=True)
            
        else:
            return GenerationConfig(
                min_length=-1,
                top_k=0.0,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                renormalize_logits=True)
            
    def _calc_next_token_logits(self,
                         input_ids,
                         scores):
        
        if not self.device:
            self.device = input_ids.device
            self.model.to(self.device)
        
        self.model.eval() # TODO: is this necessary?
        
        with torch.inference_mode():                           
            outputs = self.model.generate(input_ids=input_ids, 
                                               generation_config=self.generation_config,
                                               max_new_tokens=1, # just generate the next token
                                               return_dict=True) # outputs (lm_logits, loss, value) 
            
            next_token_logits = outputs[:, -1, :] 
                
        assert next_token_logits.shape == scores.shape, "Truncated adapter model logits must have same shape as base model logits."
                
        return next_token_logits
                 
        
    
    def __call__(self, 
                 input_ids, 
                 scores):

        
        return self._calc_next_token_logits(input_ids, scores) + scores   
            
        



class BaseModelLogitsProcessor(LogitsProcessor):
    
    def __init__(self, 
                 base_model,
                 *args,
                 generation_config=None,
                 **kwargs):
        
        super().__init__(*args, **kwargs)
        self.base_model = base_model
        
        if generation_config:
            self.generation_config = generation_config
        else:
            self.generation_config = self._get_default_generation_config()
            
    def _get_default_generation_config(self):
        
        return GenerationConfig(
            min_length=-1,
            top_k=0.0,
            top_p=1.0,
            do_sample=True,
            pad_token_id= tokenizer.eos_token_id)
            
        
    def _calc_next_token_logits(self, 
                          input_ids,
                          scores):
        
        if not self.device:
            self.device = input_ids.device
            self.base_model.to(self.device)
        
        self.base_model.eval() # TODO: is this necessary?
        
        with torch.inference_mode():                           
            outputs = self.base_model.generate(input_ids=input_ids, 
                                               generation_config=self.generation_config,
                                               max_new_tokens=1, # just generate the next token
                                               return_dict=True)[0] # outputs (lm_logits, loss, value) 
            
            next_token_logits = outputs[:, -1, :] 
                
        assert next_token_logits.shape == scores.shape, "Truncated adapter model logits must have same shape as base model logits."
                
        return next_token_logits
                 
        
    
    def __call__(self, 
                 input_ids, 
                 scores):

        
        return self._calc_next_token_logits(input_ids, scores) + scores          
            
class AdapterModelLogitsProcessor(LogitsProcessor):
    
    def __init__(self, 
                 adapter_model, 
                 *args,
                 generation_config=None,
                 post_normalize=True,
                 top_p = 0.9, 
                 **kwargs):
        
        super().__init__(*args, **kwargs)
        self.adapter_model = adapter_model
        self.post_normalize = post_normalize
        self.top_p = top_p
        self.device = None
        
        self.top_p_logits_warper = TopPLogitsWarper(self.top_p)
        
        # if post_normalize, we normalize the probability distribution after truncating
        if self.post_normalize:
            self.logit_normalizer = LogitNormalization()
            self.logits_processor_lst = LogitsProcessorList([self.top_p_logits_warper, self.logit_normalizer])
        else:
            self.logits_processor_lst = LogitsProcessorList([self.top_p_logits_warper])
        
        
    def _calc_next_token_logits(self, 
                          input_ids,
                          scores):
        
        if not self.device:
            self.device = input_ids.device
            self.adapter_model.to(self.device)
        
        self.adapter_model.eval() # TODO: is this necessary?
        
        with torch.inference_mode():                           
            outputs = self.adapter_model(input_ids=input_ids, return_dict=True)[0] # outputs (lm_logits, loss, value) 
            next_token_logits = outputs[:, -1, :] 
            trunc_logits = self.logits_processor_lst(input_ids, next_token_logits)
                
        assert trunc_logits.shape == scores.shape, "Truncated adapter model logits must have same shape as base model logits."
                
        return trunc_logits
                 
        
    
    def __call__(self, 
                 input_ids, 
                 scores):

            
        return self._calc_next_token_logits(input_ids, scores) + scores          
    

def logits_processor_wrapper(adapter_model, *args, top_p = 0.9, logits_warper=None, **kwargs):
    '''
    
    '''
    if logits_warper:
        return LogitsProcessorList([logits_warper, AdapterModelLogitsProcessor(adapter_model, *args, top_p=top_p, **kwargs)])
    
    else: 
        return LogitsProcessorList([AdapterModelLogitsProcessor(adapter_model, *args, top_p=top_p, **kwargs)])
    
    
def get_logits_warper(model, generation_config, *args, adapter_model_top_p = 0.9, **kwargs):
    
    logits_warper_lst = model._get_logits_warper(generation_config)
    adapter_model_logits_warper = AdapterModelLogitsProcessor(adapter_model, *args, top_p=adapter_model_top_p, **kwargs)
    logit_normalizer = LogitNormalization()
    
    logits_warper_lst.extend([adapter_model_logits_warper, logit_normalizer])
    
    return 
    

def update_get_logits_warper(original_method, model, model_generation_config=None, base=False):
    '''
    Monkey patching the _get_logits_warper method of the base model
    '''
    
    def wrapper(self, generation_config):
        
        # this means the basis model logits are already normalized
        logits_warper_lst = original_method(generation_config)
        
        # sum_logits_warper = ModelLogitsProcessor(model,
        #                                             generation_config=generation_config,
        #                                         base=base)
        
        if not base:
            sum_logits_warper = AdapterModelLogitsProcessor(model,
                                                            generation_config=model_generation_config)
        logit_normalizer = LogitNormalization()
        
        logits_warper_lst.extend([sum_logits_warper, logit_normalizer])
        
        return logits_warper_lst
    
    return wrapper
    
    


if __name__ == "__main__":
    
    base_model_name = "gpt2-large"
    adapter_model_name = "gpt2"
    
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(torch_device)
    
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    inputs = tokenizer('I enjoy walking with my cute dog', return_tensors='pt').to(torch_device)
    
    adapter_model = AutoModelForCausalLM.from_pretrained(adapter_model_name).to(torch_device)
    # # trl.models.modeling_value_head.AutoModelForCausalLMWithValueHead
    # # adapter_model.pretrained_model gives GPT2LMHeadModel
    adapter_model = AutoModelForCausalLMWithValueHead.from_pretrained(adapter_model_name).to(torch_device)
    adapter_model_top_p = 0.9
    base_model = AutoModelForCausalLM.from_pretrained(adapter_model_name).to(torch_device)
        
    # -------------------------------------------
    
    
    # # when using with non-sampling based generation strategies, the adaptermodellogitsprocessor must be a processor
    # # when using with (multinomial) sampling based generation strategies, the adaptermodellogitsprocessor must be a warper
    # # make sure there is renormalization occuring after
    
    # # Define generation kwargs
    # generation_kwargs = {
    # "min_length": -1,
    # "top_k": 0.0,
    # "top_p": 1.0,
    # "do_sample": True,
    # "pad_token_id": tokenizer.eos_token_id,
    # }
    
    # logits_processor_lst = logits_processor_wrapper(adapter_model)
    
    mode = 'greedy'
    
    
    # VARIANT 1: GREEDY SEARCH 
    # We don't use greedy search (we do sampling instead) but if we wanted to, we would have to switch adapter_model and base_model
    if mode == 'greedy':
        pass
                                      
    
    # VARIANT 2: SAMPLING 
    
    # every model has own configuration file for default generate args; GPT2 inherits also from PretrainedConfig
    # defaults include: do_sample=False, top_p=1, top_k=50, temperature=1.0
    
    # TODO: annoyingly, when sampling, the logitsprocessor will come first, before a logitswarper. If we do truncation
    # like with the top-k / top-p sampling, we'd want to truncate _both_ prob distributions first and then multiply + normalize
    # easy fix for now: call sample method and give list of logit_warpers. but this is non-ideal bc we just want to be able to use any configuration
    
    elif mode == 'sample':
        
        # # base_model first 
        # original_warp_creator = base_model._get_logits_warper
        # updated_get_logits_warper = update_get_logits_warper(original_warp_creator, adapter_model, adapter_model_top_p)
        # base_model._get_logits_warper = updated_get_logits_warper.__get__(base_model, AutoModelForCausalLM)
        # generation_output = base_model.generate(**inputs, 
        #                                         renormalize_logits=True,
        #                                         max_new_tokens=30,
        #                                         do_sample=True)
        
        # # adapter_model first
        original_warp_creator = adapter_model._get_logits_warper
        updated_get_logits_warper = update_get_logits_warper(original_warp_creator, adapter_model, adapter_model_top_p)
        base_model._get_logits_warper = updated_get_logits_warper.__get__(base_model, AutoModelForCausalLM)
        generation_output = base_model.generate(**inputs, 
                                                renormalize_logits=True,
                                                max_new_tokens=30,
                                                do_sample=True)
        
    # VARIANT 3: BEAMSEARCH
    elif mode == 'beamsearch':
        pass
    
    
    print(tokenizer.decode(generation_output[0], skip_special_tokens=True))
    
    
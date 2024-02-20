import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    RobertaForSequenceClassification,
    RobertaTokenizer,
)

from torch.nn.functional import softmax

from trl import AutoModelForCausalLMWithValueHead, create_reference_model


class ProductModel(torch.nn.Module):
    
    def __init__(self, base_model_name, adapter_model_name):
        
        super(ProductModel, self).__init__()
        # Load base model 
        # in bfloat16 to save memory using `transformers`.
        # Problem: for gpt2 based models bfloat16 is not yet supported
        # adapter_model = AutoModelForCausalLM.from_pretrained(config.adapter_model_name, torch_dtype=torch.bfloat16)
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        self.base_model = AutoModelForCausalLMWithValueHead.from_pretrained(self.base_model)
        
        # load adapter model
        self.adapter_model = AutoModelForCausalLM.from_pretrained(adapter_model_name)
        self.adapter_model = AutoModelForCausalLMWithValueHead.from_pretrained(self.adapter_model)
        
        # Base model should stay fixed
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # # Ensure both models are in evaluation mode
        # self.gpt2_large.eval()
        # self.gpt2_xl.eval()

    def forward(self, input_ids, attention_mask=None):
        
        # Get logits from both models
        base_logits = self.base_model(input_ids=input_ids, attention_mask=attention_mask).logits
        adapter_logits = self.adapter_model(input_ids=input_ids, attention_mask=attention_mask).logits

        # Element-wise multiplication of the logits
        combined_logits = base_logits * adapter_logits

        # Apply softmax to get probabilities
        combined_probs = softmax(combined_logits, dim=-1)

        return combined_probs
    
    def __getattr__(self, name):
        '''
        Want this to behave like a pretrained_model
        '''
        return getattr(self.adapter_model, name)
        
    
    
def create_reference_model_from_product(product_model, num_shared_layers):
    
    copied_model = product_model.deepcopy(product_model)
    copied_model.adapter_model = create_reference_model(copied_model.adapter_model, num_shared_layers)
    
    return copied_model
import torch

from copy import deepcopy
from transformers import (
    AutoModelForCausalLM,
    PreTrainedModel,
    PretrainedConfig,
    AutoModel, 
    AutoConfig
)

from torch.nn.functional import softmax

from trl import (
    AutoModelForCausalLMWithValueHead, 
    create_reference_model,
    PreTrainedModelWrapper
)


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
    

    def generate(self, input_ids, max_length=50):
        self.eval() # Put the model in eval mode
        generated = input_ids # Initialize generated sequence with input

        with torch.no_grad():
            for _ in range(max_length):
                # Predict the next token using the combined model
                combined_probs = self.forward(input_ids=generated)
                
                # Select the last token from the sequence
                next_token_probs = combined_probs[:, -1, :]
                
                # Sample or choose the next token (here we choose the most probable token for simplicity)
                next_token_id = torch.argmax(next_token_probs, dim=-1, keepdim=True)
                
                # Append the predicted token to the generated sequence
                generated = torch.cat((generated, next_token_id), dim=1)
                
                # Check if the last predicted token is the end of sentence token for GPT-2 (50256 for GPT-2 tokenizer)
                if next_token_id.item() == 50256:
                    break

        return generated

    
    @property
    def __class__(self):
        return self.adapter_model.__class__
    
    
    
class PEConfig(PretrainedConfig):

    model_type = 'pe_model'
    def __init__(self, base_model_name, adapter_model_name, **kwargs):
        super().__init__(**kwargs)
        self.base_model_name = base_model_name
        self.adapter_model_name = adapter_model_name

class PEModel(PreTrainedModelWrapper):
    
    #config_class = PEConfig
    def __init__(self, config):
        super().__init__(**config)
        self.config = config
        self.base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name)
        self.base_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.base_model_name)
        
        # Base model should stay fixed
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # load adapter model
        self.adapter_model = AutoModelForCausalLM.from_pretrained(config.adapter_model_name)
        self.adapter_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.adapter_model_name)
               
        
            
    def forward(self, input_ids, attention_mask=None):
        
        base_logits = self.base_model(input_ids=input_ids, attention_mask=attention_mask).logits
        adapter_logits = self.adapter_model(input_ids=input_ids, attention_mask=attention_mask).logits

        # Element-wise multiplication of the logits
        combined_logits = base_logits * adapter_logits

        # Apply softmax to get probabilities
        combined_probs = softmax(combined_logits, dim=-1)

        return combined_probs
    
    # @property
    # def __class__(self):
    #     return self.adapter_model.__class__


        
    
    
def create_reference_model_from_product(pe_model, num_shared_layers):
    
    copied_model = deepcopy(pe_model)
    copied_model.adapter_model = create_reference_model(copied_model.adapter_model, num_shared_layers)
    
    return copied_model
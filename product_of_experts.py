import torch

from copy import deepcopy
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
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
    def __init__(self, b_model_name, adapter_model_name, **kwargs):
        super().__init__(**kwargs)
        self.b_model_name = b_model_name
        self.adapter_model_name = adapter_model_name

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = kwargs.get('device', device)

class PEModel(PreTrainedModel):
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.b_model = AutoModelForCausalLM.from_pretrained(config.b_model_name).to(config.device)
        self.b_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.b_model_name).to(config.device)
        
        # Base model should stay fixed
        for param in self.b_model.parameters():
            param.requires_grad = False
        
        # load adapter model
        self.adapter_model = AutoModelForCausalLM.from_pretrained(config.adapter_model_name).to(config.device)
        self.adapter_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.adapter_model_name).to(config.device)
               
        
            
    def forward(self, input_ids, attention_mask=None, **kwargs):
        
        base_logits = self.b_model(input_ids=input_ids, 
                                   attention_mask=attention_mask)[0] # outputs (lm_logits, loss, value)
        adapter_logits = self.adapter_model(input_ids=input_ids, 
                                            attention_mask=attention_mask)[0]

        # Element-wise multiplication of the logits
        combined_logits = base_logits * adapter_logits

        # Apply softmax to get probabilities
        combined_probs = softmax(combined_logits, dim=-1)

        return combined_probs
    
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        
        token_type_ids = kwargs.get("token_type_ids", None)
        # Omit tokens covered by past_key_values
        if past_key_values:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -input_ids.shape[1] :]

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]
        else:
            position_ids = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "position_ids": position_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            }
        )

        return model_inputs
    
    # @property
    # def __class__(self):
    #     return self.adapter_model.__class__


        
    
    
def create_reference_model_from_product(pe_model, num_shared_layers):
    
    copied_model = deepcopy(pe_model)
    copied_model.adapter_model = create_reference_model(copied_model.adapter_model, num_shared_layers)
    
    return copied_model


if __name__ == "__main__":
    
    base_model_name = "gpt2-large"
    adapter_model_name = "gpt2"
    
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"
    print(torch_device)
    
    pe_config = PEConfig(base_model_name, adapter_model_name)
    pe_model = PEModel(pe_config)
    
    tokenizer = AutoTokenizer.from_pretrained(adapter_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model_inputs = tokenizer('I enjoy walking with my cute dog', return_tensors='pt').to(torch_device)

    # generate 40 new tokens
    greedy_output = pe_model.generate(**model_inputs, max_new_tokens=40)
    
    print(greedy_output)
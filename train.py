# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import wandb

from copy import deepcopy
from dataclasses import dataclass, field
from torch.optim import Adam
from tqdm import tqdm
from typing import Optional

from transformers import (
    AutoModelForCausalLM,
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
                 set_seed)

from trl.core import LengthSampler

# my stuff
from utils import build_dataset, collator
from PPO_adapter import PPOTrainerForProducts
from product_of_experts import (BaseModelSumLogitsProcessor, 
                    AdapterModelLogitsProcessor, 
                    logits_processor_wrapper, 
                    update_get_logits_warper)


tqdm.pandas()


@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine-tune with PPO
    """

    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable.
    base_model_name: Optional[str] = field(default="openai-community/gpt2-large", metadata={"help": "the base model name"})
    adapter_model_name: Optional[str] = field(default="openai-community/gpt2", metadata={"help": "the base model name"})
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=(1.47e-5) * 2, metadata={"help": "the learning rate"})
    mini_batch_size: Optional[int] = field(default=4, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=16, metadata={"help": "the batch size"})
    adapter_model_top_p: Optional[float] = field(default=0.9, metadata={"help": "the top-p value for thresholded adapter"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    model_save_path: Optional[str] = field(
        default="./Test",
        metadata={"help": "the path to save the model"},
    )


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]

ppo_config = PPOConfig(
    model_name=script_args.adapter_model_name,
    learning_rate=script_args.learning_rate,
    log_with='wandb', #script_args.log_with,
    ppo_epochs=3, # Originally 100
    mini_batch_size=script_args.mini_batch_size,
    batch_size=script_args.batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    remove_unused_columns=False
)

ppo_sample_config = deepcopy(ppo_config)
ppo_sample_config.model_name = script_args.base_model_name


# set seed before initializing value head for deterministic eval
set_seed(ppo_config.seed)

model = AutoModelForCausalLM.from_pretrained(ppo_config.model_name) # torch_dtype=torch.bfloat16 not available for gpt2
model = AutoModelForCausalLMWithValueHead.from_pretrained(model)

# This serves as reference model as well
# Hope there are no problems with 
base_model = AutoModelForCausalLM.from_pretrained(script_args.base_model_name) # torch_dtype=torch.bfloat16 not available for gpt2
base_model = AutoModelForCausalLMWithValueHead.from_pretrained(base_model)
ref_model = deepcopy(base_model)

# GPT-2 / GPT-J tokenizer has a pad token, but it is not eos_token by default. We need to set it to eos_token.
# only for this model.
# need to take tokenizer from smaller model #TODO ask if this might be a problem/ there's some better solution
tokenizer = AutoTokenizer.from_pretrained(script_args.base_model_name)
tokenizer.pad_token = tokenizer.eos_token

# We make sure to use `Adam` optimizer on the model parameters that require gradients.
optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=ppo_config.learning_rate)

# We retrieve the dataloader by calling the `build_dataset` function.
min_input_length = 30
max_input_length = 40
dataset = build_dataset(ppo_config, 
                        input_min_text_length=min_input_length, 
                        input_max_text_length=max_input_length,
                        tokenizer=tokenizer)


# Create PPOTrainer
ppo_trainer_for_products = PPOTrainerForProducts(
    ppo_config,
    model=model,
    source_model=base_model,
    tokenizer=tokenizer,
    ref_model=ref_model,
    dataset=dataset,
    data_collator=collator,
    optimizer=optimizer
)

# Create a second PPOTrainer that we use solely for generating samples during the rollout phase of PPO
sample_ppo_trainer = PPOTrainer(
    ppo_sample_config,
    model=base_model,
    tokenizer=tokenizer,
    ref_model=ref_model, 
    dataset=dataset,
    data_collator=collator,
    optimizer=optimizer,
)

# We then build the reward pipeline, we will use the toxicity model to compute the reward.
# We first load the toxicity model and tokenizer.
toxicity_model_id = "facebook/roberta-hate-speech-dynabench-r4-target"
toxicity_tokenizer = RobertaTokenizer.from_pretrained(toxicity_model_id)
# We load the toxicity model in fp16 to save memory.
toxicity_model = RobertaForSequenceClassification.from_pretrained(toxicity_model_id, torch_dtype=torch.float16).to(
    ppo_trainer_for_products.accelerator.device
)


# We then define the arguments to pass to the `generate` function. These arguments
# are passed to the `generate` function of the PPOTrainer, which is a wrapper around
# the `generate` function of the trained model.
generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 0.95,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "renormalize_logits": True}

output_min_length = 20
output_max_length = 30
output_length_sampler = LengthSampler(output_min_length, output_max_length)

# model_save_path = script_args.model_save_path

for epoch, batch in tqdm(enumerate(ppo_trainer_for_products.dataloader)):
    
    query_tensors = batch["input_ids"]

    empty_response_counter = 0
    
    # Get response from the policy model
    response_tensors = []
    ref_response_tensors = []
    for query in query_tensors:
        gen_len = output_length_sampler()
        generation_kwargs["max_new_tokens"] = gen_len # I'm not sure what's behind this?
        
        # for generation, we don't keep gradients, so we don't necessarily need the ppo_trainer 
        if generation_kwargs['do_sample']:
            original_warp_creator = base_model.pretrained_model._get_logits_warper
            updated_get_logits_warper = update_get_logits_warper(original_warp_creator, model)
            base_model.pretrained_model._get_logits_warper = updated_get_logits_warper.__get__(base_model.pretrained_model, base_model.pretrained_model.__class__)
            
            # Need to save ref_response for KL divergence 
            response = sample_ppo_trainer.generate(query, **generation_kwargs)
            
            # validate response tensors
            if response is None:
                empty_response_counter += 1
                print(f"Empty response {empty_response_counter}")
                
            
            response_tensors.append(response.squeeze()[-gen_len:]) # The prompt is saved separately            
            
        
    batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]
    batch["ref_response"] = [tokenizer.decode(r.squeeze()) for r in ref_response_tensors]
    
    
    # Compute sentiment score 
    texts = batch["response"]
    toxicity_inputs = toxicity_tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(
        ppo_trainer_for_products.accelerator.device
    )
    logits = toxicity_model(**toxicity_inputs).logits.float()
    toxicity_labels = (logits[:, 0]).tolist()

    rewards = [torch.tensor(output) for output in toxicity_labels]

    # Run PPO step
    stats = ppo_trainer_for_products.step(query_tensors, response_tensors, rewards)
    ppo_trainer_for_products.log_stats(stats, batch, rewards)

    # Save model every 100 epochs
    if epoch % 1 == 0:
        if ppo_trainer_for_products.accelerator.is_main_process:
            ppo_trainer_for_products.save_pretrained(script_args.model_save_path)


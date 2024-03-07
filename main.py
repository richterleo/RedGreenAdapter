import argparse
import configparser
import torch
import wandb

from copy import deepcopy
from torch.optim import Adam
from tqdm import tqdm
from typing import Optional

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    RobertaForSequenceClassification,
    RobertaTokenizer
)

from trl import (AutoModelForCausalLMWithValueHead, 
                 PPOConfig, 
                 PPOTrainer, 
                 set_seed)

from trl.core import LengthSampler

# my components
from arguments import PPOArgs, DPOArgs
from utils import build_dataset, collator
from PPO_adapter import PPOTrainerForProducts
from product_of_experts import (BaseModelSumLogitsProcessor, 
                    AdapterModelLogitsProcessor, 
                    logits_processor_wrapper, 
                    update_get_logits_warper)
from value_head import ProductValueHead, ProductAutoModelForCausalLMWithValueHead


tqdm.pandas()

def train_ppo(config):

    parser = HfArgumentParser(PPOArgs)
    script_args = parser.parse_args_into_dataclasses()[0]

    ppo_config = PPOConfig(
        model_name=config['models']['adapter_model_name'],
        learning_rate=script_args.learning_rate,
        log_with= 'wandb' if config['logs']['use_wandb'] else None, #script_args.log_with,
        ppo_epochs=script_args.ppo_epochs, # Originally 100
        mini_batch_size=script_args.mini_batch_size,
        batch_size=script_args.batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        remove_unused_columns=False,
        tracker_project_name= config['logs']['wandb_project_name']
        # tracker_kwargs= {'wandb': {"name": config['logs']['wandb_project_name']}} #TODO: Write fct that gives new exp names
    )

    ppo_sample_config = deepcopy(ppo_config)
    ppo_sample_config.model_name = config['models']['basis_model_name']


    # set seed before initializing value head for deterministic eval
    set_seed(ppo_config.seed)

    # This serves as reference model as well
    basis_model = AutoModelForCausalLM.from_pretrained(config['models']['basis_model_name']) # torch_dtype=torch.bfloat16 not available for gpt2
    basis_model = AutoModelForCausalLMWithValueHead.from_pretrained(basis_model)
    ref_model = deepcopy(basis_model)


    # define the adapter model
    model = AutoModelForCausalLM.from_pretrained(ppo_config.model_name) # torch_dtype=torch.bfloat16 not available for gpt2
    model = ProductAutoModelForCausalLMWithValueHead.from_pretrained(model, basis_model=basis_model) # my class, has changed value head


    # GPT-2 / GPT-J tokenizer has a pad token, but it is not eos_token by default. We need to set it to eos_token.
    # only for this model.
    # need to take tokenizer from smaller model #TODO ask if this might be a problem/ there's some better solution
    tokenizer = AutoTokenizer.from_pretrained(config['models']['basis_model_name'])
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
        basis_model=basis_model,
        tokenizer=tokenizer,
        ref_model=ref_model,
        dataset=dataset,
        data_collator=collator,
        optimizer=optimizer
    )

    # Create a second PPOTrainer that we use solely for generating samples during the rollout phase of PPO
    sample_ppo_trainer = PPOTrainer(
        ppo_sample_config,
        model=basis_model,
        tokenizer=tokenizer,
        ref_model=ref_model, 
        dataset=dataset,
        data_collator=collator,
        optimizer=optimizer,
    )

    # We then build the reward pipeline, we will use the toxicity model to compute the reward.
    # We first load the toxicity model and tokenizer.
    toxicity_tokenizer = RobertaTokenizer.from_pretrained(script_args.reward_function)
    # We load the toxicity model in fp16 to save memory.
    toxicity_model = RobertaForSequenceClassification.from_pretrained(script_args.reward_function, torch_dtype=torch.float16).to(
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
                original_warp_creator = basis_model.pretrained_model._get_logits_warper
                updated_get_logits_warper = update_get_logits_warper(original_warp_creator, ppo_trainer_for_products.model)
                basis_model.pretrained_model._get_logits_warper = updated_get_logits_warper.__get__(basis_model.pretrained_model, basis_model.pretrained_model.__class__)
                
                # Need to save ref_response for KL divergence 
                response = sample_ppo_trainer.generate(query, **generation_kwargs)
                
                # validate response tensors
                if response is None:
                    empty_response_counter += 1                
                
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
        if epoch % 2 == 0:
            if ppo_trainer_for_products.accelerator.is_main_process:
                ppo_trainer_for_products.save_pretrained(config['directories']['model_save_path'])
                
                

if __name__ == "__main__":
    
    # set up parser
    parser = argparse.ArgumentParser(description='RedGreen Adapter Training.')
    parser.add_argument('--config_path', type=str, default= 'config.ini', help='Path to config file.')
    parser.add_argument('--rl_method', choices=['PPO', 'DPO', 'KTO'], default='PPO', help='Choose Adapter Learning Method.')
    parser.add_argument('--use_wandb', default="True")

    args = parser.parse_args()

    config_path = args.config_path
    config = configparser.ConfigParser()
    config.read(config_path)
    
    config_dict = config._sections
    config_dict['logs']['use_wandb'] = args.use_wandb
        
    if args.rl_method == "PPO":
        train_ppo(config_dict)
    



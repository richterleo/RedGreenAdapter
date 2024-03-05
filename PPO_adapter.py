import math
import torch
import typing

from dataclasses import dataclass, field
from datasets import Dataset
from typing import Optional, Union, Callable, List

from transformers import (
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    PreTrainedTokenizerBase, 
    PreTrainedTokenizer, 
    PreTrainedTokenizerFast
)

from trl import (
    PPOTrainer, 
    PPOConfig, 
    PreTrainedModelWrapper
)

from trl.core import PPODecorators, logprobs_from_logits

@dataclass
class PPOwithAdapterConfig(PPOConfig):
    
    '''
    Configuration class for PPOwithAdapterTrainer. 
    This replaces the attribute model_name from PPOConfig with both a base_model_name and adapter_model_name.
    '''
    base_model_name: Optional[str] = None
    adapter_model_name: Optional[str] = None
    
    model_name: Optional[str] = field(default=None, metadata={'deprecated': True})
    


class PPOwithAdapterTrainer(PPOTrainer):
    '''
    '''
    def __init__(
        self,
        config: PPOwithAdapterConfig = None,
        product_model: PreTrainedModelWrapper = None,
        tokenizer:PreTrainedTokenizerBase = None,
        ref_model: Optional[PreTrainedModelWrapper] = None,
        dataset: Optional[Union[torch.utils.data.Dataset, Dataset]] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        data_collator: Optional[typing.Callable] = None,
        num_shared_layers: Optional[int] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ):
        
        super().__init__(
            # PPOwithAdapterConfig passes type check of PPOConfig
            config=config, 
            model=product_model,
            ref_model=ref_model, 
            tokenizer=tokenizer,
            dataset=dataset, 
            optimizer=optimizer, 
            data_collator=data_collator, 
            num_shared_layers=num_shared_layers, 
            lr_scheduler=lr_scheduler
        )
        
        # Additional type checks
        if not isinstance(config, PPOwithAdapterConfig):
            raise ValueError(f"config must be a PPOwithAdapterConfig, got {type(config)}")
        
        # Base model is not updated, gradients are not recorded
        
        # Data collator depends on tokenizer
        # TODO: find out if it's a problem that the accelerator.prepare method uses the other data collator


        # TODO: find way to rename the model attribute (and everything depending on that) to adapter_model
        
        
        
    def generate_base(
        self,
        query_tensor: Union[torch.Tensor, List[torch.Tensor]],
        length_sampler: Callable = None,
        batch_size: int = 4,
        return_prompt: bool = True,
        **generation_kwargs,
    ):
        """
        Generate response with the model given the query tensor.
        call the `generate` method of the model.

        Args:
            query_tensor (`torch.LongTensor`):
                A tensor of shape (`seq_len`) containing query tokens or a list of tensors of shape (`seq_len`).
            length_sampler (`Callable`, *optional*):
                Callable that returns the number of newly generated tokens.
            batch_size (`int`, *optional):
                Batch size used for generation, defaults to `4`.
            return_prompt (`bool`, *optional*):
                If set to `False` the prompt is not returned but only the newly generated tokens, defaults to `True`.
            generate_ref_response (`bool`, *optional*):
                If set to `True` the reference response is also generated, defaults to `False`.
            generation_kwargs (dict[str, Any]):
                Keyword arguments for generation.

        Returns:
            `torch.LongTensor`: A tensor of shape (`batch_size`, `gen_len`) containing response tokens.
        """

        if isinstance(query_tensor, List):
            response = self._generate_batched(
                self.base_model,
                query_tensor,
                length_sampler=length_sampler,
                batch_size=batch_size,
                return_prompt=return_prompt,
                **generation_kwargs,
            )

        else:
            if len(query_tensor.shape) == 2:
                raise ValueError(
                    "query_tensor must be a tensor of shape (`seq_len`) or a list of tensors of shape (`seq_len`)"
                )

            if length_sampler is not None:
                generation_kwargs["max_new_tokens"] = length_sampler()
            response = self.accelerator.unwrap_model(self.base_model).generate(
                input_ids=query_tensor.unsqueeze(dim=0), **generation_kwargs
            )

            if not return_prompt:
                response = response[:, query_tensor.shape[0] :]


        return response


class PPOTrainerForProducts(PPOTrainer):
    
    def __init__(
        self,
        config: PPOConfig = None,
        model: PreTrainedModelWrapper = None,
        source_model: PreTrainedModelWrapper=None,
        tokenizer:PreTrainedTokenizerBase = None,
        ref_model: Optional[PreTrainedModelWrapper] = None,
        dataset: Optional[Union[torch.utils.data.Dataset, Dataset]] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        data_collator: Optional[typing.Callable] = None,
        num_shared_layers: Optional[int] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ):
        
        super().__init__(
            # PPOwithAdapterConfig passes type check of PPOConfig
            config=config, 
            model=model,
            ref_model=ref_model, 
            tokenizer=tokenizer,
            dataset=dataset, 
            optimizer=optimizer, 
            data_collator=data_collator, 
            num_shared_layers=num_shared_layers, 
            lr_scheduler=lr_scheduler
        )
        
        self.source_model = source_model
     
    @PPODecorators.empty_device_cache()    
    def batched_forward_pass(
        self,
        model: PreTrainedModelWrapper,
        queries: torch.Tensor,
        responses: torch.Tensor,
        model_inputs: dict,
        return_logits: bool = False,
        response_masks: Optional[torch.Tensor] = None,
    ):

        bs = len(queries)
        fbs = self.config.mini_batch_size
        all_logprobs = []
        all_logits = []
        all_masks = []
        all_values = []

        model.eval()

        for i in range(math.ceil(bs / fbs)):
            input_kwargs = {key: value[i * fbs : (i + 1) * fbs] for key, value in model_inputs.items()}
            query_batch = queries[i * fbs : (i + 1) * fbs]
            response_batch = responses[i * fbs : (i + 1) * fbs]
            if response_masks is not None:
                response_masks_batch = response_masks[i * fbs : (i + 1) * fbs]
            
            # these values have to be computed differently
            logits, values = self._get_product(model, input_kwargs)

            if self.is_encoder_decoder:
                input_ids = input_kwargs["decoder_input_ids"]
                attention_mask = input_kwargs["decoder_attention_mask"]
            else:
                input_ids = input_kwargs["input_ids"]
                attention_mask = input_kwargs["attention_mask"]

            logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])
            masks = torch.zeros_like(attention_mask)
            masks[:, :-1] = attention_mask[:, 1:]

            for j in range(len(query_batch)):
                if self.is_encoder_decoder:
                    # Decoder sentence starts always in the index 1 after padding in the Enc-Dec Models
                    start = 1
                    end = attention_mask[j, :].sum() - 1
                else:
                    start = len(query_batch[j]) - 1  # logprobs starts from the second query token
                    if attention_mask[j, 0] == 0:  # offset left padding
                        start += attention_mask[j, :].nonzero()[0]
                    end = start + len(response_batch[j])
                    if response_masks is not None:
                        response_masks_batch[j] = torch.cat(
                            (torch.zeros_like(query_batch[j]), response_masks_batch[j])
                        )[1:]

                masks[j, :start] = 0
                masks[j, end:] = 0
                if response_masks is not None:
                    masks[j, start:end] = masks[j, start:end] * response_masks_batch[j][start:end]

            if return_logits:
                all_logits.append(logits)
            else:
                del logits
            all_values.append(values)
            all_logprobs.append(logprobs)
            all_masks.append(masks)

        return (
            torch.cat(all_logprobs),
            torch.cat(all_logits)[:, :-1] if return_logits else None,
            torch.cat(all_values)[:, :-1],
            torch.cat(all_masks)[:, :-1],
        )
    
    def _get_product(self, model, input_kwargs):
        '''
        '''
        
        # TODO: the values also need to be calculated differently
        logits, _, values = model(**input_kwargs)
        
        
        self.source_model.eval()
        
        with torch.inference_mode():
            source_logits, _, _ = self.source_model(**input_kwargs)
        
        return logits + source_logits, values
    

    
        
        
        
        
if __name__ == "__main__":
    
    @dataclass
    class ScriptArguments:
        """
        The name of the Casual LM model we wish to fine-tune with PPO
        """

        # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
        # models like gpt-neo* models are more suitable.
        base_model_name: Optional[str] = field(default="ybelkada/gpt-j-6b-sharded-bf16", metadata={"help": "the base model name"})
        adapter_model_name: Optional[str] = field(default="ybelkada/gpt-j-6b-sharded-bf16", metadata={"help": "the base model name"})
        log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
        learning_rate: Optional[float] = field(default=(1.47e-5) * 2, metadata={"help": "the learning rate"})
        mini_batch_size: Optional[int] = field(default=4, metadata={"help": "the PPO minibatch size"})
        batch_size: Optional[int] = field(default=16, metadata={"help": "the batch size"})
        gradient_accumulation_steps: Optional[int] = field(
            default=1, metadata={"help": "the number of gradient accumulation steps"}
        )
        model_save_path: Optional[str] = field(
            default="./gpt-j-6B-detoxified-long-context-26-shl-1e4-final",
            metadata={"help": "the path to save the model"},
        )


    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    config = PPOwithAdapterConfig(
        base_model_name=script_args.base_model_name,
        adapter_model_name=script_args.adapter_model_name,
        learning_rate=script_args.learning_rate,
        log_with=script_args.log_with,
        ppo_epochs=10, # NOTE: changed this to 10
        mini_batch_size=script_args.mini_batch_size,
        batch_size=script_args.batch_size,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    )
    
    print(config.__dict__)
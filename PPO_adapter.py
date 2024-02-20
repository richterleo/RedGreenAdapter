from trl import PPOTrainer, PPOConfig, PreTrainedModelWrapper, PreTrainedTokenizerBase

import torch
from typing import Optional, Union, Callable, List
import typing
from datasets import Dataset

from dataclasses import dataclass, field


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
        base_model: PreTrainedModelWrapper = None,
        adapter_model: PreTrainedModelWrapper = None,
        adapter_ref_model: Optional[PreTrainedModelWrapper] = None,
        adapter_tokenizer: PreTrainedTokenizerBase = None,
        dataset: Optional[Union[torch.utils.data.Dataset, Dataset]] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        data_collator: Optional[typing.Callable] = None,
        num_shared_layers: Optional[int] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ):
        
        super().__init__(
            # PPOwithAdapterConfig passes type check of PPOConfig
            config=config, 
            model=adapter_model,
            ref_model=adapter_ref_model, 
            tokenizer=adapter_tokenizer,
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
        self.base_model = base_model
        for param in self.base_model.parameters():
            param.requires_grad = False

        # "model" in PPOTrainer class is now "adapter_model"
        self.adapter_model = adapter_model
        self.adapter_model_params = filter(lambda p: p.requires_grad, self.adapter_model.parameters())
        self.is_encoder_decoder = hasattr(self.model, "is_encoder_decoder")
        self.is_peft_model = getattr(self.model, "is_peft_model", False)
        config.is_encoder_decoder = self.is_encoder_decoder
        config.is_peft_model = self.is_peft_model
        
        
    def generate_base_response(
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
            response = self.accelerator.unwrap_model(self.model).generate(
                input_ids=query_tensor.unsqueeze(dim=0), **generation_kwargs
            )

            if not return_prompt and not self.is_encoder_decoder:
                response = response[:, query_tensor.shape[0] :]


        return response


        
        
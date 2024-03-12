import torch
import torch.nn as nn

from datasets import Dataset
from typing import Optional, Union, Callable, List, Dict, Tuple
from transformers import (
    PreTrainedModel, 
    TrainingArguments, 
    PreTrainedTokenizerBase,
    AutoModelForCausalLM
)
from trl import DPOTrainer


class DPOTrainerForProducts(DPOTrainer):
    
    def __init__(
        self,
        model : Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        basis_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None, # This is the only new thing
        ref_model : Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        args : Optional[TrainingArguments] = None,
        beta: float = 0.1,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        max_length: Optional[int] = None,
        max_prompt_length: Optional[int] = None,
        max_target_length: Optional[int] = None,
        peft_config: Optional[Dict] = None,
        generate_during_eval: bool = False,
    ):
        
        super().__init__(
            model = model,
            ref_model = ref_model,
            args=args,
            beta=beta,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            max_length=max_length,
            max_target_length=max_target_length,
            max_prompt_length=max_prompt_length,
            generate_during_eval=generate_during_eval,
            peft_config=peft_config
        )
        
        self.basis_model = basis_model
        
    
    def concatenated_forward(
        self, 
        model: nn.Module, 
        batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = self.concatenated_inputs(
            batch,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device,
        )
        len_chosen = batch["chosen_labels"].shape[0]

        model_kwargs = (
            {
                "labels": concatenated_batch["concatenated_labels"],
                "decoder_input_ids": concatenated_batch.pop("concatenated_decoder_input_ids", None),
            }
            if self.is_encoder_decoder
            else {}
        )
        all_logits = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            use_cache=False,
            **model_kwargs,
        ).logits

        all_logps = self.get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            average_log_prob=self.loss_type == "ipo",
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]
        
        # get logits from basis model
        with torch.inference_mode():
            all_logits_basis_model = self.basis_model(
                concatenated_batch["concatenated_input_ids"],
                attention_mask=concatenated_batch["concatenated_attention_mask"],
                use_cache=False,
                **model_kwargs,
            ).logits

            all_logps = self.get_batch_logps(
                all_logits_basis_model,
                concatenated_batch["concatenated_labels"],
                average_log_prob=self.loss_type == "ipo",
                is_encoder_decoder=self.is_encoder_decoder,
                label_pad_token_id=self.label_pad_token_id,
            )
        
        chosen_logps_basis_model = all_logits_basis_model[:len_chosen]
        rejected_logps_basis_model = all_logits_basis_model[len_chosen:]
        
        chosen_logits_basis_model = all_logits_basis_model[:len_chosen]
        rejected_logits_basis_model = all_logits_basis_model[len_chosen:]
        
        # add logits
        chosen_logits = chosen_logits + chosen_logits_basis_model
        rejected_logits = rejected_logits + rejected_logits_basis_model
        
        # multiply probabilities
        chosen_logps = torch.mul(chosen_logps, chosen_logps_basis_model)
        rejected_logps = torch.mul(rejected_logps, rejected_logps_basis_model)


        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits)
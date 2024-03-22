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

from product_of_experts import update_get_logits_warper


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
        
        self.basis_model = self.accelerator.prepare_model(self.base_model, evaluation_mode=True)
        
    
    def concatenated_forward(
        self, 
        model: nn.Module, 
        batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

        We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        print(self.accelerator.device)
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
        
        # get logits from basis model
        
        self.basis_model.eval() #TODO: think about whether we want the basis model in eval or train mode
        
        with torch.inference_mode():
            all_logits_basis_model = self.basis_model(
                concatenated_batch["concatenated_input_ids"],
                attention_mask=concatenated_batch["concatenated_attention_mask"],
                use_cache=False,
                **model_kwargs,
            ).logits


        all_logits += all_logits_basis_model # add basis model logits before softmax

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

        

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits)
    
    
    def get_batch_samples(self,
                          model,
                          batch: Dict[str, torch.LongTensor]) -> Tuple[str, str]:
        '''
        
        '''
        generate_context_manager = nullcontext if not self._peft_has_been_casted_to_bf16 else torch.cuda.amp.autocast

        
        # we need to change how this output is generated
        with generate_context_manager():
            
            # we add a logits warper that is based on the adapter. This will be a truncated version
            # We do this by monkey patching: by using 
            original_warp_creator = self.basis_model._get_logits_warper
            updated_get_logits_warper = update_get_logits_warper(original_warp_creator, model)
            self.basis_model.pretrained_model._get_logits_warper = updated_get_logits_warper.__get__(self.basis_model.pretrained_model, 
                                                                                                self.basis_model.pretrained_model.__class__)
        
            policy_output = self.basis_model.generate(
                input_ids=batch["prompt_input_ids"],
                attention_mask=batch["prompt_attention_mask"],
                max_length=self.max_length,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            # if reference_output in batch use that otherwise use the reference model
            if "reference_output" in batch:
                reference_output = batch["reference_output"]
            else:
                if self.ref_model is None:
                    with self.null_ref_context():
                        reference_output = self.model.generate(
                            input_ids=batch["prompt_input_ids"],
                            attention_mask=batch["prompt_attention_mask"],
                            max_length=self.max_length,
                            do_sample=True,
                            pad_token_id=self.tokenizer.pad_token_id,
                        )
                else:
                    reference_output = self.ref_model.generate(
                        input_ids=batch["prompt_input_ids"],
                        attention_mask=batch["prompt_attention_mask"],
                        max_length=self.max_length,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )

        policy_output = pad_to_length(policy_output, self.max_length, self.tokenizer.pad_token_id)
        policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)

        reference_output = pad_to_length(reference_output, self.max_length, self.tokenizer.pad_token_id)
        reference_output_decoded = self.tokenizer.batch_decode(reference_output, skip_special_tokens=True)

        return policy_output_decoded, reference_output_decoded 
from dataclasses import dataclass, field
from typing import List, Tuple, Literal, Union, Optional

from transformers import HfArgumentParser, TrainingArguments
from trl import ModelConfig

from utils import create_run_string


@dataclass
class PPOArgs:
    '''
    
    '''
    dataset_name: Optional[str] = field(default="allenai/real-toxicity-prompts")

    learning_rate: Optional[float] = field(default=(1.47e-5) * 2, metadata={"help": "the learning rate"})
    mini_batch_size: Optional[int] = field(default=4, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=16, metadata={"help": "the batch size"})
    adapter_model_top_p: Optional[float] = field(default=0.9, metadata={"help": "the top-p value for thresholded adapter"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    reward_function: Optional[str] = field(
        default='facebook/roberta-hate-speech-dynabench-r4-target', 
        metadata={"help": "model used as reward signal"})
    ppo_epochs:Optional[float] = field(default=10, metadata={"help": "number of epochs for PPO training"})
    

    
    
@dataclass
class DPOArgs:
    """
    The name of the Casual LM model we wish to fine-tune with PPO
    """

    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable.
    
    
    dataset_name: Optional[str] = field(default="Anthropic/hh-rlhf")
    
    beta: float = field(default=0.1, metadata={"help": "the beta parameter for DPO loss"})
    max_length: int = field(default=512, metadata={"help": "max length of each sample"})
    max_prompt_length: int = field(default=128, metadata={"help": "max length of each sample's prompt"})
    max_target_length: int = field(
        default=128, metadata={"help": "Only used for encoder decoder model. Max target of each sample's prompt"}
    )
    sanity_check: bool = field(default=True, metadata={"help": "only train on 1000 samples"})
    ignore_bias_buffers: bool = field(
        default=False,
        metadata={
            "help": "debug argument for distributed training;"
            "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )
    generate_during_eval: bool = field(default=False, metadata={"help": "Generate during evaluation"})
    
    # model_name_or_path: str= field(default="gpt2", metadata={"help": "only train on 1000 samples"})
    # per_device_train_batch_size: int = field(default=4, metadata={"help": "only train on 1000 samples"})
    # max_steps : int = field(default=1000, metadata={"help": "only train on 1000 samples"})
    # learning_rate: Optional[float] =field(default=1e-3 , metadata={"help": "only train on 1000 samples"})
    # gradient_accumulation_steps: int = field(default=1, metadata={"help": "only train on 1000 samples"})
    # logging_step: int = field(default=10, metadata={"help": "only train on 1000 samples"}) 
    # eval_steps: int = field(default=500, metadata={"help": "only train on 1000 samples"}) 
    # output_dir: str= field(default="dpo_anthropic_hh", metadata={"help": "only train on 1000 samples"}) 
    # warmup_steps: int= field(default=150, metadata={"help": "only train on 1000 samples"}) 
    # report_to_wandb: int = field(default=False, metadata={"help": "only train on 1000 samples"})
    # bf16 : int= field(default=False, metadata={"help": "only train on 1000 samples"})
    # logging_first_step : int= field(default=False, metadata={"help": "only train on 1000 samples"})
    # no_remove_unused_columns : Optional[int]= field(default=True, metadata={"help": "only train on 1000 samples"})
    
    
    
training_args = {
    "model_name_or_path" :"gpt2",
    "per_device_train_batch_size": 4,
    "max_steps": 1000,
    "learning_rate": 1e-3,
    "gradient_accumulation_steps" :1,
    "logging_steps": 10,
    "eval_steps" :500,
    "output_dir":"dpo_anthropic_hh",
    "warmup_steps" :150,
    "report_to": "wandb",
    # "load_best_model_at_end": True,
    "generate_during_eval": True,
    "run_name": create_run_string()}
    # bf16,
    # logging_first_step)
  
    
    
    


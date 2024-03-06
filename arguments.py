import torch
#import argparse

from dataclasses import dataclass, field
from typing import List, Tuple, Literal, Union, Optional

@dataclass
class Args:
    HOME_PATH = '/root/RedGreenAdapter'
    
    # Basic / global
    seed: int = 1
    cuda: bool = torch.cuda.is_available()
    num_gpus: int = torch.cuda.device_count()
    cuda_deterministic: bool = True # sets flags for determinism when using CUDA (potentially slow!)
    
    # Directories and paths
    output_dir: str = f'{HOME_PATH}/RealToxicityPrompts'
    resume_dir: Optional[str] = None # directory to resume generation
    
    # Logging / Wandb
    use_wandb: bool = False
    exp_name: str = "IPA_Implementation"
    log_dir: str = "logs"
    wandb_project_name: str = "IPA_Implementation"
    wandb_entity: str = None
    
    # General RL params
    rl_method : Literal["PPO", "DPO", "KTO"] = "PPO"
    
    # Dataset 
    dataset_name: str = "allenai/real-toxicity-prompts"
    

    

    
    
@dataclass
class Arguments:
    """
    The name of the Casual LM model we wish to fine-tune with PPO
    """

    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable.
    
    # General RL params
    rl_method: Literal["PPO", "DPO", "KTO"] = field(default="PPO", metadata={"help": "Which RL method to use"})
    
    # Models
    basis_model_name: Optional[str] = field(default="openai-community/gpt2-large", metadata={"help": "the base model name"})
    adapter_model_name: Optional[str] = field(default="openai-community/gpt2", metadata={"help": "the base model name"})
    
    # Dataset params
    dataset_name: Optional[str] = field(default="allenai/real-toxicity-prompts")
    
    # Logging/Wandb
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    exp_name: Optional[str] = field(default="IPA_Implementation", metadata={"help": "name of experiment for logging"})
    log_dir: Optional[str] = field(default="logs", metadata={"help": ""})
    wandb_project_name: str = "IPA_Implementation"
    
    
    # PPO params
    learning_rate: Optional[float] = field(default=(1.47e-5) * 2, metadata={"help": "the learning rate"})
    mini_batch_size: Optional[int] = field(default=4, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=16, metadata={"help": "the batch size"})
    adapter_model_top_p: Optional[float] = field(default=0.9, metadata={"help": "the top-p value for thresholded adapter"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    
    # DPO params
    
    
    # Directories and paths
    home_dir: Optional[str] = field(default='/root/RedGreenAdapter', metadata={"help": "Home directory"})
    model_save_path: Optional[str] = field(
        default="Test",
        metadata={"help": "The path to save the model"},
        
    )
    


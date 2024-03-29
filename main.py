import argparse
import configparser
import os

from tqdm import tqdm
from typing import Optional

from arguments import PPOArgs, DPOArgs, training_args
from train import train_ppo, train_dpo

os.environ["WANDB_PROJECT"] = "RedGreen_Adapter"
os.environ["WANDB_API_KEY"] = "1c84a4abed1d390fbe37478c7cb82a84e4650881"
os.environ["WANDB_LOG_LEVEL"] = "debug"
os.environ["WANDB_DISABLE_FORK"] = "true"
                             

if __name__ == "__main__":
    
    # set up parser
    parser = argparse.ArgumentParser(description='RedGreen Adapter Training.')
    parser.add_argument('--config_path', type=str, default= 'config.ini', help='Path to config file.')
    parser.add_argument('--rl_method', choices=['PPO', 'DPO', 'KTO'], default='DPO', help='Choose Adapter Learning Method.')
    parser.add_argument('--use_wandb', default="True")

    args = parser.parse_args()

    config_path = args.config_path
    config = configparser.ConfigParser()
    config.read(config_path)
    
    config_dict = config._sections
    config_dict['logs']['use_wandb'] = args.use_wandb
        
    if args.rl_method == "PPO":
        train_ppo(config_dict, PPOArgs)
        
    elif args.rl_method == "DPO":
        print(f"We are using wandb: {training_args['report_to']}")
        train_dpo(config_dict, DPOArgs, training_args)
    



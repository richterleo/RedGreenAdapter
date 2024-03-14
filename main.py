import argparse
import configparser
import os

from tqdm import tqdm
from typing import Optional

from arguments import PPOArgs, DPOArgs, training_args
from train import train_ppo, train_dpo

os.environ["WANDB_PROJECT"] = "RedGreen_Adapter"
                             

if __name__ == "__main__":
    
    # set up parser
    parser = argparse.ArgumentParser(description='RedGreen Adapter Training.')
    parser.add_argument('--config_path', type=str, default= 'config.ini', help='Path to config file.')
    parser.add_argument('--rl_method', choices=['PPO', 'DPO', 'KTO'], default='DPO', help='Choose Adapter Learning Method.')
    parser.add_argument('--use_wandb', default="False")

    args = parser.parse_args()

    config_path = args.config_path
    config = configparser.ConfigParser()
    config.read(config_path)
    
    config_dict = config._sections
    config_dict['logs']['use_wandb'] = args.use_wandb
        
    if args.rl_method == "PPO":
        train_ppo(config_dict, PPOArgs)
        
    elif args.rl_method == "DPO":
        train_dpo(config_dict, DPOArgs, training_args)
    



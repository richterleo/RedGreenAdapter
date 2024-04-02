import argparse
import configparser
import logging
import os
import sys
import wandb

from pathlib import Path
from typing import Optional

from arguments import PPOArgs, DPOArgs, training_args
from train import train_ppo, train_dpo

os.environ["WANDB_PROJECT"] = "RedGreen_Adapter"
os.environ["WANDB_API_KEY"] = "1c84a4abed1d390fbe37478c7cb82a84e4650881"
os.environ["WANDB_LOG_LEVEL"] = "debug"
os.environ["WANDB_DISABLE_FORK"] = "true"
                             
# # Create a custom logger
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

# # Create handlers
# file_handler = logging.FileHandler('output.log')
# file_handler.setLevel(logging.INFO)

# # Create formatters and add it to handlers
# log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# file_handler.setFormatter(log_format)

# # Add handlers to the logger
# logger.addHandler(file_handler)

# # Replace print with logging
# print = logger.info

# # Now, instead of using print statements, use logger.info, logger.warning, etc.
# logger.info("This is an info message.")
# logger.warning("This is a warning message.")

# # Redirect stderr to the log file as well
# sys.stderr = open('output.log', 'a')



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
        run_name = train_dpo(config_dict, DPOArgs, training_args)
        
        api = wandb.Api()
        run = api.run(run_name)
        if Path('output.txt').exists():
            run.file('output.txt').upload()
    



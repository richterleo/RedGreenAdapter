# This is an example of a job script

#!/bin/bash

# Created by Leo Richter 2024-03-18
# Does DPO training using GPT2-large base model and GPT2 adapter

#$ -l tmem=6G
#$ -l h_rt=04:55:30
#$ -S /bin/bash
#$ -N test_dpo
#$ -cwd

#$ -l gpu=true
#$ -pe gpu 2
#$ -R y

#These are optional flags but you probably want them in all jobs
#$ -j y
hostname
date
conda activate redgreen
echo "$@"

# UPDATE FOR YOUR ENVIRONMENT
export WANDB_API_KEY=1c84a4abed1d390fbe37478c7cb82a84e4650881
accelerate launch --num_processes=2 main.py
date

#!/bin/bash

# Prevent auto tmux
touch ~/.no_auto_tmux

# Create directory for Miniconda and download the installer
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh

# Install Miniconda
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3

# Remove the installer script
rm -rf ~/miniconda3/miniconda.sh

# Initialize Conda for Bash shell
~/miniconda3/bin/conda init bash

# Source the bashrc to refresh the environment
source ~/.bashrc

# # Clone github repository and cd into folder
# git clone https://github.com/richterleo/RedGreenAdapter.git
# cd RedGreenAdapter

# Create a Conda environment
echo "Creating conda environment"
conda env create -f environment.yml

# Add conda activation command to bashrc
echo "conda activate redgreenvenv" >> ~/.bashrc

# Activate the Conda environment
# conda activate redgreenvenv

# Add trl library
pip install -U git+https://github.com/huggingface/trl



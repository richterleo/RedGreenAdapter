# Inference-time Policy Adapter (IPA)

This repository is an implementation of a red-green list adapter for LLMs. It is based on ["Inference-Time Policy Adapters (IPA): Tailoring Extreme-Scale LMs without Fine-tuning"](https://arxiv.org/abs/2305.15065) (EMNLP 2023) and the [trl library](https://huggingface.co/docs/trl/en/index). 

In particular, the library is inspired by [this example script](https://github.com/huggingface/trl/blob/main/examples/research_projects/toxicity/scripts/gpt-j-6b-toxicity.py) for detoxifying language models using PPO and [this example script](https://github.com/huggingface/trl/blob/main/examples/scripts/dpo.py) for DPO. 

## Requirement
We suggest using conda to setup environment. You need to first replace ``prefix`` in [environment.yml](environment.yml) with your home path. With conda installed, create an environment called `redgreen` with:
```
conda env create -f environment.yml
```

## Instruction

### PPO

### DPO

You can train a LLM adapter on the [Anthropic Helpful-Harmless dataset](https://huggingface.co/datasets/Anthropic/hh-rlhf) available on Hugging Face to guide a basis LM towards safer behaviour. 
[tbc]

```
accelerate launch --config_file '~/RedGreenAdapter/accelerate_config.yml' main.py 
```
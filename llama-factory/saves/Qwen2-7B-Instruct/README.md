---
base_model: Qwen/Qwen2-7B-Instruct
library_name: peft
license: other
tags:
- llama-factory
- lora
- generated_from_trainer
model-index:
- name: Qwen2-7B-Instruct
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# Qwen2-7B-Instruct

This model is a fine-tuned version of [Qwen/Qwen2-7B-Instruct](https://huggingface.co/Qwen/Qwen2-7B-Instruct) on the alpaca_mac dataset.
It achieves the following results on the evaluation set:
- Loss: 1.6234

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0001
- train_batch_size: 16
- eval_batch_size: 1
- seed: 42
- gradient_accumulation_steps: 8
- total_train_batch_size: 128
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 6.0

### Training results

| Training Loss | Epoch  | Step | Validation Loss |
|:-------------:|:------:|:----:|:---------------:|
| 1.715         | 0.9964 | 35   | 1.5825          |
| 1.5599        | 1.9929 | 70   | 1.5090          |
| 1.4363        | 2.9893 | 105  | 1.5067          |
| 1.2383        | 3.9858 | 140  | 1.5461          |
| 1.1633        | 4.9822 | 175  | 1.6060          |
| 1.1257        | 5.9786 | 210  | 1.6234          |


### Framework versions

- PEFT 0.11.1
- Transformers 4.43.3
- Pytorch 2.4.0+cu121
- Datasets 2.19.1
- Tokenizers 0.19.1
---
base_model: shenzhi-wang/Llama3.1-70B-Chinese-Chat
library_name: peft
license: other
tags:
- llama-factory
- lora
- generated_from_trainer
model-index:
- name: Llama3.1-70B-Chinese-Chat
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# Llama3.1-70B-Chinese-Chat

This model is a fine-tuned version of [shenzhi-wang/Llama3.1-70B-Chinese-Chat](https://huggingface.co/shenzhi-wang/Llama3.1-70B-Chinese-Chat) on the alpaca_mac dataset.
It achieves the following results on the evaluation set:
- Loss: 2.5071

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
- train_batch_size: 2
- eval_batch_size: 1
- seed: 42
- distributed_type: multi-GPU
- num_devices: 4
- gradient_accumulation_steps: 8
- total_train_batch_size: 64
- total_eval_batch_size: 4
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 6.0

### Training results

| Training Loss | Epoch  | Step | Validation Loss |
|:-------------:|:------:|:----:|:---------------:|
| 1.4367        | 0.9982 | 70   | 1.3731          |
| 1.2601        | 1.9964 | 140  | 1.3131          |
| 0.8929        | 2.9947 | 210  | 1.4369          |
| 0.383         | 3.9929 | 280  | 1.7250          |
| 0.1431        | 4.9911 | 350  | 2.0897          |
| 0.0691        | 5.9893 | 420  | 2.5071          |


### Framework versions

- PEFT 0.11.1
- Transformers 4.43.3
- Pytorch 2.4.0+cu121
- Datasets 2.19.1
- Tokenizers 0.19.1
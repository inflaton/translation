---
base_model: internlm/internlm2_5-7b-chat
library_name: peft
license: other
tags:
- llama-factory
- lora
- generated_from_trainer
model-index:
- name: internlm2_5-7b-chat
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# internlm2_5-7b-chat

This model is a fine-tuned version of [internlm/internlm2_5-7b-chat](https://huggingface.co/internlm/internlm2_5-7b-chat) on the alpaca_mac dataset.
It achieves the following results on the evaluation set:
- Loss: 1.5571

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
| 1.7363        | 0.9964 | 35   | 1.6335          |
| 1.5711        | 1.9929 | 70   | 1.5236          |
| 1.4657        | 2.9893 | 105  | 1.5022          |
| 1.2867        | 3.9858 | 140  | 1.5092          |
| 1.2283        | 4.9822 | 175  | 1.5432          |
| 1.1824        | 5.9786 | 210  | 1.5571          |


### Framework versions

- PEFT 0.11.1
- Transformers 4.43.3
- Pytorch 2.4.0+cu121
- Datasets 2.19.1
- Tokenizers 0.19.1
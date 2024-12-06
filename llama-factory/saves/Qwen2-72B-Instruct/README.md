---
base_model: Qwen/Qwen2-72B-Instruct
library_name: peft
license: other
tags:
- llama-factory
- lora
- generated_from_trainer
model-index:
- name: Qwen2-72B-Instruct
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# Qwen2-72B-Instruct

This model is a fine-tuned version of [Qwen/Qwen2-72B-Instruct](https://huggingface.co/Qwen/Qwen2-72B-Instruct) on the alpaca_mac dataset.
It achieves the following results on the evaluation set:
- Loss: 2.6303

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
- gradient_accumulation_steps: 8
- total_train_batch_size: 16
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 6.0

### Training results

| Training Loss | Epoch  | Step | Validation Loss |
|:-------------:|:------:|:----:|:---------------:|
| 1.5022        | 0.2499 | 70   | 1.4518          |
| 1.4121        | 0.4998 | 140  | 1.3727          |
| 1.4038        | 0.7497 | 210  | 1.3051          |
| 1.2739        | 0.9996 | 280  | 1.2890          |
| 1.1436        | 1.2494 | 350  | 1.3195          |
| 1.0783        | 1.4993 | 420  | 1.3106          |
| 1.1219        | 1.7492 | 490  | 1.3045          |
| 1.0966        | 1.9991 | 560  | 1.3094          |
| 0.6134        | 2.2490 | 630  | 1.4946          |
| 0.6342        | 2.4989 | 700  | 1.4859          |
| 0.6665        | 2.7488 | 770  | 1.5236          |
| 0.6101        | 2.9987 | 840  | 1.5220          |
| 0.2467        | 3.2485 | 910  | 1.8390          |
| 0.2284        | 3.4984 | 980  | 1.8253          |
| 0.2839        | 3.7483 | 1050 | 1.8688          |
| 0.2111        | 3.9982 | 1120 | 1.8910          |
| 0.0753        | 4.2481 | 1190 | 2.2224          |
| 0.072         | 4.4980 | 1260 | 2.3093          |
| 0.0351        | 4.7479 | 1330 | 2.2221          |
| 0.0644        | 4.9978 | 1400 | 2.2804          |
| 0.0257        | 5.2477 | 1470 | 2.5593          |
| 0.0249        | 5.4975 | 1540 | 2.6220          |
| 0.0238        | 5.7474 | 1610 | 2.6189          |
| 0.0262        | 5.9973 | 1680 | 2.6303          |


### Framework versions

- PEFT 0.11.1
- Transformers 4.43.3
- Pytorch 2.4.0+cu121
- Datasets 2.19.1
- Tokenizers 0.19.1
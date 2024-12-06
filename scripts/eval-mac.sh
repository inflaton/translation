#!/bin/sh

BASEDIR=$(dirname "$0")
cd $BASEDIR/..
echo Current Directory:
pwd

nvidia-smi
uname -a
cat /etc/os-release
lscpu
grep MemTotal /proc/meminfo

# pip install torch torchvision torchaudio
# pip install -r requirements.txt
# pip install --upgrade transformers

# export START_NUM_SHOTS=50

export RESULTS_PATH=results/mac-results_few_shots.csv

# ./scripts/eval-model.sh internlm/internlm2_5-7b-chat

# ./scripts/eval-model.sh Qwen/Qwen2-7B-Instruct

# ./scripts/eval-model.sh shenzhi-wang/Mistral-7B-v0.3-Chinese-Chat

# ./scripts/eval-model.sh shenzhi-wang/Llama3.1-8B-Chinese-Chat

# ./scripts/eval-model.sh microsoft/Phi-3.5-mini-instruct

export RESULTS_PATH=results/mac-results_fine_tuned.csv

# ./scripts/eval-epochs.sh internlm internlm2_5-7b-chat

# ./scripts/eval-epochs.sh Qwen Qwen2-7B-Instruct

# ./scripts/eval-epochs.sh shenzhi-wang Mistral-7B-v0.3-Chinese-Chat

# ./scripts/eval-epochs.sh shenzhi-wang Llama3.1-8B-Chinese-Chat

# ./scripts/eval-epochs.sh microsoft Phi-3.5-mini-instruct

export MAX_NEW_TOKENS=2048
export START_REPETITION_PENALTY=1.1
export END_REPETITION_PENALTY=1.1

export USING_CHAT_TEMPLATE=false
export RESULTS_PATH=results/mac-results_rpp_with_mnt_2048_generic_prompt.csv

# export USING_CHAT_TEMPLATE=true
# export RESULTS_PATH=results/mac-results_rpp_with_mnt_2048.csv

#./scripts/eval-rpp.sh internlm internlm2_5-7b-chat checkpoint-140

#./scripts/eval-rpp.sh Qwen Qwen2-7B-Instruct checkpoint-105

# ./scripts/eval-rpp.sh shenzhi-wang Mistral-7B-v0.3-Chinese-Chat checkpoint-70

# ./scripts/eval-rpp.sh microsoft Phi-3.5-mini-instruct checkpoint-210

export BATCH_SIZE=1
./scripts/eval-rpp.sh shenzhi-wang Llama3.1-8B-Chinese-Chat checkpoint-105

# ./scripts/eval-rpp.sh Qwen Qwen2-7B-Instruct checkpoint-105


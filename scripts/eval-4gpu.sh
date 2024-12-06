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

cd ../rapget-v2; python eval_modules/calc_bert_scores.py; cd -

#pip install torch torchvision torchaudio

# pip install -r requirements.txt

export BATCH_SIZE=1
export LOAD_IN_4BIT=true

# ./scripts/eval-model.sh Qwen/Qwen2-72B-Instruct

# ./scripts/eval-model.sh shenzhi-wang/Llama3.1-70B-Chinese-Chat

# export CHECKPOINTS_PER_EPOCH=4
# ./scripts/eval-epochs.sh Qwen Qwen2-72B-Instruct

# export CHECKPOINTS_PER_EPOCH=1
# ./scripts/eval-epochs.sh shenzhi-wang Llama3.1-70B-Chinese-Chat

export MAX_NEW_TOKENS=2048
export START_REPETITION_PENALTY=1.0
export END_REPETITION_PENALTY=1.1

export USING_CHAT_TEMPLATE=false
export RESULTS_PATH=results/mac-results_rpp_with_mnt_2048_generic_prompt.csv

# export USING_CHAT_TEMPLATE=true
# export RESULTS_PATH=results/mac-results_rpp_with_mnt_2048.csv

#./scripts/eval-rpp.sh shenzhi-wang Llama3.1-70B-Chinese-Chat checkpoint-210

./scripts/eval-rpp.sh Qwen Qwen2-72B-Instruct checkpoint-560

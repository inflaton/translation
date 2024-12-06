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

#pip install -r requirements.txt
#cd ../LLaMA-Factory && pip install -e .[torch,metrics,vllm] && cd -

./scripts/tune-lf-4gpu.sh Qwen Qwen2-72B-Instruct qwen

#./scripts/tune-lf-4gpu.sh shenzhi-wang Llama3.1-70B-Chinese-Chat llama3

export LOAD_IN_4BIT=true

./scripts/eval-epochs.sh Qwen Qwen2-72B-Instruct

./scripts/eval-epochs.sh shenzhi-wang Llama3.1-70B-Chinese-Chat

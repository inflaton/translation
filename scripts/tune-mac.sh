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
# cd ../LLaMA-Factory && pip install -e .[torch,metrics,vllm] && cd -

# ./scripts/tune-lf.sh internlm internlm2_5-7b-chat intern2

# ./scripts/tune-lf.sh Qwen Qwen2-7B-Instruct qwen

# ./scripts/tune-lf.sh shenzhi-wang Mistral-7B-v0.3-Chinese-Chat mistral

# ./scripts/tune-lf.sh shenzhi-wang Llama3.1-8B-Chinese-Chat llama3

./scripts/tune-lf.sh microsoft Phi-3.5-mini-instruct phi

#!/bin/sh

BASEDIR=$(dirname "$0")
cd $BASEDIR/../llama-factory/
echo Current Directory:
pwd

export ORG_NAME=$1
export MODEL_NAME=$2
export CHAT_TEMPLATE=$3
export DATA_PATH=../datasets/mac/mac.tsv
export YAML=config/mac_template.yaml

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

python ../llm_toolkit/setup_lf.py
llamafactory-cli train config/models/$MODEL_NAME.yaml

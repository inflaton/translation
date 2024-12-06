#!/bin/sh

BASEDIR=$(dirname "$0")
cd $BASEDIR/..
echo Current Directory:
pwd

export ORG_NAME=$1
export MODEL=$2

export MODEL_NAME=$ORG_NAME/$MODEL
export ADAPTER_PATH_BASE=llama-factory/saves/$MODEL

echo Evaluating $MODEL_NAME
python llm_toolkit/eval_epochs.py

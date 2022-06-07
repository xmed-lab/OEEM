#!/usr/bin/env bash
CONFIG=$1
GPUS=$2
DIR=$3
PORT=${PORT:-29501}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch --work-dir ${DIR} ${@:4}
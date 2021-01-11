#!/usr/bin/env bash

CONFIG=$1
GPUS=$2

echo $PORT
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}


# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#     $(dirname "$0")/train.py $CONFIG --resume_from=/home/jovyan/vision/mmdetection/work_dirs/mask_rcnn_r50_fpn_sample1e-3_mstrain_1x_lvis_v1/epoch_1.pth $RESUME --launcher pytorch ${@:3}
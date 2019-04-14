#!/bin/bash
set -ex

export CUDA_VISIBLE_DEVICES=2
model="C"

if [ $model = "A" ]; then
    ## model A
    python train.py \
            --gpu_id 0 \
            --net ResNet50 \
            --pretrained 1 \
            --bottleneck 1 \
            --augmentation 0 \
            --weight_init 1 \
            --batch_size 96 \
            --opt_type SGD \
            --lr 0.002 \
            --debug_str model_A
elif [ $model = "B" ]; then
    ## model B
    python train.py \
            --gpu_id 0 \
            --net ResNet50 \
            --pretrained 0 \
            --bottleneck 1 \
            --augmentation 0 \
            --weight_init 1 \
            --batch_size 96 \
            --opt_type SGD \
            --lr 0.002 \
            --debug_str model_B
elif [ $model = "C" ]; then
    ## model C
    python train.py \
            --gpu_id 0 \
            --net MyOwn \
            --bottleneck 1 \
            --augmentation 1 \
            --weight_init 1 \
            --weight 1.5 \
            --batch_size 96 \
            --opt_type SGD \
            --lr 0.002 \
            --debug_str model_C
else
    echo "Unknown model training."
fi
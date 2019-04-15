#!/bin/bash
set -ex

export CUDA_VISIBLE_DEVICES=4,5
model="C"

if [ $model = "A" ]; then
    ## model A
    python train.py \
            --gpu_id 0 \
            --net ResNet50 \
            --pretrained 1 \
            --bottleneck 0 \
            --augmentation 1 \
            --weight_init 1 \
            --batch_size 96 \
            --opt_type SGD \
            --lr 0.006 \
            --debug_str model_A
elif [ $model = "B" ]; then
    ## model B
    python train.py \
            --gpu_id 0 \
            --net ResNet50 \
            --pretrained 0 \
            --bottleneck 0 \
            --augmentation 1 \
            --weight_init 1 \
            --batch_size 96 \
            --opt_type SGD \
            --lr 0.006 \
            --debug_str model_B
elif [ $model = "C" ]; then
    ## model C
    python train.py \
            --gpu_id 0,1 \
            --net MyOwn \
            --bottleneck 0 \
            --augmentation 1 \
            --weight_init 1 \
            --batch_size 32 \
            --opt_type SGD \
            --lr 0.006 \
            --vis 1 \
            --resume_path models/pretrained-0,bottleneck-0,augmentation-1,weight_init-1,opt_type-SGD,momentum-0.9,lr-0.006,batch-32,weight-1.0,debug-model_C_att/best_model.pth \
            --debug_str model_C_att
else
    echo "Unknown model training."
fi
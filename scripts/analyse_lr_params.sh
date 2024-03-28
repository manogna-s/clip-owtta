#!/bin/bash

param_groups=("prompts" "fullv" "fullt" "full" "ln")

for params in ${param_groups[@]};
    do 
    for lr in 0.01 0.001 0.0001 0.00001;
        do 
        echo ${params} ${lr}
        CUDA_VISIBLE_DEVICES=2 nohup python tta_rosita.py --out_dir logs/analysis_params --param_group ${params} --tta_lr ${lr} --dataset ImagenetCOOD --strong_OOD MNIST --model maple --tta_method FtParamsClosedSet
        done
    done
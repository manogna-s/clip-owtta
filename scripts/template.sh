#!/bin/bash
TTA_METHOD=$1
CLASSIFIER_TYPE=$2
PL_THRESH=$3
GPU_ID=$4
ood_detectors=("maxlogit")

for ood_detect in ${ood_detectors[@]};
    do 
        echo "$TTA_METHOD $ood_detect $CLASSIFIER_TYPE"
        CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_ood.py --dataset VisdaOOD --strong_OOD MNIST      --ood_detector ${ood_detect} --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE} --pl_thresh ${PL_THRESH}
        CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_ood.py --dataset VisdaOOD --strong_OOD SVHN       --ood_detector ${ood_detect} --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE} --pl_thresh ${PL_THRESH}
        CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_ood.py --dataset ImagenetROOD --strong_OOD MNIST  --ood_detector ${ood_detect} --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE} --pl_thresh ${PL_THRESH}
        CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_ood.py --dataset ImagenetROOD --strong_OOD SVHN   --ood_detector ${ood_detect} --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE} --pl_thresh ${PL_THRESH}
        CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_ood.py --dataset cifar10OOD --strong_OOD MNIST    --ood_detector ${ood_detect} --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE} --pl_thresh ${PL_THRESH}
        CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_ood.py --dataset cifar10OOD --strong_OOD SVHN     --ood_detector ${ood_detect} --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE} --pl_thresh ${PL_THRESH}
        CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_ood.py --dataset cifar10OOD --strong_OOD cifar100 --ood_detector ${ood_detect} --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE} --pl_thresh ${PL_THRESH}
        CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_ood.py --dataset cifar10OOD --strong_OOD Tiny     --ood_detector ${ood_detect} --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE} --pl_thresh ${PL_THRESH}
        CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_ood.py --dataset cifar100OOD --strong_OOD MNIST   --ood_detector ${ood_detect} --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE} --pl_thresh ${PL_THRESH}
        CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_ood.py --dataset cifar100OOD --strong_OOD SVHN    --ood_detector ${ood_detect} --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE} --pl_thresh ${PL_THRESH}
        CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_ood.py --dataset cifar100OOD --strong_OOD cifar10 --ood_detector ${ood_detect} --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE} --pl_thresh ${PL_THRESH}
        CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_ood.py --dataset cifar100OOD --strong_OOD Tiny    --ood_detector ${ood_detect} --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE} --pl_thresh ${PL_THRESH}
    done

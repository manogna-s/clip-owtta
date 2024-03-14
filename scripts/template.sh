#!/bin/bash
MODEL=$1
TTA_METHOD=$2
CLASSIFIER_TYPE=$3
GPU_ID=$4


# CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_rosita.py --dataset VisdaOOD --strong_OOD MNIST      --model ${MODEL} --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE}
# CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_rosita.py --dataset VisdaOOD --strong_OOD SVHN       --model ${MODEL} --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE}
# CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_rosita.py --dataset ImagenetROOD --strong_OOD MNIST  --model ${MODEL} --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE}
# CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_rosita.py --dataset ImagenetROOD --strong_OOD SVHN   --model ${MODEL} --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE}
CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_rosita.py --dataset cifar10OOD --strong_OOD MNIST    --model ${MODEL} --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE}
CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_rosita.py --dataset cifar10OOD --strong_OOD SVHN     --model ${MODEL} --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE}
CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_rosita.py --dataset cifar10OOD --strong_OOD cifar100 --model ${MODEL} --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE}
CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_rosita.py --dataset cifar10OOD --strong_OOD Tiny     --model ${MODEL} --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE}
CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_rosita.py --dataset cifar100OOD --strong_OOD MNIST   --model ${MODEL} --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE}
CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_rosita.py --dataset cifar100OOD --strong_OOD SVHN    --model ${MODEL} --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE}
CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_rosita.py --dataset cifar100OOD --strong_OOD cifar10 --model ${MODEL} --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE}
CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_rosita.py --dataset cifar100OOD --strong_OOD Tiny    --model ${MODEL} --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE}

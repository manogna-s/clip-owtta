#!/bin/bash
TTA_METHOD='rosita_knn'
CLASSIFIER_TYPE='txt'
GPU_ID=$1
# ood_detectors=("maxlogit")

for K_P in 3 4 5;
    do
    for K_N in 3 5 7 10;
        do 
            echo "$K_P $K_N"

            # CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_ood_maple.py --dataset VisdaOOD --strong_OOD MNIST      --k_p ${K_P} --k_n ${K_N} --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE}
            # CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_ood_maple.py --dataset VisdaOOD --strong_OOD SVHN       --k_p ${K_P} --k_n ${K_N} --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE}
            # CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_ood_maple.py --dataset ImagenetROOD --strong_OOD MNIST  --k_p ${K_P} --k_n ${K_N} --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE}
            # CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_ood_maple.py --dataset ImagenetROOD --strong_OOD SVHN   --k_p ${K_P} --k_n ${K_N} --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE}
            CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_ood_maple.py --dataset cifar10OOD --strong_OOD MNIST    --k_p ${K_P} --k_n ${K_N} --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE}
            CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_ood_maple.py --dataset cifar10OOD --strong_OOD SVHN     --k_p ${K_P} --k_n ${K_N} --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE}
            CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_ood_maple.py --dataset cifar10OOD --strong_OOD cifar100 --k_p ${K_P} --k_n ${K_N} --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE}
            CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_ood_maple.py --dataset cifar10OOD --strong_OOD Tiny     --k_p ${K_P} --k_n ${K_N} --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE}
            CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_ood_maple.py --dataset cifar100OOD --strong_OOD MNIST   --k_p ${K_P} --k_n ${K_N} --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE}
            CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_ood_maple.py --dataset cifar100OOD --strong_OOD SVHN    --k_p ${K_P} --k_n ${K_N} --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE}
            CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_ood_maple.py --dataset cifar100OOD --strong_OOD cifar10 --k_p ${K_P} --k_n ${K_N} --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE}
            CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_ood_maple.py --dataset cifar100OOD --strong_OOD Tiny    --k_p ${K_P} --k_n ${K_N} --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE}
        done
    done
#!/bin/bash
TTA_METHOD='rosita_loss'
CLASSIFIER_TYPE='txt'
GPU_ID=$1


# for LOSS_PL in 0 1;
#     do
#     for LOSS_SIMCLR in 0;
#         do 
#             echo "$LOSS_PL $LOSS_SIMCLR"

#             # CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_ood.py --dataset VisdaOOD --strong_OOD MNIST      --loss_pl ${LOSS_PL} --loss_simclr ${LOSS_SIMCLR} --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE}
#             # CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_ood.py --dataset VisdaOOD --strong_OOD SVHN       --loss_pl ${LOSS_PL} --loss_simclr ${LOSS_SIMCLR} --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE}
#             CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_ood.py --dataset ImagenetROOD --strong_OOD MNIST  --loss_pl ${LOSS_PL} --loss_simclr ${LOSS_SIMCLR} --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE}
#             CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_ood.py --dataset ImagenetROOD --strong_OOD SVHN   --loss_pl ${LOSS_PL} --loss_simclr ${LOSS_SIMCLR} --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE}
#             CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_ood.py --dataset cifar10OOD --strong_OOD MNIST    --loss_pl ${LOSS_PL} --loss_simclr ${LOSS_SIMCLR} --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE}
#             CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_ood.py --dataset cifar10OOD --strong_OOD SVHN     --loss_pl ${LOSS_PL} --loss_simclr ${LOSS_SIMCLR} --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE}
#             CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_ood.py --dataset cifar10OOD --strong_OOD cifar100 --loss_pl ${LOSS_PL} --loss_simclr ${LOSS_SIMCLR} --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE}
#             CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_ood.py --dataset cifar10OOD --strong_OOD Tiny     --loss_pl ${LOSS_PL} --loss_simclr ${LOSS_SIMCLR} --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE}
#             CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_ood.py --dataset cifar100OOD --strong_OOD MNIST   --loss_pl ${LOSS_PL} --loss_simclr ${LOSS_SIMCLR} --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE}
#             CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_ood.py --dataset cifar100OOD --strong_OOD SVHN    --loss_pl ${LOSS_PL} --loss_simclr ${LOSS_SIMCLR} --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE}
#             CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_ood.py --dataset cifar100OOD --strong_OOD cifar10 --loss_pl ${LOSS_PL} --loss_simclr ${LOSS_SIMCLR} --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE}
#             CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_ood.py --dataset cifar100OOD --strong_OOD Tiny    --loss_pl ${LOSS_PL} --loss_simclr ${LOSS_SIMCLR} --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE}
#         done
#     done

for LOSS_PL in 0 1;
    do
    for LOSS_SIMCLR in 0;
        do 
            echo "$LOSS_PL $LOSS_SIMCLR"

            CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_ood_maple.py --dataset VisdaOOD --strong_OOD MNIST      --loss_pl ${LOSS_PL} --loss_simclr ${LOSS_SIMCLR} --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE}
            CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_ood_maple.py --dataset VisdaOOD --strong_OOD SVHN       --loss_pl ${LOSS_PL} --loss_simclr ${LOSS_SIMCLR} --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE}
            CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_ood_maple.py --dataset ImagenetROOD --strong_OOD MNIST  --loss_pl ${LOSS_PL} --loss_simclr ${LOSS_SIMCLR} --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE}
            CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_ood_maple.py --dataset ImagenetROOD --strong_OOD SVHN   --loss_pl ${LOSS_PL} --loss_simclr ${LOSS_SIMCLR} --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE}
            CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_ood_maple.py --dataset cifar10OOD --strong_OOD MNIST    --loss_pl ${LOSS_PL} --loss_simclr ${LOSS_SIMCLR} --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE}
            CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_ood_maple.py --dataset cifar10OOD --strong_OOD SVHN     --loss_pl ${LOSS_PL} --loss_simclr ${LOSS_SIMCLR} --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE}
            CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_ood_maple.py --dataset cifar10OOD --strong_OOD cifar100 --loss_pl ${LOSS_PL} --loss_simclr ${LOSS_SIMCLR} --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE}
            CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_ood_maple.py --dataset cifar10OOD --strong_OOD Tiny     --loss_pl ${LOSS_PL} --loss_simclr ${LOSS_SIMCLR} --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE}
            CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_ood_maple.py --dataset cifar100OOD --strong_OOD MNIST   --loss_pl ${LOSS_PL} --loss_simclr ${LOSS_SIMCLR} --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE}
            CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_ood_maple.py --dataset cifar100OOD --strong_OOD SVHN    --loss_pl ${LOSS_PL} --loss_simclr ${LOSS_SIMCLR} --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE}
            CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_ood_maple.py --dataset cifar100OOD --strong_OOD cifar10 --loss_pl ${LOSS_PL} --loss_simclr ${LOSS_SIMCLR} --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE}
            CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_ood_maple.py --dataset cifar100OOD --strong_OOD Tiny    --loss_pl ${LOSS_PL} --loss_simclr ${LOSS_SIMCLR} --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE}
        done
    done
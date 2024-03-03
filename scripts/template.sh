TTA_METHOD=$1
CLASSIFIER_TYPE=$2
PL_THRESH=$3
GPU_ID=$4

# CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_ood.py --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE} --pl_thresh ${PL_THRESH} --dataset cifar10OOD --strong_OOD MNIST --ood_detector msp
# CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_ood.py --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE} --pl_thresh ${PL_THRESH} --dataset cifar10OOD --strong_OOD MNIST --ood_detector maxlogit
# CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_ood.py --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE} --pl_thresh ${PL_THRESH} --dataset cifar10OOD --strong_OOD SVHN --ood_detector msp
# CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_ood.py --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE} --pl_thresh ${PL_THRESH} --dataset cifar10OOD --strong_OOD SVHN --ood_detector maxlogit
# CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_ood.py --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE} --pl_thresh ${PL_THRESH} --dataset cifar10OOD --strong_OOD cifar100 --ood_detector msp
# CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_ood.py --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE} --pl_thresh ${PL_THRESH} --dataset cifar10OOD --strong_OOD cifar100 --ood_detector maxlogit
# CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_ood.py --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE} --pl_thresh ${PL_THRESH} --dataset cifar10OOD --strong_OOD Tiny --ood_detector msp
# CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_ood.py --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE} --pl_thresh ${PL_THRESH} --dataset cifar10OOD --strong_OOD Tiny --ood_detector maxlogit
# CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_ood.py --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE} --pl_thresh ${PL_THRESH} --dataset cifar100OOD --strong_OOD MNIST --ood_detector msp
# CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_ood.py --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE} --pl_thresh ${PL_THRESH} --dataset cifar100OOD --strong_OOD MNIST --ood_detector maxlogit
# CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_ood.py --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE} --pl_thresh ${PL_THRESH} --dataset cifar100OOD --strong_OOD SVHN --ood_detector msp
# CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_ood.py --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE} --pl_thresh ${PL_THRESH} --dataset cifar100OOD --strong_OOD SVHN --ood_detector maxlogit
# CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_ood.py --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE} --pl_thresh ${PL_THRESH} --dataset cifar100OOD --strong_OOD cifar10 --ood_detector msp
# CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_ood.py --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE} --pl_thresh ${PL_THRESH} --dataset cifar100OOD --strong_OOD cifar10 --ood_detector maxlogit
# CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_ood.py --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE} --pl_thresh ${PL_THRESH} --dataset cifar100OOD --strong_OOD Tiny --ood_detector msp
# CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_ood.py --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE} --pl_thresh ${PL_THRESH} --dataset cifar100OOD --strong_OOD Tiny --ood_detector maxlogit


CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_ood.py --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE} --pl_thresh ${PL_THRESH} --dataset ImagenetROOD --strong_OOD MNIST --ood_detector msp
CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_ood.py --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE} --pl_thresh ${PL_THRESH} --dataset ImagenetROOD --strong_OOD MNIST --ood_detector maxlogit
CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_ood.py --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE} --pl_thresh ${PL_THRESH} --dataset ImagenetROOD --strong_OOD SVHN --ood_detector msp
CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_ood.py --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE} --pl_thresh ${PL_THRESH} --dataset ImagenetROOD --strong_OOD SVHN --ood_detector maxlogit
# CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_ood.py --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE} --pl_thresh ${PL_THRESH} --dataset ImagenetROOD --strong_OOD cifar10 --ood_detector msp
# CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_ood.py --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE} --pl_thresh ${PL_THRESH} --dataset ImagenetROOD --strong_OOD cifar10 --ood_detector maxlogit
# CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_ood.py --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE} --pl_thresh ${PL_THRESH} --dataset ImagenetROOD --strong_OOD Tiny --ood_detector msp
# CUDA_VISIBLE_DEVICES=${GPU_ID} python tta_ood.py --tta_method ${TTA_METHOD} --classifier_type ${CLASSIFIER_TYPE} --pl_thresh ${PL_THRESH} --dataset ImagenetROOD --strong_OOD Tiny --ood_detector maxlogit
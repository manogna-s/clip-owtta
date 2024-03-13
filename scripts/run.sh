MODEL=$1
TTA_METHOD=$2
CLASSIFIER_TYPE=$3
GPU_ID=$4

mkdir -p logs/baselines/${MODEL}
mkdir -p logs/baselines/out_files/${MODEL}

nohup bash scripts/template.sh ${MODEL} ${TTA_METHOD} ${CLASSIFIER_TYPE} ${GPU_ID} >logs/baselines/out_files/${MODEL}/${TTA_METHOD}.out&
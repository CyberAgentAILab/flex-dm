#!/usr/bin/env bash
export PYTHONPATH="src/mfp"

DATASET=${1:-"crello"}
NOW=$(date '+%Y%m%d%H%M%S')

JOB_NAME="${DATASET}_${NOW}"
JOB_NAME=$(tr - _ <<< "${JOB_NAME}")

DATA_DIR="data/${DATASET}"
JOB_DIR="tmp/jobs/${DATASET}/${NOW}"

echo "DATA_DIR=${DATA_DIR}"
echo "JOB_DIR=${JOB_DIR}"

python -m mfp \
    --dataset_name ${DATASET} \
    --data_dir ${DATA_DIR} \
    --job-dir ${JOB_DIR} \
    ${@:2}

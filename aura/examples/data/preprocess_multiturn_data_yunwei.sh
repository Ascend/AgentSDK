#!/bin/bash
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH=$SCRIPT_DIR/../..:$PYTHONPATH
PROJECT_PATH=$SCRIPT_DIR/../..

default_config="multiturn_grpo_qwq_32b"

if [ -z "$1" ]; then
    config=$default_config
else
    config=$1
fi

python "$PROJECT_PATH"/cli/preprocess_data.py $config 2>&1 | tee logs/convert_data_logs_${JOB_NAME}_${timestamp}.log

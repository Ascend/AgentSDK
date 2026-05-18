#!/bin/bash
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH=$SCRIPT_DIR/../..:$PYTHONPATH
PROJECT_PATH=$SCRIPT_DIR/../..

default_config="multiturn_grpo_qwen3_235b_dtn_code"

if [ -z "$1" ]; then
    config=$default_config
else
    config=$1
fi

python "$PROJECT_PATH"/cli/preprocess_data.py $config 2>&1

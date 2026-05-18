#!/bin/bash
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
export PYTHONPATH=$SCRIPT_DIR/../..:$PYTHONPATH
PROJECT_PATH=$SCRIPT_DIR/../..

timestamp=$(date +"%Y%m%d_%H%M%S")

python "$PROJECT_PATH"/scripts/weights/compare_magetron_checkpoints.py \
    --source-checkpoint /models/g00898995/model/Qwen3-235B-thinking-2507-tp2pp16ep4/iter_0000001 \
    --target-checkpoint /models/g00898995/model/Qwen3-235B-thinking-2507-tp2pp16ep4/iter_0000001 \
    --tensor-parallel-size 2 \
    --pipeline-parallel-size 16 \
    --expert-parallel-size 4 \
    2>&1 | tee logs/compare_magetron_checkpoints_${timestamp}.log

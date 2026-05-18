export CUDA_DEVICE_MAX_CONNECTIONS=1

# Modify ascend-toolkit path
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Set parallel strategy
python cli/convert_ckpt.py \
    --use-mcore-models \
    --model-type GPT \
    --model-type-hf llama2 \
    --load-model-type mg \
    --save-model-type hf \
    --target-tensor-parallel-size 4 \
    --target-pipeline-parallel-size 1 \
    --add-qkv-bias \
    --orm \
    --load-dir ./ckpt/ \
    --save-dir ./ckpt/qwen25-7B

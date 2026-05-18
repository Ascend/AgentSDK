export CUDA_DEVICE_MAX_CONNECTIONS=1

python cli/convert_ckpt_qwen3_moe_mcore2hf.py \
    --source-tensor-parallel-size 2 \
    --source-pipeline-parallel-size 16 \
    --source-expert-parallel-size 4 \
    --noop-layers "5,95" \
    --load-dir ./qwen3_235b_t2e4p16_noop5_95/ \
    --save-dir ./saved_235b_hf/ \
    --num-layers 96 \
    --moe-grouped-gemm \
    --moe-tp-extend-ep \
    --hidden-size 4096 \
    --num-attention-heads 64

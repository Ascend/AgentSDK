export CUDA_DEVICE_MAX_CONNECTIONS=1

python cli/convert_ckpt_qwen3_moe_hf2mcore.py \
    --target-tensor-parallel-size 2 \
    --target-pipeline-parallel-size 16 \
    --target-expert-parallel-size 4 \
    --noop-layers "5,95" \
    --load-dir ./Qwen3-235B/ \
    --save-dir ./qwen3_235b_t2e4p16_noop5_95/ \
    --num-layers 96 \
    --moe-grouped-gemm \
    --moe-tp-extend-ep \
    --hidden-size 4096 \
    --num-attention-heads 64

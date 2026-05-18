export CUDA_DEVICE_MAX_CONNECTIONS=1

python cli/convert_ckpt_qwen3_moe_hf2mcore.py \
    --target-tensor-parallel-size 4 \
    --target-pipeline-parallel-size 1 \
    --target-expert-parallel-size 2 \
    --load-dir ./Qwen3-30B-A3B-Instruct-2507/ \
    --save-dir ./mcore_qwen3_30b_2507_t4e2p1\
    --num-layers 48 \
    --moe-grouped-gemm \
    --moe-tp-extend-ep \
    --hidden-size 2048 \
    --num-attention-heads 32

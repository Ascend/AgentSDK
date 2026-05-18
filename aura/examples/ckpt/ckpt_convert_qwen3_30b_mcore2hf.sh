export CUDA_DEVICE_MAX_CONNECTIONS=1

python cli/convert_ckpt_qwen3_moe_mcore2hf.py \
    --source-tensor-parallel-size 4 \
    --source-pipeline-parallel-size 1 \
    --source-expert-parallel-size 2 \
    --load-dir ./mcore_qwen3_30b_2507_t4e2p1/ \
    --save-dir ./saved_30b_hf/ \
    --num-layers 48 \
    --moe-grouped-gemm \
    --moe-tp-extend-ep \
    --hidden-size 2048 \
    --num-attention-heads 32

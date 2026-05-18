export CUDA_DEVICE_MAX_CONNECTIONS=1

# Modify ascend-toolkit path
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# Set the required weight conversion parameters
python cli/convert_ckpt.py \
       --use-mcore-models \
       --model-type GPT \
       --load-model-type hf \
       --save-model-type mg \
       --target-tensor-parallel-size 4 \
       --target-pipeline-parallel-size 1 \
       --add-qkv-bias \
       --load-dir ./ckpt/qwen25-7B \
       --save-dir ./ckpt/ \
       --tokenizer-model ./ckpt/qwen25-7B/tokenizer.json \
       --model-type-hf llama2 \
       --params-dtype bf16 \
       --ckpt-cfg-path $PWD/configs/checkpoint/model_cfg.json

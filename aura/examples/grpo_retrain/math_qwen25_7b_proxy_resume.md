# v5.0.0断点续训部署指导——以Math、Qwen2.5—7B、proxy模式（vllm外接+pd分离）场景为例

## 环境部署介绍

### 0 代码准备

```shell
# 下载aura代码后
# third_party三方库代码下载
bash download_third_party.sh
```

### 1 镜像准备

平台 镜像名称：xxxx 镜像版本 ：xxx //TODO

### 2 启动训练任务

//TODO

#### 2.1 参数配置

##### 2.1.1 : Qwen2.5-7B模型路径

```shell
huggingface原始权重: /path/to/models/Qwen2.5-7B-Instruct
megatron格式tp4pp2权重: /path/to/original_ckpts/Qwen2.5-7B-Instruct-mcore-tp4pp2
tokenizer路径: /path/to/models/Qwen2.5-7B-Instruct
```

##### 2.1.2 : Math数据集路径

```shell
数据集路径: /path/to/datasets/deepscaler_math_01/rl
```

##### 2.1.3 : PD分离相关配置及脚本（2机16卡推理1P1D + 1机8卡训练 + 混合批次优化开启）

```shell
续训任务监控启动脚本: examples/grpo_retrain/start_checkpoint_resume_vllm_proxy.sh
外挂pd分离运行脚本: start_roma_vllm_proxy_pd_resume.sh
# 根据初始配置自动生成对应的resume配置文件
初始运行配置: configs/direct_p1d1_qwen25_7b_train_one_step_off.yaml
msrl模版配置（勿改）: configs/msrl_conf/base_integrated_grpo_trainer_template_qwen25_7b.yaml
```

##### 2.1.4 : 任务启动命令

```shell
cd /path/to/aura_retrain/AgenticRL;
bash examples/grpo_retrain/start_checkpoint_resume_vllm_proxy.sh | tee /models/xxx/retrain/xxx.log
```

#### 2.2 续训相关参数修改

##### a) 修改续训脚本配置参数

打开5.0续训监控启动脚本examples/grpo_retrain/start_checkpoint_resume_vllm_proxy.sh:

```bash
#!/bin/bash
MAX_RETRIES=100 # 自定义：retry次数
RETRY_COUNT=1 # 默认1开始，勿修改
DEFAULT_START_SH="start_roma_vllm_proxy_pd_resume.sh" # 须修改: 5.0启动脚本(vllm外挂)
DEFAULT_YAML="direct_p1d1_qwen25_7b_train_one_step_off" # 须修改: 主要配置文件，非msrl_conf，覆盖运行脚本参数文件
DEFAULT_LOG_PATH="/models/retrain/logs" # 须修改: 续训监控日志路径
DEFAULT_CLEAR_CKPT="0" # 是否开启续训清理非最新saveckpt 0:关闭 1:开启
DEFAULT_CLEAR_ALL_CKPT="0" # 是否开启首次清理所有saveckpt 0:关闭 1:开启
MASTER_TRAIN_INDEX=2 # 须修改: 训练主节点index
```

##### b) 修改Train/Rollout配置参数

在运行配置direct_p1d1_qwen25_7b_train_one_step_off_resume中，关注"须修改"字段

```shell
# =============================================================
# Train 预定义任务：
# 1. 预定义任务包括采用RL框架类型、参数等等，初始化时占用资源，触发后开始训练（支持开始、暂停、取消）
# 2. SERVE 模式下，预定义任务列表默认为空; DIRECT 模式下，初始化任务等待触发训练（触发：direct_conf.entrypoints）
# =============================================================

# 根据rl框架和模型选择不同的模板，模板文件在msrl_conf/verl_conf路径下
defaults:
  - msrl_conf: base_integrated_grpo_trainer_template_qwen25_7b  # 参数模板, 该文件内容不用修改
  - verl_conf: null
  - _self_

# 覆盖MSRL模板中的参数（部分省略）
msrl_conf:
  megatron_training:
    global_batch_size: 8
    train_iters: 100
    save_interval: 2 # 须修改：开启混合批次时，最好是2的倍数
    tokenizer_name_or_path: /path/to/models/Qwen2.5-7B-Instruct
    data_path: /path/to/datasets/deepscaler_math_01/rl
    dataset_additional_keys: ['labels']
  actor_config:
    #ori_train_model_path:/path/to/original_ckpts/Qwen2.5-7B-Instruct-mcore-tp4pp2
    load: /path/to/original_ckpts/Qwen2.5-7B-Instruct-mcore-tp4pp2 # 须修改
    save: /models/retrain/save_ckpts/qwen25_7b # 须修改：注意save路径不放重要数据，续训程序DEFAULT_CLEAR_ALL_CKPT开启后会清空目录
    finetune: true # 续训开始前为true
    tensor_model_parallel_size: 4
    pipeline_model_parallel_size: 2
  generate_config:
    # onestepoff：混合批次配置
    init_num_group_batches: 2
    hybrid_batch_num: 2
    enable_version_control: true
    weight_save_dir: /models/retrain/update_weight/qwen25_7b #
    ckpt_delta: 1
```

##### c) 修改推理配置参数

打开start_roma_vllm_proxy_pd_resume.sh文件:

```bash
###################################################################################
# 相关待修改配置如下:（其他省略）
MASTER_TRAIN_INDEX=2   # 须修改：train/rollout集群master主节点idx
CONFIG_NAME=direct_p1d1_qwen25_7b_train_one_step_off_resume  # 配置文件：会被外层start_checkpoint_resume_vllm_proxy.sh的覆盖
export MODEL_PATH=/path/to/models/Qwen2.5-7B-Instruct #须修改：推理hf模型权重路径
export SERVED_MODEL_NAME=Qwen2.5-7B-Instruct #须修改
export PREFILL_INSTANCE_COUNT=1    # 须修改：PD分离时表示P实例的个数
# set DECODE_INSTANCE_COUNT to 0 if USE_PD is 0
export DECODE_INSTANCE_COUNT=1     # 须修改：PD分离时表示D实例的个数
export PREFILL_TENSOR_PARALLEL_SIZE=4 # 须修改：如以8卡为一个实例，推理tp4dp2
export PREFILL_DATA_PARALLEL_SIZE=2 # 须修改
export DECODE_TENSOR_PARALLEL_SIZE=4 # 须修改
export DECODE_DATA_PARALLEL_SIZE=2 # 须修改
export USE_PD=${USE_PD:-1}     # 须修改：0: PD混部， 1: PD分离
###################################################################################
```

P实例和D实例及Train和Rollout的节点分布如下：(假设部署2P6D共384卡的训练推理任务)

```text
|----|----|----|----|----|----|----|----|----------------|
| P1   P2   D1   D2   D3   D4   D5   D6    Train&Rollout
|
|--->VC_TASK_INDEX=0
```

##### 注意vllm的max-model-len配置须根据模型修改

vllm的配置在script/deploy_vllm/vllm_serve.sh中：
例如, 7b模型可将 --max-model-len 66536 \ 修改为 --max-model-len 16384 \

```bash
start_vllm_serve_separate() {
  KV_TRANSFER_CONFIG=$(get_llmdatadist_kv_transfer_conf)
  if [ $KV_BACKEND = "mooncake" ]; then
    KV_TRANSFER_CONFIG=$(get_mooncake_kv_transfer_conf $TENSOR_PARALLEL_SIZE $DATA_PARALLEL_SIZE $ENGINE_ID)
  fi
  if [ "$ROLE" = "prefill" ]; then
      KV_TRANSFER_CONFIG=$(echo $KV_TRANSFER_CONFIG | sed "s/kv_consumer/kv_producer/g")
      export TASK_QUEUE_ENABLE="2"
      vllm serve "$MODEL_PATH" \
          --served-model-name "$SERVED_MODEL_NAME" \
          --host "$HOST" \
          --port "$PORT" \
          $HEADLESS_FLAG \
          --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
          --data-parallel-size "$DATA_PARALLEL_SIZE" \
          --data-parallel-size-local "$DATA_PARALLEL_SIZE_LOCAL" \
          $EXPERT_PARALLEL_ARG \
          --data-parallel-start-rank "$DP_START_RANK" \
          --data-parallel-address "$MASTER_ADDR" \
          --data-parallel-rpc-port "$DP_RPC_PORT" \
          --max-model-len 16384 \ # 须修改
          --max-num-batched-tokens 8192 \
          --gpu-memory-utilization 0.7 \
          --trust-remote-code \
          --enforce-eager \
          --enable-chunked-prefill \
          --max-num-seqs 6 \
          --enable-prefix-caching \
          --worker_extension_cls "aura.runner.infer_adapter.vllm.extension.custom_worker_extensions.CustomWorkerExtensions" \
          --additional-config '{"ascend_scheduler_config":{"enabled":true,"enable_chunked_prefill":true}}' \
          --kv-transfer-config "$KV_TRANSFER_CONFIG"
  else
      vllm serve "$MODEL_PATH" \
          --served-model-name "$SERVED_MODEL_NAME" \
          --host "$HOST" \
          --port "$PORT" \
          $HEADLESS_FLAG \
          --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
          --data-parallel-size "$DATA_PARALLEL_SIZE" \
          --data-parallel-size-local "$DATA_PARALLEL_SIZE_LOCAL" \
          $EXPERT_PARALLEL_ARG \
          --data-parallel-start-rank "$DP_START_RANK" \
          --data-parallel-address "$MASTER_ADDR" \
          --data-parallel-rpc-port "$DP_RPC_PORT" \
          --max-model-len 16384 \ # 须修改
          --max-num-batched-tokens 128 \
          --gpu-memory-utilization 0.85 \
          --trust-remote-code \
          --enable-chunked-prefill \
          --max-num-seqs 24 \
          --enable-prefix-caching \
          --worker_extension_cls "aura.runner.infer_adapter.vllm.extension.custom_worker_extensions.CustomWorkerExtensions" \
          --additional-config '{"ascend_scheduler_config":{"enabled":true,"enable_chunked_prefill":true}}' \
          --compilation_config '{"cudagraph_capture_sizes":[4,8,12,16,24],"cudagraph_mode":"FULL_DECODE_ONLY"}' \
          --kv-transfer-config "$KV_TRANSFER_CONFIG"
  fi
}
```

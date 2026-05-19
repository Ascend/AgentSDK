# 脚本修改规范和使用方法

## 训练脚本修改原则

```text
    原则上训练脚本稳定之后不同的训练任务不需要改动脚本
    只需要改动configs/base.conf、configs/hosts.conf、configs/train/***.yaml、configs/infer/***.yaml即可
    如果训练脚本有不适配的地方，需要按照通用处理逻辑去改，当做需求去开发，避免大家散弹式到处修改
```

## 训练脚本调用方法

```text
    启动基于msrl+vllm的训练：
    sh start_rl_with_msrl_vllm.sh
    启动基于verl+vllm的训练：
    sh start_rl_with_verl_vllm.sh
```

## 断点续训调用方法

```text
    断点续训的配置文件修改:
    configs/bash.conf

    启动基于checkpoint的断点续训：
    sh start_rl_with_resume.sh

    启动基于verl的checkpoint的断点续训：
    sh start_rl_with_resume_verl.sh
```

# v5.0.0断点续训部署指导

# —— 以single-turn、Qwen3.5-27B、verl(fsdp) + vllm、训推分离异步场景为例

## 0 镜像准备

### 0.1 支持msrl的镜像

```text
支持Qwen3：
    agentic-rl-a2-vllm-011:4.0.1或4.0.3
支持DeepseekV3.2：
    agentic-rl-dpsk32-a2:5.0.1
```

### 0.2 支持verl的镜像

```text
最新支持Qwen3.5：
    aura-verl-fsdp-sp-a2:5.0.3
```

## 1 前置准备

### 1.1 运行配置

#### configs/base.conf

```shell
# 工作模式：hybrid 共卡模式 | one_step_off 全异步分离模式
work_mode=one_step_off

# 共卡和分离模式均需要配置训练yaml文件
train_config_name=verl_train_async_t8_qwen35_27B_single_turn

# 分离模式需要单独配置推理yaml文件, 共卡模式该配置不生效
infer_config_name=vllm_infer_i8_qwen35_27b

# [resume]
# 需要监控的启动脚本：start_rl_with_msrl_vllm.sh | start_rl_with_verl_vllm.sh
monitor_cmd=start_rl_with_verl_vllm.sh

# 断点续训重试次数, 默认100次
max_retries=100

# 第一次启动是否需要清空ckpt文件夹: 0 不清理; 1 需要清理
clean_old_ckpt=0 # 首次训练前，建议清空default_local_dir下不需要的ckpt
```

### 1.2 推理配置

#### configs/infer/vllm_infer_i8_qwen35_27b.yaml

```shell
# 修改模型权重路径
infer_mode_path: /path/to/aura/person/Qwen35_27B
```

#### 推理如需扩充实例，配置如下

```shell
# 0为pd混部模式 1为pd分离模式
pd_mode: 0

# prefill实例数量
prefill_instance_count: 1

# decode实例数量, PD混部时配置为0
decode_instance_count: 0

# prefill/decode tp size
tensor_parallel_size: 4

# prefill/decode dp size
data_parallel_size: 2  # 可通过修改dp，扩充实例数
```

**推理所需总卡数 = (prefill_instance_count+decode_instance_count) * (tensor_parallel_size * data_parallel_size)**

### 1.3 训练配置

#### configs/train/verl_train_async_t8_qwen35_27B_single_turn.yaml

```shell
# 关键参数修改
nnodes: 1 # 训练需要的总机数
n_gpus_per_node: 8 # 每台机子的总卡数 910B=8 | 910C=16

!!!注意：配置中所有路径不要在weight_save_dir目录下，过程中可能会被清空

--- 分离场景下训推权重转换路径 ---
## 须是共享盘路径
weight_save_dir: ${hydra:runtime.cwd}/weights/

--- 模型原始权重路径 ----
## 不能在weight_save_dir目录下 | 路径必须包含模型名称 | 不能以"/"结尾
path: /path/to/aura/person/Qwen35_27B

---  以save_freq间隔保存的ckpt路径 ---
## 不能在weight_save_dir目录下 | 须是真实路径，续训脚本不识别引用参数${hydra:runtime.cwd}
default_local_dir: ./outputs/save_ckpt/

--- 数据集路径 ---
## 不能在weight_save_dir目录下
train_data_path: /path/to/aura/person/dtn_code_data/rl

--- Aura框架AEE生成的轨迹数据保存路径 ---
## 不能在weight_save_dir目录下
trajectory_save_dir: ${hydra:runtime.cwd}/outputs

--- verl自生成的轨迹数据保存路径 ---
## 不能在weight_save_dir目录下
rollout_data_dir: ${hydra:runtime.cwd}/outputs/rollout_data_dir/
```

#### fsdp减少显存占用配置

```shell
# 部分关键参数修改 TODO
    actor:
      strategy: fsdp2
      ulysses_sequence_parallel_size: 8
      fsdp_config:
        strategy: fsdp2
        param_offload: true
        optimizer_offload: true
        model_dtype: bfloat16
        fsdp_size: -1
        ulysses_sequence_parallel_size: ${verl_conf.actor_rollout_ref.actor.ulysses_sequence_parallel_size}
    ref:
      strategy: fsdp2
      log_prob_micro_batch_size_per_gpu: 1
      log_prob_max_token_len_per_gpu: 16384
      ulysses_sequence_parallel_size: ${verl_conf.actor_rollout_ref.actor.ulysses_sequence_parallel_size}
      fsdp_config:
        model_dtype: bfloat16
        param_offload: true
        ulysses_sequence_parallel_size: ${verl_conf.actor_rollout_ref.actor.ulysses_sequence_parallel_size}
```

## 2 手动断点续训

### 2.1 权重转换

```shell
bash verl_merge.sh $已保存的权重路径(例如：/xxx/global_step_5) $自定义转换后的权重路径(路径带模型名称：/xxx/resume/qwenxxx_5)
```

### 2.2 推理配置修改

```shell
infer_mode_path: /xxx/resume/qwenxxx_5 # 修改为自定义的转换权重路径
```

### 2.3 训练配置修改

```shell
path: /xxx/resume/qwenxxx_5 # 修改为自定义的转换权重路径
```

### 注意：由于手动配置断点续训，global_step仍从1开始，如果default_local_dir不改变，会覆盖已保存的ckpt

### 任务启动命令

```shell
source /root/.bashrc;
cd /path/to/aura;
export MASTER_TRAIN_INDEX=1;
bash scripts/start_rl_with_verl_vllm.sh;
```

## 3 自动断点续训（需合入自动续训功能代码）

### 会自动转换权重及参数替换，可续训任务启动命令如下

```shell
source /root/.bashrc;
cd /path/to/aura;
export MASTER_TRAIN_INDEX=1;
bash scripts/start_rl_with_resume_verl.sh;
```

### 如首次启动非原始权重，可直接修改配置实现续训，无需手动转换权重

#### 训练配置修改

```shell
trainer:
    resume_mode: resume_path # disable / auto (load from default_local_dir) / resume_path (load from resume_from_path)
    resume_from_path: /xxx/global_step_5 # 修改为已保存的待续训权重路径(例如：/xxx/global_step_5)
```

#### 续训相关代码目录结构

```shell
AgenticRL
   - agents/
   - aura/
   - cli/
   - configs/
      - base.conf  # 运行配置
      - infer/  # 推理配置目录
      - train/  # 训练配置目录
   - dockers/
   - examples/
   - logs/  # 运行推理及训练日志目录
   - outputs/  # 轨迹数据保存目录
   - resume/  # 续训相关目录
      - resume_hf_path/  # 续训verl转换hf权重目录
      - status/  # 各节点续训状态日志目录
      - save_ckpt/  # 训练权重保存路径
   - scripts/  # 入口：续训启动脚本及readme目录
   - tensorboard_log/
   - tests/
   - third_party/
   - traj_proxy/
   - weights/  # 训推分离权重转换目录
````

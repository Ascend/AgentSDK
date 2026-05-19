# Search_R1

Search_R1 是一个具有 **搜索（Search）能力**的 Agent，支持端到端训练、离线/在线评测，并可扩展至多任务、多工具场景。

---

## 1. 项目结构

### 1.1 Search_R1 Agent 目录结构

Search_R1 Agent 实现主要参考[Search_R1 GitHub Repository](https://github.com/PeterGriffinJin/Search-R1/tree/598e61bd1d36895726d28a8d06b3a15bed19f5d3)：

```text
search_r1_agent/
    ├── environment/ # 环境
    ├── eval/        # 评测
    ├── parser/      # 工具解析
    ├── prompt/
    ├── reward/      # reward计算
    ├── process_data.py     # 数据处理
    ├── retrieval_launch.sh     # 工具启动脚本
    └── retrieval_server.py    # 工具启动函数
    └── search_r1_agent.py      # Search_R1 agent文件
```

---

## 2. 安装

### 2.1 工具环境

```bash
conda create -n retriever python=3.10
conda activate retriever

# we recommend installing torch with conda for faiss-gpu
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers datasets pyserini

## install the gpu version faiss to guarantee efficient RL rollout
conda install -c pytorch -c nvidia faiss-gpu=1.8.0

## API function
pip install uvicorn fastapi
```

### 2.2 训练环境

```bash
参考整体框架运行环境
```

---

## 3. 训练

### 3.1 数据准备

1. **工具数据集**
   启动工具所需的数据集可参考
   [Search_R1 GitHub Repository - Quick Start](https://github.com/PeterGriffinJin/Search-R1/blob/598e61bd1d36895726d28a8d06b3a15bed19f5d3/README.md)。

2. **训练数据**

   下载 [nq_hotpot 数据集](https://huggingface.co/datasets/PeterJinGo/nq_hotpotqa_train)。

   ```bash
   cd agents/search_r1_agent

   python process_data.py ./nq_hotpotqa/train.parquet \
       --columns golden_answers question \
       --prefix-column question \
       --prefix "Answer the given question. \\\n"
        "You must conduct reasoning inside <think> and </think> first every time you get new information. \\\n"
        "After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search>\n"
        "You can search as many times as your want. \\\n Information will be gaven in the next round of dialogue. \\\n"
        "If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, "
        "without detailed illustrations. For example, <answer> Beijing </answer>. Question: " \
       --rename golden_answers:response question:prompt
    ```

3. **修改以下配置数据路径并运行脚本**
   ```configs/datasets/multiturn_grpo_qwen_7b_search_r1.yaml```
   ```bash examples/data/preprocess_multiturn_data_search_r1.sh```

### 3.2 工具启动

建议使用GPU启动工具

``` bash
conda activate retriever
cd agents/search_r1_agent
bash retrieval_launch.sh
```

### 3.3 单机训练

修改 ```examples/grpo/start_search_r1_agent_rl.sh``` 中的配置为```integrated_grpo_trainer_qwen25_7b_1node_search_r1_new_gbs256_nq_hotpot.yaml```

``` bash
bash examples/grpo/start_search_r1_agent_rl.sh
```

### 3.3 多机训练(推荐)

在多节点环境下启动训练。默认配置为 **4 节点**，在 910B3 机器上的性能表现约为 **800s / iter**。

1. **配置修改**
   训练脚本中默认采用四机配置。如需调整节点数量或资源配置，可直接编辑以下脚本：

    ```text
    examples/grpo/start_search_r1_agent_rl.sh
    ```

   主要可配置项包括
   **[可选]多机/单机配置**:```integrated_grpo_trainer_qwen25_7b_4node_search_r1_new_gbs256_nq_hotpot.yaml```
   **[必须]工具调用服务地址**：```export SEARCH_R1_SERVICE_URL=http://127.0.0.1:8000/retrieve```
   **[可选]训练时reward计算方式(默认为subem)**：```export COMPUTE_REWARD_FUNC="subem" # subem | em```
2. **master节点启动**

    ``` bash
    bash examples/grpo/start_search_r1_agent_rl.sh
    ```

3. **worker节点启动**
   同步修改工作节点启动脚本```start_search_r1_agent_rl_worker.sh```中的工具调用服务和reward计算方式与主节点一致。修改工作节点启动脚本中的```HEAD_IP```和```HEAD_PORT```与主节点对应。
   在每个工作节点执行以下命令：

    ```bash
    bash examples/grpo/start_search_r1_agent_rl_worker.sh
    ```

## 4. 评测

### 4.1 vLLM评测方式

推荐使用此评测方式，3610条数据约耗时7-8min

1. **参数说明**

评测脚本支持通过 **命令行参数** 传入配置

| 参数名 | 类型 | 默认值 | 说明 |
|------|------|------|------|
| `--data-file` | str | `data/nq_search/test.json` | 评测数据文件路径 |
| `--output-file` | str | `eval_results.jsonl` | 评测结果输出文件（jsonl 格式） |
| `--model-id` | str | 无 | 模型路径或 HuggingFace Model ID |
| `--num-gpus` | int | 8 | 使用的 GPU 数量 |
| `--use-batch-eval` | bool | true | 是否启用批量评测模式(多并发推理加速) |
| `--batch-size` | int | 48 | 单实例并发处理的样本数量（根据显存调整） |
| `--score-method` | str | `em` | 评分方式：`em`（精确匹配）或 `subem`（子串匹配） |

---

1. **参数含义说明**

   2.1 **数据与输出**

- `--data-file`
  指定评测数据集路径。
  **换用自己的数据集时需要自行适配问题和答案的解析，即修改`def extract_qa(item)`**

- `--output-file`
  评测结果输出路径，推荐使用 `.jsonl` 以支持逐样本记录。

  2.2 **模型相关**

- `--model-id`
  可以是：
  - 本地模型路径
  - HuggingFace Hub 模型名

- `--num-gpus`
  指定 vLLM 推理使用的 GPU 数量。

  2.3 **批量评测配置**

- `--use-batch-eval`
  是否启用批量并发评测模式，用于提升推理吞吐。

- `--batch-size`
  每轮并发处理的样本数量：
  - 需根据模型大小与显存容量调整

  2.4 **评分方式**

- `--score-method`
  - `em`
  - `subem`

1. **使用示例**

   3.1 **单机多卡评测（推荐）**

```bash
python batch_eval_vllm.py \
  --data-file data/nq_search/test.json \
  --output-file eval_results_vllm_nq_subem.jsonl \
  --model-id /opt/DPC/models/model/Qwen2.5-7B-Instruct \
  --num-gpus 8 \
  --use-batch-eval \
  --batch-size 48 \
  --score-method subem
```

### 4.2 huggingFace评测方式

``` bash
cd agents/search_r1_agent/eval
修改数据集路径与权重路径
python eval.py
```

## 5. AgenticRL5.0 运行SearchR1

(1) **设置已有的GPU工具服务**
在容器内```export SEARCH_R1_SERVICE_URL=http://127.0.0.1:8000/retrieve```，也可以直接在代码里面修改（agents\search_r1_agent\environment\search_tool.py），**免得启动任务后忘记**
(2) **确定数据集**
    数据集路径为```/path/to/data/dataset/nq_hotpot/nq_hotpotqa_train_transformed/rl```
(3) **参考配置**
   着重PD分离的多机配置：以跑Qwen3-30B-A3B-Instruct-2507的MOE模型，one-step, 2T(2个训练节点)为例，参考运行配置：```direct_p1d1t2_qwen3_30b2507_SearchR1_train_one_step_off.yaml```,以及基础配置```base_integrated_grpo_trainer_search_r1_qwen3_30b.yaml```，**大部分参数非必要可以不用动。经常修改的主要有以下**：

- **修改其中的num_npus来确定训练节点数**
  主要关注以下参数，当前例子中训练节点数为2

   ```text
   rl_config:
         actor_resource:
           num_npus: 16
   ```

- **各种路径**

- **混合批次配置**
  混合批次修改主要涉及一下两个参数，注意hybrid_batch_num修改为1的时候，同时关闭enable_version_control，否则会遇到在iter2卡住的问题

   ```text
     hybrid_batch_num: 1 #TODO
     enable_version_control: false #TODO
   ```

- **并行配置**
  训练并行参数与权重切分对齐
  推理秉性参数中的EP与runtime_env的VLLM_DP_SIZE一致

- **P实例与D实例**
  注意P与D的个数在启动脚本里面设置，启动脚本参考```start_roma_vllm_proxy_pd.sh```，注意一个实例所需要的卡数为并行参数的乘积，用此来确定节点数量。

**重点说明**

- 如何修改xPyD，调整x和y
  P实例与D实例个数变化时，主要修改```start_roma_vllm_proxy_pd.sh```的以下参数为对应个数：

   ```text
  export PREFILL_INSTANCE_COUNT=1 #TODO PREFILL实例数
  export DECODE_INSTANCE_COUNT=1 #TODO DECODE实例数
   ```

  注意卡数为P实例个数和并行参数之积除以8 + D实例个数和并行参数之积除以8
- 开启或者关闭，混合批次，关注以下两个参数：

    ```text
    hybrid_batch_num: 2 #TODO
    enable_version_control: true #TODO
    ```

  注意当hybrid_batch_num改回1时，enable_version_control设置为false

**其它注意事项**

- 配置中generate_config/infer_expert_parallel_size,需要与runtime_env里面的VLLM_DP_SIZE保持一致，否则会出现vLLM启动中shape不对应的问题
- 配置中最好内层配置和外层配置同时修改，防止读取失效的问题发生
- PD分离模式下启动脚本中的，MASTER_TRAIN_INDEX一定要根据训练的节点起始值调整，不然就会直接宕机，无任何原因退出
- PD分离模式下镜像要使用agentic-rl-a2-vllm-011:4.0.1,不然vLLM拉起失败
- PD分离模式下runtime_env里面VLLM的版本需要修改为0.11.0
- PD分离模式下所有节点有不同的日志，其中训练日志重定向了，vLLM日志未重定向,DEBUG时需要注意
- PD分离模式下运行脚本中在上需要将所有的```VC_WORKER_HOSTS```修改为```VC_TASK_HOSTS```,在云道上使用```VC_WORKER_HOSTS```
- 内嵌vLLM mp方式下，若是权重读取超时可以增加collect_rpc任务超时时间到1000，在```aura/runner/infer_service/infer_server/vllm_mp_infer_server.py```中353行
- 注意max_model_len是vLLM启动脚本中硬编码的```vllm_serve.sh```外部设置可能没作用

(4) **启动命令**

- (i) **内嵌vLLM mp启动方式（后续多机以PD分离为主，关注单机调测启动）**
  - 单机启动验证功能：
      ```sh run_start_in_local.sh --config-name xxx.yaml```
  - 多机裸机拉起：
      主节点：```sh run_start_in_local.sh --config-name xxx.yaml --is-master true --master-addr x.x.x.x:6000 --ray-port 6000```
      工作节点：```sh run_start_in_local.sh --config-name xxx.yaml --is-master false--master-addr x.x.x.x:6000 --ray-port 6000```
  - 环境多机拉起参考以下脚本：

      ```bash
        #!/bin/bash
        # Copyright Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
        # 训练任务启动入口

        export VLLM_CACHE_ROOT=/models/.cache/vllm/
        export TORCH_EXTENSIONS_DIR=/models/.cache/torch_extension
        export RAY_heartbeat_timeout_milliseconds=120000
        export RAY_raylet_heartbeat_period_milliseconds=120000


        HOSTS="$VC_TASK_HOSTS"
        MASTER_HOST_0="${HOSTS%%,*}"
        MASTER_HOST_16="$(echo "$VC_TASK_HOSTS" | cut -d ',' -f 3)"
        current_time=$(date +"%Y%m%d_%H%M%S")
        echo "current_time: $current_time"
        export MASTER_NODE=$MASTER_HOST_0

        if [ "$VC_TASK_INDEX" = "0" ]; then
          echo "master head starts"
          bash run_start_in_local.sh \
          --config-name direct_4node_qwen3_30b_search_r1_train_one_step_off_hybrid_batch.yaml \
          --is-master true \
          --master-addr "${MASTER_NODE}":6000 \
          --ray-port 6000 \
          2>&1 | tee ./logs_SearchR1_${current_time}.log
          sleep 30000000000000000000000000000000

        else
          echo "worker head starts"
          sleep 30
          bash run_start_in_local.sh \
          --config-name direct_4node_qwen3_30b_search_r1_train_one_step_off_hybrid_batch.yaml \
          --is-master false \
          --master-addr "${MASTER_NODE}":6000 \
          --ray-port 6000
          sleep 30000000000000000000000000000000
        fi
      ```

- (ii) **PD分离方式启动**
  修改```start_roma_vllm_proxy_pd.sh```文件中的8-24行，修改方式详见注释，未注释的不用管

  ```bash
  #!/bin/bash
  # Copyright Huawei Technologies Co., Ltd. 2021-2021. All rights reserved.
  # 训练任务启动入口

  ###################################################################################
  # 待修改配置如下:
  #export VC_TASK_HOSTS="127.0.0.1"
  MASTER_TRAIN_INDEX=2 #TODO 写成第一个训练的节点的索引，训练节点在后，推理节点在前,从0索引开始，比如1p1d2t,此值为2
  CONFIG_NAME=direct_p1d1t2_qwen3_4b_train_one_step_off_searchR1 #TODO 对应的配置文件
  export VLLM_VERSION=0.11.0
  export ENABLE_EXPERT_PARALLEL=false #TODO 与配置一致
  export MODEL_PATH=/models/m00951355/Qwen/Qwen3-4B/ #TODO 模型路径
  export SERVED_MODEL_NAME=Qwen3-4B #TODO 模型名称，与配置中完全一致
  export PREFILL_INSTANCE_COUNT=1 #TODO PREFILL实例数
  export DECODE_INSTANCE_COUNT=1 #TODO DECODE实例数
  export PREFILL_TENSOR_PARALLEL_SIZE=4  #TODO PREFILL TP
  export PREFILL_DATA_PARALLEL_SIZE=2 #TODO PREFILL DP
  export DECODE_TENSOR_PARALLEL_SIZE=4 #TODO DECODE TP
  export DECODE_DATA_PARALLEL_SIZE=2 #TODO DECODE DP
  #打点统计
  export ENABLE_VLLM_STAT=false
  task_label="235b_prepare_post_detail_$(date +%s%3N)"
  export VLLM_STAT_SAVE_PATH_SUFFIX=${task_label}
  export ENABLE_TENSOR_SIMILARITY_CHECK=true #TODO 开启
  ```

  在运行脚本```start_roma_vllm_proxy_pd.sh```

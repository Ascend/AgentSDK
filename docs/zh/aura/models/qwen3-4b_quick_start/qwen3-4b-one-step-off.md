# Qwen3-4B 分离模式快速拉起指南

## **容器环境部署**

本文档以 Atlas A3 服务器 16 卡、Ubuntu 系统为例进行说明。

第一步：拉取预构建镜像：

```shell
docker pull swr.cn-south-1.myhuaweicloud.com/ascendhub/agentsdk:26.1.0-a3-ubuntu22.04-py3.11
```

第二步：创建容器：

```shell
docker run --name your_container_name \
    --hostname agent \
    --network host \
    -it -d --shm-size=500g \
    --device=/dev/davinci0 --device=/dev/davinci1 \
    --device=/dev/davinci2 --device=/dev/davinci3 \
    --device=/dev/davinci4 --device=/dev/davinci5 \
    --device=/dev/davinci6 --device=/dev/davinci7 \
    --device=/dev/davinci8 --device=/dev/davinci9 \
    --device=/dev/davinci10 --device=/dev/davinci11 \
    --device=/dev/davinci12 --device=/dev/davinci13 \
    --device=/dev/davinci14 --device=/dev/davinci15 \
    --device=/dev/davinci_manager \
    --device=/dev/hisi_hdc \
    --device=/dev/devmm_svm \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /usr/share/zoneinfo/Asia/Shanghai:/etc/localtime \
    -v /usr/local/sbin:/usr/local/sbin \
    swr.cn-south-1.myhuaweicloud.com/ascendhub/agentsdk:26.1.0-a3-ubuntu22.04-py3.11  \
    sleep infinity
```

第三步：进入容器环境

```shell
docker exec -it your_container_name bash
```

## **模型获取**

本实验使用 Qwen3-4B 模型，相关模型可以通过[ModelScope Qwen3-4B 模型仓库](https://www.modelscope.cn/models/Qwen/Qwen3-4B)获取。

```shell
modelscope download --model Qwen/Qwen3-4B --local_dir /path/to/Qwen3-4B
```

## **数据集获取**

### **下载数据集**

本实验使用的 Math 领域的 gsm8k 数据集可通过[ModelScope gsm8k 数据集](https://www.modelscope.cn/datasets/AI-ModelScope/gsm8k)获取。

```shell
modelscope download --dataset AI-ModelScope/gsm8k --local_dir /path/to/gsm8k
```

### **数据集处理**

使用下面 verl 官方提供的数据处理脚本[gsm8k.py](https://github.com/verl-project/verl/blob/v0.7.0/examples/data_preprocess/gsm8k.py)处理数据集：

```shell
# 处理数据集
python3 gsm8k.py \
    --local_dataset_path /path/to/gsm8k \
    --local_save_dir /path/to/gsm8k-output
```

### **数据格式转换**

分离模式需要将数据集转换为 bin/idx 格式。训练集和测试集的处理流程一致，以下以训练集为例进行说明，转换流程如下：

```text
parquet → jsonl → bin/idx
```

#### **parquet 转 jsonl**

创建转换脚本 `convert_data.py`：

```python
import pandas as pd
import json
import os
import argparse

def convert_parquet_to_filtered_jsonl(input_parquet, output_jsonl):
    """
    将 Parquet 转换为 JSONL 格式，提取特定字段。
    """
    print(f"正在读取 Parquet 文件: {input_parquet} ...")

    try:
        df = pd.read_parquet(input_parquet)
        records = df.to_dict('records')
    except Exception as e:
        print(f"读取 Parquet 失败: {e}")
        return

    print(f"读取到 {len(records)} 行数据，开始提取字段...")

    count = 0
    with open(output_jsonl, 'w', encoding='utf-8') as f_out:
        for data in records:
            try:
                new_data = {
                    "data_source": data.get('data_source'),
                    "question": data['prompt'][0]['content'],
                    "answer": data['reward_model']['ground_truth'],
                    "labels": data['reward_model']['ground_truth']
                }
                f_out.write(json.dumps(new_data, ensure_ascii=False) + '\n')
                count += 1
            except KeyError:
                pass
            except Exception as e:
                print(f"处理行出错: {e}")

    print(f"处理完成！成功提取 {count} 条数据，保存至: {output_jsonl}")

def main():
    parser = argparse.ArgumentParser(description="将 Parquet 转换为 JSONL")
    parser.add_argument('--input', type=str, required=True, help='输入的 parquet 文件路径')
    parser.add_argument('--output', type=str, default='output.jsonl', help='输出的 jsonl 文件路径')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"错误: 找不到输入文件 {args.input}")
        return

    convert_parquet_to_filtered_jsonl(args.input, args.output)

if __name__ == "__main__":
    main()
```

执行转换：

```shell
python convert_data.py --input train.parquet --output train.jsonl
python convert_data.py --input test.parquet --output test.jsonl
```

#### **jsonl 转 bin/idx**

**准备配置文件**

gsm8k 数据集对应的数据处理配置文件已存在于 [`aura/configs/datasets/gsm8k.yaml`](../../../../../aura/configs/datasets/gsm8k.yaml)，用户可直接修改该文件中的路径参数：

```yaml
# configs/datasets/gsm8k.yaml
input: /path/to/input_data_dir
tokenizer_name_or_path: /path/to/Qwen3-4B
output_prefix: /path/to/output/train/rl
handler_name: R1AlpacaStyleInstructionHandler
tokenizer_type: HuggingFaceTokenizer
workers: 8
log_interval: 1000
prompt_type: qwen
dataset_additional_keys: [labels]
map_keys: {"query":"", "response":"labels", "prompt": "question"}
```

**配置参数说明**

| 参数                        | 类型   | 说明                                                                               |
|---------------------------|------|----------------------------------------------------------------------------------|
| `input`                   | str  | 输入的 jsonl 文件所在目录的路径，包含 train.jsonl 和 test.jsonl                                  |
| `tokenizer_name_or_path`  | str  | 分词器路径，必须与后续训练的模型保持一致                                                             |
| `output_prefix`           | str  | /path/to/output/train 为输出文件路径，需提前创建，rl 为输出文件前缀                                   |
| `handler_name`            | str  | 数据处理器名称，决定了数据的拼接模板 。`R1AlpacaStyleInstructionHandler`是 Qwen 等模型进行 SFT/RL 训练的标准格式 |
| `tokenizer_type`          | str  | 分词器类型，常用 `HuggingFaceTokenizer`                                                  |
| `workers`                 | int  | 并行 worker 数                                                                      |
| `log_interval`            | int  | 日志输出间隔（处理多少条数据后输出）                                                               |
| `prompt_type`             | str  | 指定模型对应的 chat template，例如 qwen/chatml/llama3                                      |
| `dataset_additional_keys` | list | 需额外保留的数据字段                                                                       |
| `map_keys`                | dict | 字段映射，将原始 json 字段映射到框架内部标准字段                                                      |

**执行数据处理**

```shell
# 处理训练集
python3 /path/to/AgentSDK/aura/cli/preprocess_data.py gsm8k
```

**生成文件**

处理完成后会生成以下文件：

```text
/path/to/output/
├── rl_packed_attention_mask_document.bin    # attention mask 二进制数据
├── rl_packed_attention_mask_document.idx    # attention mask 索引文件
├── rl_packed_input_ids_document.bin         # input ids 二进制数据
├── rl_packed_input_ids_document.idx         # input ids 索引文件
├── rl_packed_labels_document.bin            # labels 二进制数据
└── rl_packed_labels_document.idx            # labels 索引文件
```

## **文件修改**

在快速拉起 qwen3-4b math 场景前，需要修改以下配置文件，需要进行修改的参数可以参照文件头的注释，请将其中的示例路径修改为实际路径。

1. [分离训练配置文件](../../../../../aura/configs/train/verl_train_async_A3_t16_qwen3_4b_math_fsdp.yaml)
2. [分离推理配置文件](../../../../../aura/configs/infer/vllm_infer_i16_qwen3_4b.yaml)

### 修改hosts.conf

修改为双机对应的 IP 地址：

```shell
# [多机训练+推理]
# 双机, 训推分离, 分节点部署
# host,index,train_master_index,infer_master_index(可选)
<节点IP1>,0,0
<节点IP2>,1,1
```

### 修改base.conf

```shell
# [train]
# 启动训练相关参数
# 工作模式：hybrid 共卡模式 | one_step_off 全异步分离模式
work_mode=one_step_off

# 共卡和分离模式均需要配置训练yaml文件
train_config_name=verl_train_async_A3_t16_qwen3_4b_math_fsdp

# 分离模式需要单独配置推理yaml文件
infer_config_name=vllm_infer_i16_qwen3_4b

# [resume]
# 启动断点续训相关参数
# 需要监控的启动脚本：start_rl_with_msrl_vllm.sh | start_rl_with_verl_vllm.sh
monitor_cmd=start_rl_with_verl_vllm.sh

# 断点续训重试次数, 默认100次
max_retries=100

# 第一次启动是否需要清空ckpt文件夹: 0 不清理; 1 需要清理
clean_old_ckpt=0
```

## **配置环境变量**

### 配置 DEFAULT_SOCKET_IFNAME

包含正确本地 IP 的网络接口名称。

1. 执行 ifconfig 命令，查看网络配置：

    ```shell
    ifconfig
    ```

2. 假设得到打印信息（部分）为：

    ```text
    docker0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
            inet 172.17.0.1  netmask 255.255.0.0  broadcast 172.17.255.255

    enp189s0f0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
            inet 192.168.0.1  netmask 255.255.0.0  broadcast 192.168.255.255

    enp189s0f1: flags=4099<UP,BROADCAST,MULTICAST>  mtu 1500
            inet 192.168.100.100  netmask 255.255.255.0  broadcast 192.168.100.255
    ```

3. 假设本地 IP 为 192.168.0.1，那么指向本地 IP 对应网络接口的值即为 enp189s0f0，即需要执行：

    ```shell
    export DEFAULT_SOCKET_IFNAME=enp189s0f0
    ```

### 配置 ASCEND_RT_VISIBLE_DEVICES

配置可用的 NPU 的卡数。

```shell
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
```

## **启动训练**

启动训练脚本

```shell
# 进入自己的工作目录
cd /home/work/AgentSDK/aura
# 启动训练脚本
bash scripts/start_rl_with_verl_vllm.sh
```

> [!NOTE] 说明
> 本文档专门针对 Qwen3-4B 分离模式（one-step-off）的启动示例，提供该场景下的快速拉起步骤；通用且详细的完整启动流程请参考 [快速入门指南](../../03_quick_start.md)。

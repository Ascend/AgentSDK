# 安装部署指南<a name="ZH-CN_TOPIC_0000002492554169"></a>

Agent SDK的安装部署流程包含以下三个主要步骤：

1. 容器环境部署
2. 准备模型权重
3. 准备训练数据

## 容器环境部署<a name="ZH-CN_TOPIC_0000002492554221"></a>

通过 Dockerfile 可快速构建镜像，Dockerfile 可在 AgentSDK 项目源码的 [`dockers`](../../dockers) 目录下获取，用户可根据实际需求修改 Dockerfile 中的路径参数。

### 步骤1：构建镜像

将 Dockerfile 放置到任意目录后，进入该目录执行构建命令：

```shell
cd /path/to/dockerfile_directory
bash build_image.sh
```

构建脚本将根据自动识别服务器类型，构建对应的镜像。

### 步骤2：创建容器

以Atlas A3镜像为例，创建容器：

```shell
docker run --name your_container_name --privileged \
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
    aura-a3:26.1.0  \
    sleep infinity
```

> 说明：根据NPU数量的不同，挂载不同数量的设备ID。例如：Atlas A3有16个NPU，需挂载16个设备ID，每个设备ID对应一个NPU。

### 步骤3：进入容器

```shell
docker exec -it your_container_name bash
```

## 准备模型权重<a name="ZH-CN_TOPIC_0000002459514672"></a>

### 下载模型权重<a name="ZH-CN_TOPIC_0000002492554173"></a>

本小节介绍Agent SDK所需模型权重的下载方式。用户可根据实际需求选择合适的模型，以下以[Qwen2.5-7B-Instruct](https://www.modelscope.cn/models/Qwen/Qwen2.5-7B-Instruct)为例：

```shell
modelscope download --model Qwen/Qwen2.5-7B-Instruct --local_dir /path/to/Qwen2.5-7B-Instruct
```

### 处理模型权重<a name="ZH-CN_TOPIC_0000002492554173"></a>

模型权重下载完成后，需根据使用的训练后端以及训练并行策略，决定是否将其转换为Megatron格式。

> 说明：详细的权重处理步骤将在后续版本中更新。

## 准备训练数据<a name="ZH-CN_TOPIC_0000002492554221"></a>

### 下载训练数据<a name="ZH-CN_TOPIC_0000002492554173"></a>

本小节介绍Agent SDK训练数据的获取方式。以数学Agent场景为例，我们使用[gsm8k](https://www.modelscope.cn/datasets/AI-ModelScope/gsm8k)数据集，包含训练集和测试集数据：

```shell
# 下载数据集
modelscope download --dataset AI-ModelScope/gsm8k --local_dir /path/to/gsm8k
```

> 说明： 首次训练时，应根据模型能力选择合适的数据集，参数量较低的模型应选择较为简单的数据集，便于模型学习

### 处理训练数据<a name="ZH-CN_TOPIC_0000002492554173"></a>

根据训练模式的不同，数据处理方式也有所区别：

| 训练模式 | 数据处理方式 | 说明 |
|---------|-------------|------|
| **共卡模式 (Hybrid)** | verl 官方脚本 | 数据格式为 parquet |
| **分离模式 (One-Step-Off)** | AgentSDK 脚本 | 数据格式为 Megatron |

#### 共卡模式数据处理

使用 verl 官方提供的数据处理脚本处理数据集：

- [gsm8k.py](https://github.com/verl-project/verl/blob/v0.7.0/examples/data_preprocess/gsm8k.py)：处理数据集

```python
import argparse
import os
import re

import datasets

from verl.utils.hdfs_io import copy, makedirs


def extract_solution(solution_str):
    solution = re.search("#### (\\-?[0-9\\.\\,]+)", solution_str)
    assert solution is not None
    final_solution = solution.group(0)
    final_solution = final_solution.split("#### ")[1].replace(",", "")
    return final_solution


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=None, help="The save directory for the preprocessed dataset.")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--local_dataset_path", default=None, help="The local path to the raw dataset, if it exists.")
    parser.add_argument(
        "--local_save_dir", default="~/data/gsm8k", help="The save directory for the preprocessed dataset."
    )

    args = parser.parse_args()
    local_dataset_path = args.local_dataset_path

    data_source = "openai/gsm8k"

    if local_dataset_path is not None:
        dataset = datasets.load_dataset(local_dataset_path)
    else:
        dataset = datasets.load_dataset(data_source, "main")

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    instruction_following = 'Let\'s think step by step and output the final answer after "####".'

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question_raw = example.pop("question")

            question = question_raw + " " + instruction_following

            answer_raw = example.pop("answer")
            solution = extract_solution(answer_raw)
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer_raw,
                    "question": question_raw,
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    hdfs_dir = args.hdfs_dir
    local_save_dir = args.local_dir
    if local_save_dir is not None:
        print("Warning: Argument 'local_dir' is deprecated. Please use 'local_save_dir' instead.")
    else:
        local_save_dir = args.local_save_dir

    train_dataset.to_parquet(os.path.join(local_save_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_save_dir, "test.parquet"))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_save_dir, dst=hdfs_dir)
```

```shell
# 处理数据集
python3 gsm8k.py \
    --local_dataset_path /path/to/gsm8k \
    --local_save_dir /path/to/gsm8k-output
```

#### 分离模式数据处理

分离模式需要将数据集转换为 bin/idx 格式。训练集和测试集的处理流程一致，以下以训练集为例进行说明，转换流程如下：

```text
parquet → jsonl → bin/idx
```

> **说明**：后续版本将与共卡模式进行统一，数据格式将统一为 parquet。

##### 步骤 1：parquet 转换为 jsonl

首先，与共卡模式相同，通过gsm8k.py将gsm8k数据集转换为标准格式数据集

其次，创建转换脚本 `convert_data.py`：

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

##### 步骤 2：jsonl 转换为 bin/idx

**准备配置文件**

在 `configs/` 目录下创建 `datasets/` 目录，用于存放不同数据集的数据处理配置文件

```shell
cd /path/to/AgentSDK/aura/configs
mkdir -p datasets
```

在 `configs/datasets/` 目录下创建gsm8k数据集对应的数据处理配置文件：

```yaml
# configs/datasets/gsm8k.yaml
input: /path/to/input_data_dir
tokenizer_name_or_path: /path/to/tokenizer
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

| 参数 | 类型 | 说明                                                                                 |
|------|------|------------------------------------------------------------------------------------|
| `input` | str | 输入的 jsonl 文件所在目录的路径，包含train.jsonl和test.jsonl                                       |
| `tokenizer_name_or_path` | str | 分词器路径，必须与后续训练的模型保持一致                                                               |
| `output_prefix` | str | /path/to/output/train为输出文件路径，需提前创建，rl为输出文件前缀，生成 `rl_train.bin` 和 `rl_train.idx` 文件 |
| `handler_name` | str | 数据处理器名称，决定了数据的拼接模板 。`R1AlpacaStyleInstructionHandler`是 Qwen 等模型进行 SFT/RL 训练的标准格式   |
| `tokenizer_type` | str | 分词器类型，常用 `HuggingFaceTokenizer`                                                    |
| `workers` | int | 并行worker数                                                                          |
| `log_interval` | int | 日志输出间隔（处理多少条数据后输出）                                                                 |
| `prompt_type` | str | 指定模型对应的 chat template，例如 qwen/chatml/llama3                                        |
| `dataset_additional_keys` | list | 需额外保留的数据字段                                                                         |
| `map_keys` | dict | 字段映射，将原始 json 字段映射到框架内部标准字段                                                        |

**执行数据处理**

```shell
# 处理训练集
python3 /path/to/AgentSDK/cli/preprocess_data.py gsm8k
```

> **说明**：`gsm8k` 为配置文件名（不带 `.yaml` 后缀），脚本会自动从 `configs/datasets/` 目录加载对应的配置。

**生成文件**

处理完成后会生成以下文件：

```text
/path/to/output/
├── train.bin    # 训练集二进制数据
├── train.idx    # 训练集索引文件
├── test.bin     # 测试集二进制数据
└── test.idx     # 测试集索引文件
```

---

> 安装部署流程已完成，请参考[快速启动文档](quick_start.md)来使用AgentSDK。

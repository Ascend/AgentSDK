# Qwen3-4B 共卡模式快速拉起指南

## **容器环境部署**

第一步：进入 docker 目录，执行构建镜像脚本：

```shell
cd /path/to/AgentSDK/docker/aura
docker build -f Dockerfile.a3.ubuntu -t your_image_name:your_image_tag .
```

> [!NOTE] 说明
> 如果服务器架构为 A3，会生成镜像 `aura-a3:26.1.0`。

第二步：创建容器：

```shell
docker run --name your_container_name \
    --hostname agent \
    --network host \
    -it -d --shm-size=500g \
    --device=/dev/davinci0 --device=/dev/davinci1 \
    --device=/dev/davinci2 --device=/dev/davinci3 \
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

第三步：进入容器环境

```shell
docker exec -it your_container_name bash
```

> [!NOTE] 说明
>
>- 后续将提供预构建镜像下载链接，用户可直接下载预构建的镜像

## **模型获取**

本实验使用 Qwen3-4b 模型，相关模型可以通过[本链接](https://www.modelscope.cn/models/Qwen/Qwen3-4B)获取。

```shell
modelscope download --model Qwen/Qwen3-4B --local_dir /path/to/Qwen3-4B
```

## **数据集获取**

### **下载数据集**

本实验使用的 math 领域的 gsm8k 数据集可通过[本链接](https://www.modelscope.cn/datasets/AI-ModelScope/gsm8k)获取。

```shell
modelscope download --dataset AI-ModelScope/gsm8k --local_dir /path/to/gsm8k
```

### **数据集处理**

使用 verl 官方提供的数据处理脚本[gsm8k.py](https://github.com/verl-project/verl/blob/v0.7.0/examples/data_preprocess/gsm8k.py)处理数据集：

```python
# gsm8k.py
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

## **文件修改**

在快速拉起 qwen3-4b math 场景前，需要您修改以下配置文件，需要进行修改的参数可以参照文件头的注释。

 [共卡配置文件](../../../../../aura/configs/train/verl_train_hybrid_A3_t16_qwen3_4b_math_fsdp.yaml)

## **配置环境变量**

### 配置 DEFAULT_SOCKET_IFNAME

包含正确本地 IP 的虚拟网桥名称。

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

3. 假设本地 IP 为 192.168.0.1，那么指向本地 IP 对应虚拟网桥的值即为 enp189s0f0 ，即需要执行：

    ```shell
    export DEFAULT_SOCKET_IFNAME=enp189s0f0
    ```

### 配置 ASCEND_RT_VISIBLE_DEVICES

配置可用的 NPU 的卡数。

```shell
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
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
> 本文档专门针对 Qwen3-4B 共卡模式（hybrid）的启动示例，提供该场景下的快速拉起步骤；通用且详细的完整启动流程请参考 [快速入门指南](../../03_quick_start.md)。

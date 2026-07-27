# 安装指南

当前 Aura 仅提供环境部署流程，Aura 的环境部署流程包含以下三个主要步骤：

1. 容器环境部署
2. 准备模型权重
3. 准备训练数据

## 容器环境部署

容器环境部署有两种方式：

1. 基于镜像构建容器环境
2. 基于 CANN 9.0.0 的容器环境，执行一键式环境配置脚本 [build_env.sh](../../../docker/aura/build_env.sh)

### 方式一：基于镜像构建容器环境

基于镜像构建容器环境，首先需要获取镜像。获取镜像有两种方式：

- 直接从昇腾镜像仓库拉取预构建镜像（推荐）
- 从 Dockerfile 构建镜像

#### 步骤 1：获取镜像

##### 选项一：直接拉取预构建镜像

可从昇腾镜像仓库直接拉取已构建好的镜像，[AgentSDK 镜像发布页面](https://www.hiascend.com/developer/ascendhub/detail/72825ebadb23432ba55dea3f58e68a69)

以 A3 服务器、Ubuntu 系统为例，拉取镜像的命令为：

```shell
docker pull swr.cn-south-1.myhuaweicloud.com/ascendhub/agentsdk:26.1.0-cann9.0.0-torch_npu2.9.0-a3-ubuntu22.04-py3.11
```

##### 选项二：从 Dockerfile 构建镜像

通过 Dockerfile 可快速构建镜像， Dockerfile 可在 [docker](../../../docker) 目录下获取，用户可根据实际需求修改 Dockerfile 中的路径参数。

拉取 Aura 项目源码，进入 docker 目录，构建镜像，以 a3-ubuntu 的 [Dockerfile](../../../docker/aura/Dockerfile.a3.ubuntu) 为例：

```shell
git clone https://gitcode.com/Ascend/AgentSDK.git
cd /path/to/AgentSDK/docker/aura
docker build -f Dockerfile.a3.ubuntu -t your_image_name:your_image_tag .
```

用户需根据服务器类型选择对应的 Dockerfile 构建镜像。

#### 步骤 2：创建容器

以 Atlas A3服务器16卡为例，创建容器：

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
    your_image_name:your_image_tag  \
    sleep infinity
```

> [!NOTE]
>
> 1. 根据 NPU 数量的不同，挂载不同数量的设备 ID。例如： Atlas A3 有 16 个 NPU，需挂载 16 个设备 ID，每个设备 ID 对应一个 NPU。
> 2. 镜像内默认工作目录为 /home/work，因此不建议挂载整个 /home 目录，以避免覆盖容器内默认工作空间或引发权限冲突。

#### 步骤 3：进入容器

```shell
docker exec -it your_container_name bash
```

### 方式二：使用一键式环境配置脚本 build_env.sh

使用一键式环境配置脚本前，需提前准备好 CANN 9.0.0 的容器环境，包括安装 CANN 9.0.0 的驱动、配置环境变量等，用户可根据实际需求，修改第三方库安装路径。一键式环境配置脚本将自动安装 Aura 及其所有依赖，包含 vLLM、vllm-ascend、MindSpeed、Megatron-LM、verl、transformers 等第三方库依赖，以及 Python 相关依赖。

```shell
cd /path/to/AgentSDK/docker/aura
bash build_env.sh
```

> [!NOTE]
>
> 一键拉起脚本 `build_env.sh` 会对当前 Python 环境执行全局 `pip install -e .` 等操作，并克隆多个仓库到 `/home/work`，因此**请勿在宿主机原生 Python 环境或已有其他项目依赖的虚拟环境中执行**。建议仅在全新的 CANN 9.0.0 容器内使用；若需要隔离环境，请自行创建独立虚拟环境后再运行该脚本。

## 准备模型权重

### 下载模型权重

本小节介绍 Aura 所需模型权重的下载方式。用户可根据实际需求选择合适的模型，以下以[Qwen2.5-7B-Instruct](https://www.modelscope.cn/models/Qwen/Qwen2.5-7B-Instruct)为例：

```shell
modelscope download --model Qwen/Qwen2.5-7B-Instruct --local_dir /path/to/Qwen2.5-7B-Instruct
```

## 准备训练数据

### 下载训练数据

本小节介绍 Aura 训练数据的获取方式。以数学 Agent 场景为例，我们使用[gsm8k](https://www.modelscope.cn/datasets/AI-ModelScope/gsm8k)数据集，包含训练集和测试集数据：

```shell
# 下载数据集
modelscope download --dataset AI-ModelScope/gsm8k --local_dir /path/to/gsm8k
```

> 说明：首次训练时，应根据模型能力选择合适的数据集，参数量较低的模型应选择较为简单的数据集，便于模型学习

### 处理训练数据

根据训练模式的不同，数据处理方式也有所区别：

| 训练模式 | 数据处理方式 | 说明 |
|---------|-------------|------|
| **共卡模式 (Hybrid)** | verl 官方脚本 | 数据格式为 parquet |
| **分离模式 (One-Step-Off)** | Aura 脚本 | 数据格式为 Megatron |

#### 共卡模式数据处理

使用 verl 官方提供的数据处理脚本[gsm8k.py](https://github.com/verl-project/verl/blob/v0.7.0/examples/data_preprocess/gsm8k.py)处理数据集：

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

##### 步骤 1： parquet 数据结构处理

与共卡模式相同，通过 gsm8k.py 将 gsm8k 数据集转换为训练所需的标准格式数据集，参考[共卡模式数据处理](#共卡模式数据处理)

##### 步骤 2： parquet 转换为 jsonl

使用代码仓中提供的转换脚本 [`convert_data.py`](../../../aura/cli/convert_data.py) 将 parquet 转换为 jsonl 格式，执行时通过 `--input` 和 `--output` 参数指定输入 parquet 文件与输出 jsonl 文件的绝对路径：

```shell
cd /path/to/AgentSDK/aura/cli
python convert_data.py --input /path/to/gsm8k-parquet/train.parquet --output /path/to/gsm8k-jsonl/train.jsonl
python convert_data.py --input /path/to/gsm8k-parquet/test.parquet --output /path/to/gsm8k-jsonl/test.jsonl
```

##### 步骤 3： jsonl 转换为 bin/idx

**准备配置文件**

gsm8k 数据集对应的数据处理配置文件已存在于 [aura/configs/datasets/gsm8k.yaml](../../../aura/configs/datasets/gsm8k.yaml)，用户可直接修改该文件中的路径参数：

```yaml
# aura/configs/datasets/gsm8k.yaml
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

执行前必须先进入 `aura` 目录，因为脚本运行时需要导入 `aura` 目录下的 `third_party` 依赖。

```shell
# 进入 aura 目录（必须，因脚本会导入 third_party 依赖）
cd /path/to/AgentSDK/aura
# 处理训练集
python3 /path/to/AgentSDK/aura/cli/preprocess_data.py gsm8k
```

> **说明**：`gsm8k` 为配置文件名（不带 `.yaml` 后缀），脚本会自动从 `configs/datasets/` 目录加载对应的配置。

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

## 环境变量配置

### 设置 DEFAULT_SOCKET_IFNAME

通过`ifconfig`指令，查看自己的网卡 ID，以本机 IP 地址为 192.168.0.1 为例：

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

3. 假设本地 IP 为 192.168.0.1，那么指向本地 IP 对应网络接口的值即为 enp189s0f0 ，即需要执行：

    ```shell
    export DEFAULT_SOCKET_IFNAME=enp189s0f0
    ```

### 设置 ASCEND_RT_VISIBLE_DEVICES

根据自己实际需要设置可用的 NPU 的卡数，例如需要指定 0~15 号 NPU：

```shell
# 配置可用的NPU的卡数
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
```

## 卸载与清理

当不再需要 Aura 运行环境时，可按以下步骤清理容器、镜像及相关数据，释放宿主机资源。

### 1. 停止并删除容器

```shell
# 查看当前运行的容器，确认容器名称
docker ps -a

# 停止容器（将 your_container_name 替换为实际容器名）
docker stop your_container_name

# 删除容器
docker rm your_container_name
```

### 2. 删除镜像

```shell
# 查看本地镜像，确认镜像名和标签
docker images

# 删除镜像（将 <image_name>:<tag> 替换为实际镜像名，如 aura-a3:26.1.0）
docker rmi <image_name>:<tag>
```

> [!NOTE]
>
> 删除镜像前需确保没有运行中或已停止的容器依赖该镜像，否则需先执行步骤 1 删除对应容器。

### 3. 清理构建缓存（可选）

若构建过程中产生了大量中间层镜像或缓存，可一并清理以回收磁盘空间：

```shell
# 清理悬空镜像（<none> 标签的中间层镜像）
docker image prune -f

# 清理所有未使用的镜像、容器、网络和构建缓存
docker system prune -f
```

> [!CAUTION]
>
> `docker system prune -a` 会删除所有未被任何容器引用的镜像，执行前请确认其他项目不依赖这些镜像。

### 4. 清理 Aura 相关数据（可选）

如需彻底清除 Aura 运行产生的数据，可删除以下目录：

- 容器内工作目录 `/home/work` 下的第三方仓库克隆（如 verl、vLLM、Megatron-LM 等）
- 训练产出的权重、日志和轨迹文件，默认位于 `${hydra:runtime.cwd}/weights` 和 `${hydra:runtime.cwd}/outputs`
- 共享存储中的 rollout 数据和 checkpoint 目录

---

> 环境部署流程已完成，请参考[快速启动文档](./03_quick_start.md)来使用 Aura。

# 快速入门

## **简介**

Aura 使用脚本`start_rl_with_verl_vllm.sh`来启动，本章节通过介绍该脚本的使用，帮助用户熟悉本软件。

## **环境准备**

Aura 提供两种构建运行环境的方式，用户可根据实际情况任选其一：

1. **使用预构建镜像创建容器**（推荐）

    直接基于预构建镜像运行容器。具体操作请参见“[基于镜像构建容器环境](02_installation_guide.md#方式一基于镜像构建容器环境)”。

2. **在 CANN 9.0.0 容器中使用一键拉起脚本**

    如果您已经在 CANN 9.0.0 的基础镜像容器内，可以执行一键拉起脚本完成 Aura 及其所有依赖（vLLM、 vllm-ascend、 MindSpeed、 Megatron-LM、 verl、 transformers 等）的安装。具体操作请参见“[使用一键式环境配置脚本 build_env.sh](02_installation_guide.md#方式二使用一键式环境配置脚本-build_envsh)”。

## **使用流程**

Aura 提供了训练模型示例。

- 准备训练模型和数据集，具体操作请参见“[准备模型权重](02_installation_guide.md#准备模型权重)”与“[准备训练数据](02_installation_guide.md#准备训练数据)”。
- 配置环境变量，具体操作请参见“[环境变量配置](02_installation_guide.md#环境变量配置)”。
- 根据实际环境修改 YAML 配置文件中的路径参数，完整示例请参见“[配置文件示例](05_api_python.md#配置文件示例)”。
- 根据实际环境修改 hosts.conf 和 base.conf 配置文件，相关参数解释请参见“[服务启动说明](05_api_python.md#服务启动说明)”。

### 修改base.conf

根据工作模式选择 `work_mode`，并根据使用的模型选择相应的 `train_config_name` 和 `infer_config_name`，以模型 Qwen3-32B，基于 math 场景启动分离模式为例，设置配置文件与启动方式如下：

```shell
# [train]
# 启动训练相关参数
# 工作模式：hybrid 共卡模式 | one_step_off 单步滞后分离模式
work_mode=one_step_off

# 共卡和分离模式均需要配置训练yaml文件
train_config_name=verl_train_async_t16_qwen3_32B_math

# 分离模式需要单独配置推理yaml文件, 共卡模式该配置不生效
infer_config_name=vllm_infer_i16_qwen3_32b

# [resume]
# 启动断点续训相关参数
# 需要监控的启动脚本：start_rl_with_msrl_vllm.sh | start_rl_with_verl_vllm.sh
monitor_cmd=start_rl_with_verl_vllm.sh

# 断点续训重试次数, 默认100次
max_retries=100

# 第一次启动是否需要清空ckpt文件夹: 0 不清理; 1 需要清理
clean_old_ckpt=0
```

### 修改hosts.conf

根据工作模式选择对应的配置方式：

**hybrid 模式（单机）**：修改为单机对应的 IP 地址，以下以 `192.168.0.1` 为例：

```shell
# host,index,train_master_index,infer_master_index(可选)
# 如果单机训练+推理共部署, 则需要配置infer_master_index
192.168.0.1,0,1,1
```

**one-step-off 模式（双机）**：修改为双机对应的 IP 地址，以下以 `192.168.0.1` 和 `192.168.0.2` 为例：

```shell
# host,index,train_master_index,infer_master_index(可选)
# 双机, 训推分离, 分节点部署
192.168.0.1,0,0
192.168.0.2,1,1
```

### 修改环境变量配置

#### 配置 DEFAULT_SOCKET_IFNAME

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

3. 假设本地 IP 为 192.168.0.1，那么指向本地 IP 对应网络接口的值即为 enp189s0f0 ，即需要执行：

    ```shell
    export DEFAULT_SOCKET_IFNAME=enp189s0f0
    ```

#### 配置 ASCEND_RT_VISIBLE_DEVICES

根据自己实际需要设置可用的 NPU 的卡数，例如需要指定 0-15 号 NPU：

```shell
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
```

### 启动方式

```shell
# 进入自己的工作目录
cd /home/work/AgentSDK/aura

bash scripts/start_rl_with_verl_vllm.sh
```

> [!NOTE]
>
>- 请确保模型权重路径，Aura 安装路径及所有文件的属主与运行用户一致。
>- 请确保路径不为软链接。
>- 请确保路径为本地绝对路径。
>- 请确保路径权限为 750，文件为 640。
>- 请确保模型文件来源可信，文件未被篡改，且已完成了训练模型转换和数据集处理。如果模型来源不可靠，可能会发生 torch.load 导致的序列化问题。
>- 分离多机器模式下请将代码、权重均保存在共享盘内，保证数据可以同时被所有机器获取

## **后续步骤**

**Aura 使用样例请参考[使用指南](04_user_guide/01_user_guide.md)**

**Aura Qwen3-4B Math 场景一键拉起样例请参考[Qwen3-4B Math 场景样例](models/qwen3-4b.md)**

**Aura Qwen3-8B Math 场景一键拉起样例请参考[Qwen3-8B Math 场景样例](models/qwen3_8b.md)**

**Aura Qwen3-30B-A3B Math 场景一键拉起样例请参考[Qwen3-30B-A3B Math 场景样例](models/qwen3-30b-a3b.md)**

**Aura Qwen3-14B Math 场景一键拉起样例请参考[Qwen3-14B Math 场景样例](models/qwen3-14b_quick_start/qwen3-14b-hybrid.md)**

**Aura Qwen3-32B Math 场景一键拉起样例请参考[Qwen3-32B Math 场景样例](models/qwen3_32b.md)**

**Aura 支持的后端与模型列表请参考[支持推理后端](10_appendix.md#支持的推理后端)，[支持训练后端](10_appendix.md#支持的训练后端)，[支持agent后端](10_appendix.md#支持的Agent后端)，[支持模型列表](10_appendix.md#支持的模型列表)**

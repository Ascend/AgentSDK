# Qwen3-4B 共卡模式快速拉起指南

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

```shell
# 处理数据集
python3 gsm8k.py \
    --local_dataset_path /path/to/gsm8k \
    --local_save_dir /path/to/gsm8k-output
```

## **文件修改**

在快速拉起 qwen3-4b math 场景前，需要您修改以下配置文件，需要进行修改的参数可以参照文件头的注释，请将其中的示例路径修改为您自己的实际路径。

1. [共卡训练配置文件](../../../../../aura/configs/train/verl_train_hybrid_A3_t16_qwen3_4b_math_fsdp.yaml)
2. [共卡推理配置文件](../../../../../aura/configs/infer/vllm_infer_i16_qwen3_4b.yaml)

### 修改hosts.conf

修改为单机对应的 IP 地址，以下以 `192.168.0.1` 为例：

```shell
# [单机训练+推理]
# 单机，训推共节点部署, 方便本地调测
# host,index,train_master_index,infer_master_index(可选)
192.168.0.1,0,1,1
```

### 修改base.conf

```shell
# [train]
# 启动训练相关参数
# 工作模式：hybrid 共卡模式 | one_step_off 全异步分离模式
work_mode=hybrid

# 共卡和分离模式均需要配置训练yaml文件
train_config_name=verl_train_hybrid_A3_t16_qwen3_4b_math_fsdp

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
> 本文档专门针对 Qwen3-4B 共卡模式（hybrid）的启动示例，提供该场景下的快速拉起步骤；通用且详细的完整启动流程请参考 [快速入门指南](../../03_quick_start.md)。

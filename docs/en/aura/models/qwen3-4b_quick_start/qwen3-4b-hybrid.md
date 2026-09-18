# Quick Startup Guide for Qwen3-4B in Hybrid Mode

## Container Environment Deployment

This document uses an Atlas A3 server with 16 devices running Ubuntu as an example.

**Step 1: Pull the prebuilt image.**

```shell
docker pull swr.cn-south-1.myhuaweicloud.com/ascendhub/agentsdk:26.1.0-cann9.0.0-torch_npu2.9.0-a3-ubuntu22.04-py3.11
```

**Step 2: Create a container.**

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
    swr.cn-south-1.myhuaweicloud.com/ascendhub/agentsdk:26.1.0-cann9.0.0-torch_npu2.9.0-a3-ubuntu22.04-py3.11  \
    sleep infinity
```

**Step 3: Enter the container environment.**

```shell
docker exec -it your_container_name bash
```

## Model Acquisition

This experiment uses the Qwen3-4B model. You can obtain the model from the [ModelScope Qwen3-4B model repository](https://www.modelscope.cn/models/Qwen/Qwen3-4B).

```shell
modelscope download --model Qwen/Qwen3-4B --local_dir /path/to/Qwen3-4B
```

## Dataset Acquisition

### Downloading the Dataset

This experiment uses the GSM8K dataset in the math domain. You can obtain the dataset from the [ModelScope GSM8K dataset](https://www.modelscope.cn/datasets/AI-ModelScope/gsm8k).

```shell
modelscope download --dataset AI-ModelScope/gsm8k --local_dir /path/to/gsm8k
```

### Processing the Dataset

Use the [gsm8k.py](https://github.com/verl-project/verl/blob/v0.7.0/examples/data_preprocess/gsm8k.py) data processing script provided by the official verl project to process the dataset.

```shell
# Process the dataset
python3 gsm8k.py \
    --local_dataset_path /path/to/gsm8k \
    --local_save_dir /path/to/gsm8k-output
```

## Modifying Files

Before starting the Qwen3-4B math scenario, modify the following configuration files. Refer to the comments at the beginning of each file for the parameters that need to be modified, and replace the example paths with the actual paths.

1. [Hybrid training configuration file](../../../../../aura/configs/train/verl_train_hybrid_A3_t16_qwen3_4b_math_fsdp.yaml)
2. [Hybrid inference configuration file](../../../../../aura/configs/infer/vllm_infer_i16_qwen3_4b.yaml)

### Modifying `hosts.conf`

Set the IP address for the single-node deployment:

```shell
# [Single-node training + inference]
# Single-node deployment with training and inference on the same node for convenient local debugging
# host,index,train_master_index,infer_master_index(optional)
<node_IP>,0,1,1
```

### Modifying `base.conf`

```shell
# [train]
# Training-related startup parameters
# Working mode: hybrid (Hybrid mode) | one_step_off (One-Step-Off mode)
work_mode=hybrid

# Both Hybrid mode and One-Step-Off mode require a training YAML file
train_config_name=verl_train_hybrid_A3_t16_qwen3_4b_math_fsdp

# One-Step-Off mode requires a separate inference YAML file
infer_config_name=vllm_infer_i16_qwen3_4b

# [resume]
# Startup parameters for resuming training from a checkpoint
# Startup script to monitor: start_rl_with_msrl_vllm.sh | start_rl_with_verl_vllm.sh
monitor_cmd=start_rl_with_verl_vllm.sh

# Number of retries for resuming training from a checkpoint. The default is 100.
max_retries=100

# Whether to clear the ckpt folder on the first startup: 0: do not clear; 1: clear
clean_old_ckpt=0
```

## Configuring Environment Variables

### Configuring `DEFAULT_SOCKET_IFNAME`

Set the name of the network interface associated with the correct local IP address.

1. Run the `ifconfig` command to view the network configuration:

    ```shell
    ifconfig
    ```

2. Suppose the output includes the following:

    ```text
    docker0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
            inet 172.17.0.1  netmask 255.255.0.0  broadcast 172.17.255.255

    enp189s0f0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
            inet 192.168.0.1  netmask 255.255.0.0  broadcast 192.168.255.255

    enp189s0f1: flags=4099<UP,BROADCAST,MULTICAST>  mtu 1500
            inet 192.168.100.100  netmask 255.255.255.0  broadcast 192.168.100.255
    ```

3. Suppose the local IP address is `192.168.0.1`. The network interface corresponding to the local IP address is `enp189s0f0`. Therefore, run the following command:

   ```shell
   export DEFAULT_SOCKET_IFNAME=enp189s0f0
   ```

### Configuring `ASCEND_RT_VISIBLE_DEVICES`

Configure the available NPU devices.

```shell
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
```

## Starting Training

Start the training script.

```shell
# Enter your working directory
cd /home/work/AgentSDK/aura
# Start the training script
bash scripts/start_rl_with_verl_vllm.sh
```

> [!NOTE]
> This document provides a quick startup example specifically for Qwen3-4B in Hybrid mode. For the complete and general startup procedure, see [Quick Start](../../03_quick_start.md).

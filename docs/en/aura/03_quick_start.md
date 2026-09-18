# Quick Start

## Introduction

Aura is started using the `start_rl_with_verl_vllm.sh` script. This document describes how to use the script and helps you get familiar with Aura.

## Environment Setup

Aura provides two ways to set up the runtime environment. Choose either method as needed:

1. **Create a container using a pre-built image** (recommended)

    Run a container directly from the pre-built image. See [Building the Container Environment from an Image](02_installation_guide.md#method-1-building-the-container-environment-from-an-image).

2. **Use the environment configuration script in a CANN 9.0.0 container**

    If you already have a container based on a CANN 9.0.0 base image, run the environment configuration script to install Aura and all its dependencies, including vLLM, vllm-ascend, MindSpeed, Megatron-LM, veRL, and transformers. For details, see [Using the Environment Configuration Script `build_env.sh`](02_installation_guide.md#method-2-using-the-environment-configuration-script-build_envsh).

## Usage Process

Aura provides a model training example.

- Prepare the model weights and training data. See [Model Weight Preparation](02_installation_guide.md#model-weight-preparation) and [Training Data Preparation](02_installation_guide.md#training-data-preparation).
- Configure the environment variables. See [Environment Variable Configuration](02_installation_guide.md#environment-variable-configuration).
- Modify the path parameters in the YAML configuration file as needed. For a complete example, see [Configuration File Example](05_api_python.md#configuration-file-example).
- Modify the `hosts.conf` and `base.conf` configuration files as needed. For parameter descriptions, see [Service Startup](05_api_python.md#service-startup).

### Modifying `base.conf`

Select `work_mode` based on the training mode, and select the corresponding `train_config_name` and `infer_config_name` based on the model. The following sample shows how to configure the files and start Aura with Qwen3-32B in the math scenario in One-Step-Off mode:

```shell
# [train]
# Training-related startup parameters
# Training mode: hybrid | one_step_off
work_mode=one_step_off

# Both Hybrid and One-Step-Off modes require a training YAML configuration file.
train_config_name=verl_train_async_t16_qwen3_32B_math

# One-Step-Off mode requires a separate inference YAML configuration file. This parameter is not used in Hybrid mode.
infer_config_name=vllm_infer_i16_qwen3_32b

# [resume]
# Checkpoint resume parameters
# Startup script to monitor: start_rl_with_msrl_vllm.sh | start_rl_with_verl_vllm.sh
monitor_cmd=start_rl_with_verl_vllm.sh

# Number of checkpoint resume retries. The default value is 100.
max_retries=100

# Whether to clear the ckpt directory on the first startup: 0 = no; 1 = yes.
clean_old_ckpt=0
```

### Modifying `hosts.conf`

Select the appropriate configuration based on the training mode.

**Hybrid mode (single-node)**: Set the IP address for the node. The following example uses `192.168.0.1`.

```shell
# host, index, train_master_index, infer_master_index (optional)
# Configure infer_master_index when training and inference are deployed on the same node.
192.168.0.1,0,1,1
```

**One-Step-Off mode (two-node)**: Set the IP addresses for the two nodes. The following example uses `192.168.0.1` and `192.168.0.2`.

```shell
# host, index, train_master_index, infer_master_index (optional)
# Training and inference services deployed separately on two nodes
192.168.0.1,0,0
192.168.0.2,1,1
```

### Modifying the Environment Variable Configuration

#### Setting `DEFAULT_SOCKET_IFNAME`

Set the name of the network interface that has the correct local IP address.

1. Run the `ifconfig` command to view the network configuration:

    ```shell
    ifconfig
    ```

2. Assume that the output is as follows (partial):

    ```text
    docker0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
           inet 172.17.0.1  netmask 255.255.0.0  broadcast 172.17.255.255

    enp189s0f0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
           inet 192.168.0.1  netmask 255.255.0.0  broadcast 192.168.255.255

    enp189s0f1: flags=4099<UP,BROADCAST,MULTICAST>  mtu 1500
           inet 192.168.100.100  netmask 255.255.255.0  broadcast 192.168.100.255
    ```

3. Assuming that the local IP is `192.168.0.1`, the corresponding network interface is `enp189s0f0`. Run the following command:

    ```shell
    export DEFAULT_SOCKET_IFNAME=enp189s0f0
    ```

#### Setting `ASCEND_RT_VISIBLE_DEVICES`

Set the available NPUs as needed. The following example shows how to specify NPUs 0 through 15:

```shell
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
```

### Starting Aura

```shell
# Enter your working directory
cd /home/work/AgentSDK/aura

bash scripts/start_rl_with_verl_vllm.sh
```

> [!NOTE]
>
>- Ensure that the model weight path, Aura installation path, and all files are owned by the user running Aura.
>- Ensure that the paths are not symbolic links.
>- Ensure that the paths are local absolute paths.
>- Ensure that the directory permissions are `750` and the file permissions are `640`.
>- Ensure that the model files come from a trusted source, have not been tampered with, and that model conversion and dataset processing have been completed. If the model source is untrusted, serialization issues may occur when `torch.load` is called.
>- In One-Step-Off multi-node mode, store the code and model weights on shared storage to ensure that all nodes can access them simultaneously.

## What to Do Next

**For Aura usage examples, see [User Guide](04_user_guide/01_user_guide.md).**

**For an example of starting Aura in the Qwen3-4B Math scenario, see [Qwen3-4B Math Scenario Example](models/qwen3-4b.md).**

**For an example of starting Aura in the Qwen3-8B Math scenario, see [Qwen3-8B Math Scenario Example](models/qwen3_8b.md).**

**For an example of starting Aura in the Qwen3-30B-A3B Math scenario, see [Qwen3-30B-A3B Math Scenario Example](models/qwen3-30b-a3b.md).**

**For an example of starting Aura in the Qwen3-14B Math scenario, see [Qwen3-14B Math Scenario Example](models/qwen3-14b_quick_start/qwen3-14b-hybrid.md).**

**For an example of starting Aura in the Qwen3-32B Math scenario, see [Qwen3-32B Math Scenario Example](models/qwen3_32b.md).**

**For supported Aura backends and models, see [Supported Inference Backends](10_appendix.md#supported-inference-backends), [Supported Training Backends](10_appendix.md#supported-training-backends), [Supported Agent Backends](10_appendix.md#supported-agent-backends), and [Supported Models](10_appendix.md#supported-models).**

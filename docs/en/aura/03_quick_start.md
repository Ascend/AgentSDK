# Quick Start

## **Introduction**

You start Aura with the `start_rl_with_verl_vllm.sh` script. This section describes how to use the script to help you become familiar with the software.

## **Environment Preparation**

Aura provides two methods for building the runtime environment. You can choose either one based on your actual situation:

1. **Create a container from a prebuilt image** (Recommended)

    Run the container directly from the prebuilt image. For details, see [Building a Container Environment from an Image](02_installation_guide.md#method-1-building-a-container-environment-from-an-image).

2. **Use the one-click startup script in a CANN 9.0.0 container**

    If you are already in a container based on the CANN 9.0.0 image, you can run the one-click startup script to install Aura and all its dependencies (vLLM, vllm-ascend, MindSpeed, Megatron-LM, verl, transformers, and so on). For details, see [Using the One-Click Environment Configuration Script `build_env.sh`](02_installation_guide.md#method-2-using-the-one-click-environment-configuration-script-build_envsh).

> [!NOTE]
> The code in the prebuilt image is a snapshot taken when the image was built and may differ from the latest code in the release branch of the repository. After you enter the container, you are advised to first check whether the latest commit of the local branch matches that of the corresponding remote branch. If they differ, pull the latest code to update it, and then configure and start Aura as described in this document.
>
> In multi-node scenarios, you must manually pull the latest code in the shared directory.

## **Usage Process**

Aura provides examples for training models.

- Prepare the training model and dataset. For details, see [Preparing Model Weights](02_installation_guide.md#preparing-model-weights) and [Preparing Training Data](02_installation_guide.md#preparing-training-data).
- Configure environment variables. For details, see [Environment Variable Configuration](02_installation_guide.md#environment-variable-configuration).
- Modify the path parameters in the YAML configuration file based on your actual environment. For a complete example, see [Configuration File Example](05_api_python.md#configuration-file-example).
- Modify the `hosts.conf` and `base.conf` configuration files based on your actual environment. For descriptions of the related parameters, see [Service Startup Instructions](05_api_python.md#service-startup-instructions).

### Modifying `base.conf`

Select `work_mode` based on the working mode, and select the corresponding `train_config_name` and `infer_config_name` based on the model you use. The following example uses the Qwen3-32B model and starts disaggregation mode for a math scenario. Set the configuration file and the startup method as follows:

```shell
# [train]
# Parameters for starting training
# Working mode: hybrid co-located mode | one_step_off single-step asynchronous disaggregation mode | fully_async fully asynchronous disaggregation mode
work_mode=one_step_off

# Both co-located mode and disaggregation mode require a training YAML file
train_config_name=verl_train_async_A3_t16_qwen3_32b_math_fsdp

# Disaggregation mode requires a separate inference YAML file. This configuration does not take effect in co-located mode
infer_config_name=vllm_infer_i16_qwen3_32b

# [resume]
# Parameters for starting training resumption
# Startup script to monitor: start_rl_with_msrl_vllm.sh | start_rl_with_verl_vllm.sh
monitor_cmd=start_rl_with_verl_vllm.sh

# Number of retries for training resumption, default: 100
max_retries=100

# Whether to clear the ckpt folder on the first startup: 0 no cleanup, 1 cleanup required
clean_old_ckpt=0
```

### Modifying `hosts.conf`

Select the corresponding configuration method based on the working mode:

**Co-located Mode (Hybrid), Single Node**: Change it to the IP address of the single node. The following example uses 192.168.0.1:

```shell
# host,index,train_master_index,infer_master_index(optional)
# If training and inference are co-deployed on a single node, configure infer_master_index
192.168.0.1,0,1,1
```

**Disaggregation Mode (One Step Off and Fully Async), Two Nodes**: Change it to the IP addresses of the two nodes. The following example uses 192.168.0.1 and 192.168.0.2:

```shell
# host,index,train_master_index,infer_master_index(optional)
# Two nodes, with training and inference separated, deployed by node
192.168.0.1,0,0
192.168.0.2,1,1
```

### Modifying the Environment Variable Configuration

The one-click startup script reads `aura/configs/env/env.conf` by default. When you modify settings such as the network interface and the visible NPU devices for your local environment, you are advised to override the default values in `aura/configs/env/env.local`. For temporary debugging, you can also `export` them in the shell first, because values exported this way have the highest priority. Write one `KEY=VALUE` per line in `env.local`, and you do not need to add `export`. Training and inference service parameters are still configured in the corresponding YAML file. For the complete loading priority and Python path configuration, see [Environment Variable Configuration](02_installation_guide.md#environment-variable-configuration).

```shell
touch aura/configs/env/env.local
```

#### Setting `DEFAULT_SOCKET_IFNAME`

The name of the network interface that has the correct local IP address.

1. Run the `ifconfig` command to view the network configuration:

    ```shell
    ifconfig
    ```

2. Assume that part of the output is as follows:

    ```text
    docker0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
            inet 172.17.0.1  netmask 255.255.0.0  broadcast 172.17.255.255

    enp189s0f0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
            inet 192.168.0.1  netmask 255.255.0.0  broadcast 192.168.255.255

    enp189s0f1: flags=4099<UP,BROADCAST,MULTICAST>  mtu 1500
            inet 192.168.100.100  netmask 255.255.255.0  broadcast 192.168.100.255
    ```

3. Assume that the local IP address is 192.168.0.1. Then the value that points to the virtual bridge corresponding to the local IP address is `enp189s0f0`. Write the following into `aura/configs/env/env.local`:

    ```text
    DEFAULT_SOCKET_IFNAME=enp189s0f0
    ```

#### Setting `ASCEND_RT_VISIBLE_DEVICES`

Set the number of available NPUs based on your actual needs. For example, to specify NPUs 0 to 15:

```text
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
```

### Startup Method

```shell
# Go to your own working directory
cd /home/work/AgentSDK/aura

bash scripts/start_rl_with_verl_vllm.sh
```

> [!NOTE]
>
>- Ensure that the owner of the model weight path, the Aura installation path, and all files is the same as the user running the program.
>- Ensure that the paths are not symbolic links.
>- Ensure that the paths are local absolute paths.
>- Ensure that the paths have the permissions 750 and the files have the permissions 640.
>- Ensure that the model files come from a trusted source, have not been tampered with, and that training model conversion and dataset processing are complete. If the model source is unreliable, `torch.load` may cause serialization problems.
>- In multi-node disaggregation mode, save both the code and the weights on the shared drive to ensure that all machines can obtain the data at the same time.

## **Next Steps**

**For Aura usage examples, see the [User Guide](04_user_guide/01_user_guide.md).**

**The scenario modes supported by each Aura model and the corresponding one-click startup examples:**

| Model           | Co-located Mode                                          | Single-Step Asynchronous Disaggregation Mode                   | Mixed Batch Scheduling        | Fully Asynchronous Disaggregation Mode |
|-----------------|----------------------------------------------------------|----------------------------------------------------------------|-------------------------------|-----------------------------|
| Qwen3-4B        | [✓ Supported](models/qwen3-4b_quick_start/qwen3-4b-hybrid.md)   | [✓ Supported](models/qwen3-4b_quick_start/qwen3-4b-one-step-off.md)   | —                             | —                           |
| Qwen3-8B        | [✓ Supported](models/qwen3_8b.md)                               | [✓ Supported](models/qwen3_8b.md)                                     | —                             | —                           |
| Qwen3-14B       | [✓ Supported](models/qwen3-14b_quick_start/qwen3-14b-hybrid.md) | [✓ Supported](models/qwen3-14b_quick_start/qwen3-14b-one-step-off.md) | —                             | —                           |
| Qwen3-30B-A3B   | [✓ Supported](models/qwen3-30b-a3b.md)                          | [✓ Supported](models/qwen3-30b-a3b.md)                                | —                             | —                           |
| Qwen3-32B       | [✓ Supported](models/qwen3_32b.md)                              | [✓ Supported](models/qwen3_32b.md)                                    | —                             | [✓ Supported](models/qwen3_32b.md) |
| Qwen3.6-27B     | [✓ Supported](models/qwen3.6-27b.md)                            | [✓ Supported](models/qwen3.6-27b.md)                                  | [✓ Supported](models/qwen3.6-27b.md) | —                           |
| Qwen3.6-35B-A3B | [✓ Supported](models/qwen3.6-35b-a3b.md)                        | [✓ Supported](models/qwen3.6-35b-a3b.md)                              | —                             | —                           |

> [!NOTE]
> The symbol "—" indicates that this model currently has no one-click startup example for the corresponding mode. "✓ Supported" is a link to the example for the corresponding scenario.

**For the list of backends and models supported by Aura, see [Supported Inference Backends](10_appendix.md#supported-inference-backends), [Supported Training Backends](10_appendix.md#supported-training-backends), [Supported Agent Backends](10_appendix.md#supported-agent-backends), and [Supported Models](10_appendix.md#supported-models).**

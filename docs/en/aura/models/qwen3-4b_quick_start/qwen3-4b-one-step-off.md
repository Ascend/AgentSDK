# Qwen3-4B Single-Step Asynchronous Disaggregation Mode (One Step Off) Quick Start Guide

## **Container Environment Deployment**

This document uses an Atlas 900 A3 SuperPoD supernode (16 × 64 GB) running Ubuntu as an example.

Step 1: Pull the prebuilt image:

```shell
docker pull swr.cn-south-1.myhuaweicloud.com/ascendhub/agentsdk:26.2.0-cann9.0.0-torch_npu2.10.0-a3-ubuntu22.04-py3.11
```

Step 2: Create a container:

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
    swr.cn-south-1.myhuaweicloud.com/ascendhub/agentsdk:26.2.0-cann9.0.0-torch_npu2.10.0-a3-ubuntu22.04-py3.11  \
    sleep infinity
```

Step 3: Enter the container environment.

```shell
docker exec -it your_container_name bash
```

Step 4: Update the code (recommended).

The code in the prebuilt image is a snapshot taken at the time the image was built. Therefore, it may be inconsistent with the latest code on the release branch of the code repository. After you enter the container, you are advised to first check whether the latest commit of the local branch matches that of the corresponding remote branch, and to pull the latest code to update it if they do not match.

> [!NOTE]
> In a multi-node scenario, you need to pull the latest code manually in a shared directory.

## **Obtaining the Model**

This experiment uses the Qwen3-4B model. You can obtain the related model from the [ModelScope Qwen3-4B model repository](https://www.modelscope.cn/models/Qwen/Qwen3-4B).

```shell
modelscope download --model Qwen/Qwen3-4B --local_dir /path/to/Qwen3-4B
```

## **Obtaining the Dataset**

### **Downloading the Dataset**

You can obtain the gsm8k dataset of the math domain used in this experiment from the [ModelScope gsm8k dataset](https://www.modelscope.cn/datasets/AI-ModelScope/gsm8k).

```shell
modelscope download --dataset AI-ModelScope/gsm8k --local_dir /path/to/gsm8k
```

### **Processing the Dataset**

Use the official data processing script [gsm8k.py](https://github.com/verl-project/verl/blob/v0.7.0/examples/data_preprocess/gsm8k.py) provided by verl to process the dataset:

```shell
# Process the dataset
python3 gsm8k.py \
    --local_dataset_path /path/to/gsm8k \
    --local_save_dir /path/to/gsm8k-output
```

### **Data Format Conversion**

In disaggregation mode, you must convert the dataset to the bin/idx format. The training set and the test set are processed in the same way. The following uses the training set as an example. The conversion process is as follows:

```text
parquet → jsonl → bin/idx
```

#### **Converting parquet to jsonl**

Use the conversion script [`convert_data.py`](../../../../../aura/cli/convert_data.py) provided in the code repository, and specify the absolute paths of the input parquet file and the output jsonl file with the `--input` and `--output` parameters. For complete data processing instructions, see the [Installation Guide](../../02_installation_guide.md#disaggregation-mode-data-processing).

```shell
cd /path/to/AgentSDK/aura/cli
python convert_data.py \
    --input /path/to/gsm8k-parquet/train.parquet \
    --output /path/to/gsm8k-jsonl/train.jsonl
```

#### **Converting jsonl to bin/idx**

**Preparing the Configuration File**

The data processing configuration file for the gsm8k dataset is already available at [aura/configs/datasets/gsm8k.yaml](../../../../../aura/configs/datasets/gsm8k.yaml). You can directly modify the path parameters in the file:

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

**Configuration Parameters**

| Parameter                        | Type   | Description                                                                               |
|---------------------------|------|----------------------------------------------------------------------------------|
| `input`                   | str  | Path to the directory containing the input jsonl files, which includes `train.jsonl` and `test.jsonl`. |
| `tokenizer_name_or_path`  | str  | Tokenizer path. It must match the model used in subsequent training. |
| `output_prefix`           | str  | `/path/to/output/train` is the output file path, which you must create in advance. `rl` is the output file prefix. |
| `handler_name`            | str  | Name of the data handler, which determines the data concatenation template. `R1AlpacaStyleInstructionHandler` is the standard format for SFT/RL training of models such as Qwen. |
| `tokenizer_type`          | str  | Tokenizer type. `HuggingFaceTokenizer` is commonly used. |
| `workers`                 | int  | Number of parallel workers |
| `log_interval`            | int  | Log output interval (the number of data records processed before a log is output) |
| `prompt_type`             | str  | Chat template for the model, for example, qwen/chatml/llama3 |
| `dataset_additional_keys` | list | Data fields to be additionally retained |
| `map_keys`                | dict | Field mapping that maps the original JSON fields to the standard fields inside the framework. |

**Running Data Processing**

```shell
# Process the training set
python3 /path/to/AgentSDK/aura/cli/preprocess_data.py gsm8k
```

**Generated Files**

After processing is complete, the following files are generated:

```text
/path/to/output/
├── rl_packed_attention_mask_document.bin    # Attention mask binary data
├── rl_packed_attention_mask_document.idx    # Attention mask index file
├── rl_packed_input_ids_document.bin         # Input ids binary data
├── rl_packed_input_ids_document.idx         # Input ids index file
├── rl_packed_labels_document.bin            # Labels binary data
└── rl_packed_labels_document.idx            # Labels index file
```

## **Modifying Files**

Before you get started with the Qwen3-4B Math scenario, you must modify the following configuration files. The configuration items that you must modify are summarized as follows. For the meaning of the parameters in a YAML file, see the comments at the beginning of the corresponding file:

| # | Configuration File | Required Parameters | Description |
|---|---------|---------|------|
| 1 | aura/configs/datasets/gsm8k.yaml | `input`, `tokenizer_name_or_path`, and `output_prefix` | Used for data format conversion (disaggregation mode requires bin/idx data) |
| 2 | aura/configs/train/verl_train_async_A3_t16_qwen3_4b_math_fsdp.yaml | `hydra.searchpath`, `verl_conf.extras.data_loader.train_data_path`, and `verl_conf.actor_rollout_ref.model.path` | Dataset path and model weight path for disaggregation mode |
| 3 | aura/configs/infer/vllm_infer_i16_qwen3_4b.yaml | `infer_model_path` | Model weight path on the inference side |
| 4 | aura/configs/hosts.conf | `host`, `index`, and `train_master_index` | Node IP addresses and node roles. See the following sections. |
| 5 | aura/configs/base.conf | `work_mode`, `train_config_name`, and `infer_config_name` | See the following sections. |
| 6 | aura/configs/env/env.local | `DEFAULT_SOCKET_IFNAME` and `ASCEND_RT_VISIBLE_DEVICES` | See the following sections. |

### Modifying the Training/Inference Configuration File

For the meaning of the parameters, see the comments at the beginning of the file. In the training configuration, you must modify at least the following paths (keep the remaining parameters at their default values):

- [Single-step asynchronous disaggregation training configuration file](../../../../../aura/configs/train/verl_train_async_A3_t16_qwen3_4b_math_fsdp.yaml)

```yaml
hydra:
  searchpath:
    - file:///verl/verl/trainer/config
    - file:///path/to/AgentSDK/aura/configs/train/verl_conf # Change this to the absolute path of the aura/configs/train/verl_conf directory in this code repository

verl_conf:
  extras:
    data_loader:
      train_data_path: /path/to/data/math_train_dataset/rl # Disaggregation mode dataset path (bin/idx). The path must end with /rl
  actor_rollout_ref:
    model:
      path: /path/to/models/Qwen3-4B # Model weight path
```

> [!NOTE]
> `weight_save_dir` defaults to `aura/weights` in the code directory. In multi-node disaggregation mode, save both the code and the weights on the shared drive. You do not need to modify this parameter.

In the inference configuration, you only need to modify the model weight path:

- [Single-step asynchronous disaggregation inference configuration file](../../../../../aura/configs/infer/vllm_infer_i16_qwen3_4b.yaml)

```yaml
infer_model_path: /path/to/models/Qwen3-4B # Model weight path on the inference side
```

### Modifying `hosts.conf`

- [hosts.conf](../../../../../aura/configs/hosts.conf)

Change this to the IP addresses of the two nodes:

```shell
# [Multi-node training + inference]
# Two nodes, with training and inference disaggregated, deployed by node
# host,index,train_master_index,infer_master_index(optional)
<node IP address 1>,0,0
<node IP address 2>,1,1
```

### Modifying `base.conf`

- [base.conf](../../../../../aura/configs/base.conf)

```shell
# [train]
# Parameters for starting training
# Working mode: hybrid co-located mode | one_step_off single-step asynchronous disaggregation mode | fully_async fully asynchronous disaggregation mode
work_mode=one_step_off

# Both co-located mode and disaggregation mode require a training YAML file
train_config_name=verl_train_async_A3_t16_qwen3_4b_math_fsdp

# Disaggregation mode requires a separate inference YAML file
infer_config_name=vllm_infer_i16_qwen3_4b

# [resume]
# Parameters for starting training resumption
# Startup script to monitor: start_rl_with_msrl_vllm.sh | start_rl_with_verl_vllm.sh
monitor_cmd=start_rl_with_verl_vllm.sh

# Number of retries for training resumption, default: 100
max_retries=100

# Whether to clear the ckpt folder on the first startup: 0 no cleanup, 1 cleanup required
clean_old_ckpt=0
```

## **Environment Variable Configuration**

The one-click startup script reads `aura/configs/env/env.conf` by default. To override variables such as the network interface and the visible NPU devices for your local environment, create `aura/configs/env/env.local` and write the override values in the `KEY=VALUE` format, one per line. You do not need to add `export`. For temporary debugging, you can also `export` them in the shell first, because values exported this way have the highest priority. Training and inference service parameters are still configured in the corresponding YAML files. For complete details, see [Environment Variable Configuration](../../02_installation_guide.md#environment-variable-configuration).

```shell
touch aura/configs/env/env.local
```

### Setting `DEFAULT_SOCKET_IFNAME`

Name of the network interface that contains the correct local IP address.

1. Run the `ifconfig` command to view the network configuration:

    ```shell
    ifconfig
    ```

2. Assume that the printed information (partial) is as follows:

    ```text
    docker0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
            inet 172.17.0.1  netmask 255.255.0.0  broadcast 172.17.255.255

    enp189s0f0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
            inet 192.168.0.1  netmask 255.255.0.0  broadcast 192.168.255.255

    enp189s0f1: flags=4099<UP,BROADCAST,MULTICAST>  mtu 1500
            inet 192.168.100.100  netmask 255.255.255.0  broadcast 192.168.100.255
    ```

3. Assume that the local IP address is 192.168.0.1. The value that points to the virtual bridge corresponding to the local IP address is `enp189s0f0`. Therefore, write the following line to `aura/configs/env/env.local`:

    ```text
    DEFAULT_SOCKET_IFNAME=enp189s0f0
    ```

### Setting `ASCEND_RT_VISIBLE_DEVICES`

Set the number of available NPUs.

```text
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
```

## **Starting Training**

Start the training script.

```shell
# Enter your working directory
cd /home/work/AgentSDK/aura
# Start the training script
bash scripts/start_rl_with_verl_vllm.sh
```

> [!NOTE]
> This document provides a startup example specifically for Qwen3-4B single-step asynchronous disaggregation mode (One Step Off) and describes the quick start steps for this scenario. For the general and complete startup process with details, see the [Quick Start Guide](../../03_quick_start.md).

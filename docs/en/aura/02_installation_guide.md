# Installation Guide

Aura currently provides an environment deployment procedure, which consists of the following three main steps:

1. Container environment deployment
2. Model weight preparation
3. Training data preparation

## Container Environment Deployment

There are two ways to deploy the container environment:

1. Build the container environment from an image.
2. Use a CANN 9.0.0 container environment and run the environment configuration script [build_env.sh](../../../docker/aura/build_env.sh).

### Method 1: Building the Container Environment from an Image

To build the container environment from an image, you must first obtain the image. There are two ways to obtain it:

- Pull the pre-built image directly from the Ascend image registry (recommended).
- Build the image from a Dockerfile.

#### Step 1: Obtaining the Image

##### Option 1: Pulling the Pre-built Image Directly

You can pull the pre-built image directly from the Ascend image registry. For details, see the [Agent SDK image release page](https://www.hiascend.com/developer/ascendhub/detail/72825ebadb23432ba55dea3f58e68a69).

Using an A3 server with Ubuntu as an example, run the following command to pull the image:

```shell
docker pull swr.cn-south-1.myhuaweicloud.com/ascendhub/agentsdk:26.1.0-cann9.0.0-torch_npu2.9.0-a3-ubuntu22.04-py3.11
```

##### Option 2: Building the Image from a Dockerfile

You can quickly build an image using a Dockerfile. The Dockerfile is available in the [docker](../../../docker) directory. You can modify the path parameters in the Dockerfile as needed.

Clone the Agent SDK source code, go to the `docker` directory, and build the image. [Dockerfile.a3.ubuntu](../../../docker/aura/Dockerfile.a3.ubuntu) is used in the following example:

```shell
git clone https://gitcode.com/Ascend/AgentSDK.git
cd /path/to/AgentSDK/docker/aura
docker build -f Dockerfile.a3.ubuntu -t your_image_name:your_image_tag .
```

Select the appropriate Dockerfile based on your server type.

#### Step 2: Creating the Container

The following code sample shows how to create a container on an Atlas A3 server with 16 NPUs:

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
    your_image_name:your_image_tag \
    sleep infinity
```

> [!NOTE]
>
> 1. Mount a different number of device IDs depending on the number of NPUs. For example, the Atlas A3 server has 16 NPUs, so 16 device IDs must be mounted, with each device ID corresponding to one NPU.
> 2. The default working directory inside the image is `/home/work`. To avoid overwriting the default workspace inside the container or causing permission conflicts, do not mount the entire `/home` directory.

#### Step 3: Entering the Container

```shell
docker exec -it your_container_name bash
```

### Method 2: Using the Environment Configuration Script `build_env.sh`

Before using the environment configuration script, set up a CANN 9.0.0 container environment, including installing the CANN 9.0.0 driver and configuring environment variables. You can modify the installation paths of third-party libraries as needed. The environment configuration script automatically installs Aura and all its dependencies, including third-party libraries such as vLLM, vllm-ascend, MindSpeed, Megatron-LM, veRL, and transformers, as well as Python dependencies.

```shell
cd /path/to/AgentSDK/docker/aura
bash build_env.sh
```

> [!NOTE]
>
> The script `build_env.sh` performs global operations such as `pip install -e .` in the current Python environment and clones multiple repositories to `/home/work`. Therefore, **do not run it in the native Python environment on the host or in a virtual environment that has dependencies for other projects**. You are advised to run it only in a fresh CANN 9.0.0 container. If you need an isolated environment, create a separate virtual environment before running the script.

## Model Weight Preparation

### Downloading Model Weights

This section describes how to download the model weights required by Aura. You can choose an appropriate model as needed. The [Qwen2.5-7B-Instruct](https://www.modelscope.cn/models/Qwen/Qwen2.5-7B-Instruct) model is used as an example.

```shell
modelscope download --model Qwen/Qwen2.5-7B-Instruct --local_dir /path/to/Qwen2.5-7B-Instruct
```

## Training Data Preparation

### Downloading Training Data

This section describes how to obtain training data for Aura. The [`gsm8k`](https://www.modelscope.cn/datasets/AI-ModelScope/gsm8k) dataset is used as an example. It contains both training and test sets.

```shell
# Download the dataset
modelscope download --dataset AI-ModelScope/gsm8k --local_dir /path/to/gsm8k
```

> Note: For the first training run, choose a dataset that matches the model's capabilities. Models with fewer parameters should use simpler datasets to facilitate learning.

### Processing Training Data

Data processing methods differ depending on the training mode.

| Training Mode                 | Data Processing Method | Description                     |
| ----------------------------- | ---------------------- | ------------------------------- |
| **Hybrid**        | Official veRL script   | Data format: parquet            |
| **One-Step-Off** | Aura script            | Data format: Megatron (bin/idx) |

#### Processing Data in Hybrid Mode

Use the official data processing script [gsm8k.py](https://github.com/verl-project/verl/blob/v0.7.0/examples/data_preprocess/gsm8k.py) provided by veRL to process the dataset:

```shell
# Process the dataset
python3 gsm8k.py \
    --local_dataset_path /path/to/gsm8k \
    --local_save_dir /path/to/gsm8k-output
```

#### Processing Data in One-Step-Off Mode

One-Step-Off mode requires converting the dataset to bin/idx format. The processing flow is the same for the training and test sets. The conversion process for the training set is described below:

```text
parquet → jsonl → bin/idx
```

##### Step 1: Processing parquet Data

As in Hybrid mode, use `gsm8k.py` to convert the `gsm8k` dataset into the standard format required for training. For details, see [Processing Data in Hybrid Mode](#processing-data-in-hybrid-mode).

##### Step 2: Converting parquet to jsonl

Use the conversion script [convert_data.py](../../../aura/cli/convert_data.py) provided in the repository to convert parquet to jsonl format. Specify the absolute paths for the input parquet file and output jsonl file using the `--input` and `--output` options:

```shell
cd /path/to/AgentSDK/aura/cli
python convert_data.py --input /path/to/gsm8k-parquet/train.parquet --output /path/to/gsm8k-jsonl/train.jsonl
python convert_data.py --input /path/to/gsm8k-parquet/test.parquet --output /path/to/gsm8k-jsonl/test.jsonl
```

##### Step 3: Converting jsonl to bin/idx

**Preparing the Configuration File**

The data processing configuration file for the `gsm8k` dataset is available at [aura/configs/datasets/gsm8k.yaml](../../../aura/configs/datasets/gsm8k.yaml). You can modify the path parameters in this file:

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

**Configuration Parameters**

| Parameter | Type | Description |
| --- | --- | --- |
| `input` | str | Path to the directory containing the input jsonl files, including `train.jsonl` and `test.jsonl` |
| `tokenizer_name_or_path` | str | Tokenizer path, which must be consistent with the model used for subsequent training |
| `output_prefix` | str | `/path/to/output/train` is the output file path, which must be created in advance, and `rl` is the output file prefix |
| `handler_name` | str | Data processor name that determines the data formatting template. `R1AlpacaStyleInstructionHandler` is the standard format for SFT/RL training of models such as Qwen |
| `tokenizer_type` | str | Tokenizer type, commonly `HuggingFaceTokenizer` |
| `workers` | int | Number of parallel workers |
| `log_interval` | int | Number of records processed between log outputs |
| `prompt_type` | str | Specifies the chat template for the model, such as qwen/chatml/llama3 |
| `dataset_additional_keys` | list | Additional data fields to retain |
| `map_keys` | dict | Field mapping that maps original json fields to the framework's internal standard fields |

**Running Data Processing**

Before running the script, you must enter the `aura` directory because the script needs to import the `third_party` dependencies from the `aura` directory.

```shell
# Enter the aura directory (required because the script needs to import third-party dependencies)
cd /path/to/AgentSDK/aura
# Process the training set
python3 /path/to/AgentSDK/aura/cli/preprocess_data.py gsm8k
```

> **Note**: `gsm8k` is the configuration file name without the `.yaml` suffix. The script automatically loads the corresponding configuration from the `configs/datasets/` directory.

**Generated Files**

After processing, the following files are generated:

```text
/path/to/output/
├── rl_packed_attention_mask_document.bin    # Attention mask binary data
├── rl_packed_attention_mask_document.idx    # Attention mask index file
├── rl_packed_input_ids_document.bin         # Input IDs binary data
├── rl_packed_input_ids_document.idx         # Input IDs index file
├── rl_packed_labels_document.bin            # Labels binary data
└── rl_packed_labels_document.idx            # Labels index file
```

## Environment Variable Configuration

### Setting `DEFAULT_SOCKET_IFNAME`

Use the `ifconfig` command to check your network interface name. Using a local IP address of `192.168.0.1` as an example:

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

3. Assuming that the local IP is `192.168.0.1`, the network interface corresponding to the local IP is `enp189s0f0`. Run:

   ```shell
   export DEFAULT_SOCKET_IFNAME=enp189s0f0
   ```

### Setting `ASCEND_RT_VISIBLE_DEVICES`

Set the available NPUs as needed. The following example shows how to specify NPUs 0 through 15:

```shell
# Configure the available NPUs
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
```

## Uninstallation and Cleanup

When the Aura runtime environment is no longer needed, perform the following steps to clean up containers, images, and related data to release host resources.

### 1. Stopping and Removing the Container

```shell
# View the current containers to confirm the container name
docker ps -a

# Stop the container (replace your_container_name with the actual container name)
docker stop your_container_name

# Remove the container
docker rm your_container_name
```

### 2. Removing the Image

```shell
# View local images to confirm the image name and tag
docker images

# Remove the image (replace <image_name>:<tag> with the actual image name, e.g., aura-a3:26.1.0)
docker rmi <image_name>:<tag>
```

> [!NOTE]
>
> Before removing an image, ensure that no running or stopped containers depend on it. Otherwise, run Step 1 to remove the corresponding containers first.

### 3. Cleaning the Build Cache (Optional)

If the build process generated a large number of intermediate layer images or caches, you can clean them up to reclaim drive space:

```shell
# Clean dangling images (intermediate layer images with <none> tags)
docker image prune -f

# Clean all unused images, containers, networks, and build cache
docker system prune -f
```

> [!CAUTION]
>
> `docker system prune -a` removes all images that are not referenced by any container. Before executing this command, make sure that no other projects depend on these images.

### 4. (Optional) Cleaning Aura-related Data

To completely remove data generated by Aura, delete the following directories:

- Third-party repository clones (such as veRL, vLLM, and Megatron-LM) under the container working directory `/home/work`
- Training outputs, including weights, logs, and trajectory files, typically located at `${hydra:runtime.cwd}/weights` and `${hydra:runtime.cwd}/outputs`
- Rollout data and checkpoint directories on shared storage

---

> The environment deployment process is now complete. See the [Quick Start Guide](./03_quick_start.md) to start using Aura.

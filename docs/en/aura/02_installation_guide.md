# Installation Guide

Currently, Aura provides only the environment deployment process, which consists of the following three main steps:

1. Deploying the container environment
2. Preparing model weights
3. Preparing training data

## Container Environment Deployment

Container environment deployment offers two methods:

1. Building a container environment from an image
2. Running the one-click environment configuration script [build_env.sh](../../../docker/aura/build_env.sh) in a container environment based on CANN 9.0.0

### Method 1: Building a Container Environment from an Image

To build a container environment from an image, you must first obtain an image. There are two ways to obtain an image:

- Pulling a prebuilt image directly from the Ascend image repository (recommended)
- Building an image from a Dockerfile

#### Step 1: Obtaining an Image

##### Option 1: Pulling a Prebuilt Image Directly

You can pull a prebuilt image directly from the Ascend image repository. For details, see the [AgentSDK image release page](https://www.hiascend.com/developer/ascendhub/detail/72825ebadb23432ba55dea3f58e68a69).

The following example uses an Atlas A3 product with the Ubuntu system. Run the following command to pull the image:

```shell
docker pull swr.cn-south-1.myhuaweicloud.com/ascendhub/agentsdk:26.2.0-cann9.0.0-torch_npu2.10.0-a3-ubuntu22.04-py3.11
```

##### Option 2: Building an Image from a Dockerfile

You can quickly build an image from a Dockerfile. The Dockerfile is available in the [docker](../../../docker) directory, and you can modify its path parameters as needed.

Clone the Aura project source code, go to the `docker` directory, and build the image. The following example uses the a3-ubuntu [Dockerfile](../../../docker/aura/Dockerfile.a3.ubuntu):

```shell
git clone https://atomgit.com/Ascend/AgentSDK.git
cd /path/to/AgentSDK/docker/aura
docker build -f Dockerfile.a3.ubuntu -t your_image_name:your_image_tag .
```

Select the corresponding Dockerfile for the server type when you build an image.

#### Step 2: Creating a Container

The following example uses an Atlas 900 A3 SuperPoD supernode (16 × 64 GB) to create a container:

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
> 1. Mount a different number of device IDs depending on the number of NPUs. For example, an Atlas A3 product has 16 NPUs. Therefore, you need to mount 16 device IDs, and each device ID corresponds to one NPU.
> 2. The default working directory in the image is `/home/work`. Therefore, you are advised not to mount the entire `/home` directory, which avoids overwriting the default workspace in the container or causing permission conflicts.
> 3. Disaggregation mode requires mounting an additional shared directory to store code, weights, and data.

#### Step 3: Accessing the Container

```shell
docker exec -it your_container_name bash
```

### Method 2: Using the One-Click Environment Configuration Script `build_env.sh`

Before you use the one-click environment configuration script, you must first prepare a CANN 9.0.0 container environment, including installing the CANN 9.0.0 driver and configuring environment variables. You can modify the installation path of third-party libraries as needed. The one-click environment configuration script automatically installs Aura and all of its dependencies, including third-party library dependencies such as vLLM, vllm-ascend, MindSpeed, Megatron-LM, verl, and transformers, as well as Python-related dependencies.

```shell
cd /path/to/AgentSDK/docker/aura
bash build_env.sh
```

> [!NOTE]
>
> The one-click startup script `build_env.sh` performs operations such as a global `pip install -e .` in the current Python environment and clones multiple repositories to `/home/work`. Therefore, **do not run it in the native Python environment of the host or in a virtual environment that already has dependencies for other projects**. You are advised to use it only in a new CANN 9.0.0 container. If you need an isolated environment, create a dedicated virtual environment first and then run the script.

## Preparing Model Weights

### Downloading Model Weights

This section describes how to download the model weights required by Aura. You can select a suitable model as needed. The following example uses [Qwen2.5-7B-Instruct](https://www.modelscope.cn/models/Qwen/Qwen2.5-7B-Instruct):

```shell
modelscope download --model Qwen/Qwen2.5-7B-Instruct --local_dir /path/to/Qwen2.5-7B-Instruct
```

## Preparing Training Data

### Downloading Training Data

This section describes how to obtain training data for Aura. In the math agent scenario, Aura uses the [gsm8k](https://www.modelscope.cn/datasets/AI-ModelScope/gsm8k) dataset, which contains training set and test set data:

```shell
# Download the dataset
modelscope download --dataset AI-ModelScope/gsm8k --local_dir /path/to/gsm8k
```

> Note: When you train a model for the first time, select a suitable dataset based on the model capability. A model with fewer parameters should use a simpler dataset to facilitate model learning.

### Processing Training Data

The data processing method differs depending on the training mode:

| Training Mode                        | Data Processing Method | Description |
|-----------------------------|-------------|------|
| **Co-located (Hybrid)**           | Official verl script | The data format is parquet. |
| **One-Step Asynchronous Disaggregation (One Step Off)** | Aura script | The data format is Megatron. |
| **Fully Asynchronous Disaggregation (Fully Async)**   | Aura script | The data format is Megatron. |

#### Co-located Mode Data Processing

Use the official data processing script [gsm8k.py](https://github.com/verl-project/verl/blob/v0.7.0/examples/data_preprocess/gsm8k.py) provided by verl to process the dataset:

```shell
# Process the dataset
python3 gsm8k.py \
    --local_dataset_path /path/to/gsm8k \
    --local_save_dir /path/to/gsm8k-output
```

#### Disaggregation Mode Data Processing

Disaggregation mode requires converting the dataset to the bin/idx format. The training set and the test set are processed in the same way. The following example uses the training set. The conversion process is as follows:

```text
parquet → jsonl → bin/idx
```

##### Step 1: Processing the parquet Data Structure

As in co-located mode, use `gsm8k.py` to convert the gsm8k dataset into the standard dataset format required for training. For details, see [Co-located Mode Data Processing](#co-located-mode-data-processing).

##### Step 2: Converting parquet to jsonl

Use the conversion script [`convert_data.py`](../../../aura/cli/convert_data.py) provided in the code repository to convert parquet to the jsonl format. When you run the script, specify the absolute paths of the input parquet file and the output jsonl file with the `--input` and `--output` parameters:

```shell
cd /path/to/AgentSDK/aura/cli
python convert_data.py --input /path/to/gsm8k-parquet/train.parquet --output /path/to/gsm8k-jsonl/train.jsonl
python convert_data.py --input /path/to/gsm8k-parquet/test.parquet --output /path/to/gsm8k-jsonl/test.jsonl
```

##### Step 3: Converting jsonl to bin/idx

**Preparing the Configuration File**

The data processing configuration file for the gsm8k dataset is already available at [aura/configs/datasets/gsm8k.yaml](../../../aura/configs/datasets/gsm8k.yaml). You can directly modify the path parameters in the file:

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

**Parameters**

| Parameter                        | Type   | Description                                                                               |
|---------------------------|------|----------------------------------------------------------------------------------|
| input                   | str  | Path to the directory containing the input jsonl files, including `train.jsonl` and `test.jsonl` |
| tokenizer_name_or_path  | str  | Path of the tokenizer. It must match the model used in subsequent training. |
| output_prefix           | str  | `/path/to/output/train` is the output file path, which must be created in advance, and `rl` is the output file prefix. |
| handler_name            | str  | Name of the data handler, which determines the data concatenation template. `R1AlpacaStyleInstructionHandler` is the standard format for SFT/RL training of models such as Qwen. |
| tokenizer_type          | str  | Tokenizer type. `HuggingFaceTokenizer` is commonly used. |
| workers                 | int  | Number of parallel workers |
| log_interval            | int  | Log output interval (the number of data records processed before a log is output) |
| prompt_type             | str  | Chat template for the model, for example, qwen/chatml/llama3 |
| dataset_additional_keys | list | Data fields to be additionally retained |
| map_keys                | dict | Field mapping that maps the original JSON fields to the standard fields inside the framework. |

**Running Data Processing**

Before you run the command, you must go to the `aura` directory, because the script needs to import the `third_party` dependency in the `aura` directory when it runs.

```shell
# Go to the aura directory (required, because the script imports the third_party dependency)
cd /path/to/AgentSDK/aura
# Process the training set
python3 /path/to/AgentSDK/aura/cli/preprocess_data.py gsm8k
```

> **Note**: `gsm8k` is the configuration file name (without the `.yaml` suffix). The script automatically loads the corresponding configuration from the `configs/datasets/` directory.

**Generated Files**

The following files are generated after processing:

```text
/path/to/output/
├── rl_packed_attention_mask_document.bin    # Attention mask binary data
├── rl_packed_attention_mask_document.idx    # Attention mask index file
├── rl_packed_input_ids_document.bin         # Input ids binary data
├── rl_packed_input_ids_document.idx         # Input ids index file
├── rl_packed_labels_document.bin            # Labels binary data
└── rl_packed_labels_document.idx            # Labels index file
```

## Environment Variable Configuration

The Aura one-click startup script uniformly loads the default environment variables from `aura/configs/env/env.conf`. If you need to override the default values for your local environment, you are advised to create an `env.local` file in the same directory. The file takes effect only on the local machine and does not need to be committed to the code repository.

The priority for loading environment variables is as follows: external environment variables that have been `export`ed > `aura/configs/env/env.local` > `aura/configs/env/env.conf` > script default values. For temporary debugging, you can use `export` directly. For long-term fixed local configuration, you are advised to write it to `env.local`. This priority applies only to the stage in which the startup script loads environment variables. Training and inference service parameters still follow the corresponding YAML configuration files, and a configuration with the same name in the inference YAML overrides the environment variable during the subsequent parsing stage.

```shell
touch aura/configs/env/env.local
cat >> aura/configs/env/env.local <<'EOF'
DEFAULT_SOCKET_IFNAME=enp189s0f0
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
EOF
```

`env.local` uses the `KEY=VALUE` format, with one variable per line, and you do not need to add `export`. Variable names must match `[A-Za-z_][A-Za-z0-9_]*`. Otherwise, a warning is generated and the variable is skipped during loading. `PYTHON_HOME` can be left empty by default, and the script automatically derives the installation prefix from the current `python3`. If multiple Python versions exist in the environment, or torch/torch_npu is installed in the specified Python directory, you can explicitly set the Python installation prefix in `env.local`:

```text
PYTHON_HOME=/usr/local/python3.11.14
```

### Setting `DEFAULT_SOCKET_IFNAME`

Use the `ifconfig` command to view your NIC ID. The following example uses the local IP address 192.168.0.1:

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

3. Assume that the local IP address is 192.168.0.1. The value pointing to the virtual bridge that corresponds to the local IP address is then `enp189s0f0`, which you need to write to `env.local`:

    ```text
    DEFAULT_SOCKET_IFNAME=enp189s0f0
    ```

### Setting `ASCEND_RT_VISIBLE_DEVICES`

Set the number of available NPUs as needed. For example, to specify NPUs 0 to 15:

```text
# Configure the number of available NPUs
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
```

## Uninstallation and Cleanup

When you no longer need the Aura runtime environment, perform the following steps to clean up the container, the image, and related data to release host resources.

### 1. Stopping and Deleting the Container

```shell
# View the currently running containers and confirm the container name
docker ps -a

# Stop the container (replace your_container_name with the actual container name)
docker stop your_container_name

# Delete the container
docker rm your_container_name
```

### 2. Deleting the Image

```shell
# View the local images and confirm the image name and tag
docker images

# Delete the image (replace <image_name>:<tag> with the actual image name, for example, aura-a3:26.2.0)
docker rmi <image_name>:<tag>
```

> [!NOTE]
>
> Before you delete an image, ensure that no running or stopped container depends on the image. Otherwise, run the command in step 1 first to delete the corresponding container.

### 3. Cleaning Up the Build Cache (Optional)

If the build process generates a large number of intermediate layer images or caches, you can clean them up as well to reclaim drive space:

```shell
# Clean up dangling images (intermediate layer images with the <none> tag)
docker image prune -f

# Clean up all unused images, containers, networks, and build cache
docker system prune -f
```

> [!CAUTION]
>
> `docker system prune -a` deletes all images that are not referenced by any container. Before you run the command, confirm that other projects do not depend on these images.

### 4. Cleaning Up Aura Data (Optional)

To completely remove the data generated by Aura at runtime, you can delete the following directories:

- Clones of third-party repositories in the `/home/work` working directory of the container (such as verl, vLLM, and Megatron-LM)
- Weights, logs, and trajectory files produced by training, which are located in `${hydra:runtime.cwd}/weights` and `${hydra:runtime.cwd}/outputs` by default
- The rollout data and checkpoint directory in shared storage

---

> The environment deployment process is now complete. To use Aura, see the [quick start document](./03_quick_start.md).

# Quick Startup Guide for Qwen3-4B in One-Step-Off Mode

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

### Converting the Data Format

One-Step-Off mode requires the dataset to be converted to the `bin/idx` format. The training and test datasets are processed in the same way. The following uses the training dataset as an example. The conversion process is as follows:

```text
parquet → jsonl → bin/idx
```

#### Converting `parquet` to `jsonl`

Create a conversion script named `convert_data.py`:

```python
import pandas as pd
import json
import os
import argparse

def convert_parquet_to_filtered_jsonl(input_parquet, output_jsonl):
    """
    Convert a Parquet file to JSONL format and extract specific fields.
    """
    print(f"Reading Parquet file: {input_parquet} ...")

    try:
        df = pd.read_parquet(input_parquet)
        records = df.to_dict('records')
    except Exception as e:
        print(f"Failed to read Parquet file: {e}")
        return

    print(f"Read {len(records)} rows of data. Extracting fields...")

    count = 0
    with open(output_jsonl, 'w', encoding='utf-8') as f_out:
        for data in records:
            try:
                new_data = {
                    "data_source": data.get('data_source'),
                    "question": data['prompt'][0]['content'],
                    "answer": data['reward_model']['ground_truth'],
                    "labels": data['reward_model']['ground_truth']
                }
                f_out.write(json.dumps(new_data, ensure_ascii=False) + '\n')
                count += 1
            except KeyError:
                pass
            except Exception as e:
                print(f"Error processing row: {e}")

    print(f"Processing complete! Successfully extracted {count} records and saved them to: {output_jsonl}")

def main():
    parser = argparse.ArgumentParser(description="Convert Parquet to JSONL")
    parser.add_argument('--input', type=str, required=True, help='Input parquet file path')
    parser.add_argument('--output', type=str, default='output.jsonl', help='Output jsonl file path')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return

    convert_parquet_to_filtered_jsonl(args.input, args.output)

if __name__ == "__main__":
    main()
```

Execute the conversion:

```shell
python convert_data.py --input train.parquet --output train.jsonl
python convert_data.py --input test.parquet --output test.jsonl
```

#### Converting `jsonl` to `bin/idx`

**Preparing the Configuration File**

The data processing configuration file for the GSM8K dataset is available at [aura/configs/datasets/gsm8k.yaml](../../../../../aura/configs/datasets/gsm8k.yaml). You can directly modify the path parameters in the file:

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

| Parameter | Type | Description |
| --------- | ---- | ----------- |
| `input` | `str` | Path to the directory containing the input JSONL files, including `train.jsonl` and `test.jsonl`. |
| `tokenizer_name_or_path` | `str` | Path to the tokenizer. It must be consistent with the model used for subsequent training. |
| `output_prefix` | `str` | `/path/to/output/train` is the output file path and must be created in advance. `rl` is the output file prefix. |
| `handler_name` | `str` | Name of the data handler, which determines the data concatenation template. `R1AlpacaStyleInstructionHandler` is the standard data format for SFT/RL training of models such as Qwen. |
| `tokenizer_type` | `str` | Tokenizer type. `HuggingFaceTokenizer` is commonly used. |
| `workers` | `int` | Number of parallel workers. |
| `log_interval` | `int` | Log output interval, that is, the number of data records processed before a log is output. |
| `prompt_type` | `str` | Chat template corresponding to the model, such as `qwen`, `chatml`, or `llama3`. |
| `dataset_additional_keys` | `list` | Additional data fields to retain. |
| `map_keys` | `dict` | Field mapping that maps the original JSON fields to the standard fields used by the framework. |

**Processing the Dataset**

```shell
# Process the training dataset
python3 /path/to/AgentSDK/aura/cli/preprocess_data.py gsm8k
```

**Generated Files**

After processing is complete, the following files are generated:

```text
/path/to/output/
├── rl_packed_attention_mask_document.bin    # Binary attention mask data
├── rl_packed_attention_mask_document.idx    # Attention mask index file
├── rl_packed_input_ids_document.bin         # Binary input IDs data
├── rl_packed_input_ids_document.idx         # Input IDs index file
├── rl_packed_labels_document.bin            # Binary labels data
└── rl_packed_labels_document.idx            # Labels index file
```

## Modifying Files

Before starting Qwen3-4B in the math scenario, modify the following configuration files. Refer to the comments at the beginning of each file for the parameters that need to be modified, and replace the example paths with the actual paths.

1. [One-Step-Off training configuration file](../../../../../aura/configs/train/verl_train_async_A3_t16_qwen3_4b_math_fsdp.yaml)
2. [One-Step-Off inference configuration file](../../../../../aura/configs/infer/vllm_infer_i16_qwen3_4b.yaml)

### Modifying `hosts.conf`

Set the IP addresses for the two-node deployment:

```shell
# [Multi-node training + inference]
# Two-node deployment with training and inference in One-Step-Off mode with training and inference deployed on separate nodes
# host, index, train_master_index, infer_master_index (optional)
<node_IP1>,0,0
<node_IP2>,1,1
```

### Modifying `base.conf`

```shell
# [train]
# Training-related startup parameters
# Working mode: hybrid (Hybrid mode) | one_step_off (One-Step-Off mode)
work_mode=one_step_off

# Both Hybrid mode and One-Step-Off mode require a training YAML file
train_config_name=verl_train_async_A3_t16_qwen3_4b_math_fsdp

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

2. Assume the output includes the following:

   ```text
   docker0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
           inet 172.17.0.1  netmask 255.255.0.0  broadcast 172.17.255.255

   enp189s0f0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
           inet 192.168.0.1  netmask 255.255.0.0  broadcast 192.168.255.255

   enp189s0f1: flags=4099<UP,BROADCAST,MULTICAST>  mtu 1500
           inet 192.168.100.100  netmask 255.255.255.0  broadcast 192.168.100.255
   ```

3. Assume the local IP address is `192.168.0.1`. The network interface corresponding to the local IP address is `enp189s0f0`. Therefore, run the following command:

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
> This document provides a quick startup example specifically for Qwen3-4B in One-Step-Off mode. For the complete and general startup procedure, see [Quick Start](../../03_quick_start.md).

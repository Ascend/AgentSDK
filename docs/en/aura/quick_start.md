# Quick Start<a name="ZH-CN_TOPIC_0000002459355024"></a>

## **Introduction<a name="section142771553125211"></a>**

Agent SDK provides the `agentic_rl` command. This document explains how to use it and helps you get familiar with the software.

## **Environment Preparation<a name="section543617275526"></a>**

Install Agent SDK and its dependencies. For details, see [Installation and Deployment](installation_guide.md#installation-and-deployment).

## **How to Use<a name="section167395353541"></a>**

Agent SDK provides a training model example.

- Convert the training model as follows:

    ```bash
    # Prepare model weights in advance and adjust the model path as needed
    cd /home/third-party/MindSpeed-LLM
    python3 convert_ckpt.py   \
        --use-mcore-models   \
        --model-type GPT  \
        --load-model-type hf   \
        --save-model-type mg   \
        --target-tensor-parallel-size 4   \
        --target-pipeline-parallel-size 1   \
        --add-qkv-bias   \
        --load-dir /home/models/Qwen2.5-7B-Instruct/  \
        --save-dir /home/models/Qwen2.5-7B-Instruct-mcore/    \
        --tokenizer-model /home/models/Qwen2.5-7B-Instruct/tokenizer.json    \
        --model-type-hf llama2   \
        --params-dtype bf16
    ```

- Process the training dataset as follows:

    ```bash
    mkdir -p /home/datasets/deepscalar/
    cd /home/datasets/deepscalar/
    # Download the dataset
    wget https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset/resolve/main/deepscaler.json
    ```

- After you finish model conversion and dataset download, start training.

    ```bash
    # Enter your working directory
    cd /home/work-dir

    # For multi-node training, start the Ray service first. Example commands:
    # Master node: ray start --head --port {ray_port} --dashboard-host={master_ip} --node-ip-address={current_ip} --dashboard-port={dashboard_port} --resources='{"NPU": {npus_per_node}}'
    # Worker node: ray start --address={master_ip}:{ray_port} --node-ip-address={current_ip} --resources='{"NPU": {npus_per_node}}'
    # Run the command and adjust the configuration file path according to your installation. If you changed the weight or dataset directory in the preceding steps, adjust the configuration file as needed
    agentic_rl --config-path /home/agent-7.3.0/configs/agent-parameters.yaml
    ```

> [!NOTE]
>
>- Ensure that the owner of the model weight path, Agent SDK installation path, and all files is consistent with the running user.
>- Ensure that the path is not a symbolic link.
>- Ensure that the path is a local absolute path.
>- Ensure that the path permissions are `750` and the file permissions are `640`.
>- Ensure that the model files come from a trusted source, have not been tampered with, and have completed model conversion and dataset processing. If the model source is untrusted, serialization issues may occur in `torch.load`.

## **Follow-up Procedure<a name="section167395353541"></a>**

**For Agent usage examples, see [Examples and Guidance](user_guide/user_guide.md)**

**For the list of supported backends and models in Agent SDK, see [Supported Inference Backends], [Supported Training Backends], and [Supported Models]**

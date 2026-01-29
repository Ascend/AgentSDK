# 快速入门<a name="ZH-CN_TOPIC_0000002459355024"></a>

**简介<a name="section142771553125211"></a>**

Agent SDK提供agentic\_rl命令，本章节通过介绍该命令的使用，帮助用户熟悉本软件。

**环境准备<a name="section543617275526"></a>**

安装Agent SDK以及相关依赖，具体安装过程请参见[安装部署](installation_guide.md#安装部署)。

**使用流程<a name="section167395353541"></a>**

Agent SDK提供了训练模型示例。

-   将训练模型进行转换，具体操作如下：

    ```
    # 提前准备模型权重，按情况修改模型路径
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

-   将训练数据集进行处理，具体操作如下：

    ```
    mkdir -p /home/datasets/deepscalar/
    cd /home/datasets/deepscalar/
    # 下载数据集
    wget https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset/resolve/main/deepscaler.json
    
    # 修改 /home/third-party/MindSpeed-RL/configs/datasets/deepscaler.json
    input: /home/datasets/deepscalar/deepscaler.json
    tokenizer_name_or_path: /home/models/Qwen2.5-7B-Instruct/
    output_prefix: /home/datasets/deepscalar/data
    handler_name: R1AlpacaStyleInstructionHandler
    tokenizer_type: HuggingFaceTokenizer
    workers: 8
    log_interval: 1000
    prompt_type: qwen_r1
    map_keys: {"prompt":"problem", "query":"", "response": "answer", "system":""}
    dataset_additional_keys: ["labels"]
    
    # 处理数据集
    cd /home/third-party/MindSpeed-RL/
    bash examples/data/preprocess_data.sh deepscaler
    ```

-   完成训练模型转换和数据集处理后，开始进行训练。

    ```
    # 进入自己的工作目录
    cd /home/work-dir
    
    # 执行命令，按照安装情况修改配置文件路径，如果前面的步骤中修改了权重或者数据集的目录，请酌情修改配置文件
    agentic_rl --config-path /home/agent-7.3.0/configs/agent-parameters.yaml
    ```

> [!NOTE] 说明
>-   请确保模型权重路径，Agent SDK安装路径及所有文件的属主与运行用户一致。
>-   请确保路径不为软链接。
>-   请确保路径为本地绝对路径。
>-   请确保路径权限为750，文件为640。
>-   请确保模型文件来源可信，文件未被篡改，且已完成了训练模型转换和数据集处理。如果模型来源不可靠，可能会发生torch.load导致的序列化问题。


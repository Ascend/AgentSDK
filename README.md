# AgentSDK
-   [简介](#简介)
-   [目录结构](#目录结构)
-   [安装部署](#安装部署)
-   [快速入门](#快速入门)
-   [API参考](#API参考)
-   [FAQ](#FAQ)
-   [安全声明](#安全声明)
-   [免责声明](#免责声明)
-   [License](#License)
-   [建议与交流](#建议与交流)

# 简介

    Agent SDK用来帮助用户快速训练AI智能体。
    - 多种轨迹生成方法，支持插件。
    - 融合调度高效利用显存。


# 目录结构

``` 
│  __init__.py
│
├─base
│  │  __init__.py
│  │
│  ├─log
│  │      loggers.py
│  │      __init__.py
│  │
│  ├─utils
│  │      checker.py
│  │      class_loader.py
│  │      file_utils.py
│  │      get_local_rank.py
│  │      logger_patch.py
│  │      ray_secure_init.py
│  │      __init__.py
│  │
│  └─weight_loaders
│          megatron_weight_loaders.py
│          __init__.py
│
├─configs
│      agentic_rl_config.py
│      ray_env_config.py
│      __init__.py
│
├─data_manager
│      data_manager.py
│      data_registry.py
│      data_transform.py
│      mindspeed_rl_data.py
│      __init__.py
│
├─runner
│  │  runner_worker.py
│  │  __init__.py
│  │
│  ├─agent_engine_wrapper
│  │      base.py
│  │      base_engine_wrapper.py
│  │      __init__.py
│  │
│  └─infer_adapter
│      │  async_server.py
│      │  async_server_base.py
│      │  infer_registry.py
│      │  __init__.py
│      │
│      └─vllm
│          │  base_inference_engine.py
│          │  cache_manager.py
│          │  memory_manager.py
│          │  vllm_async_server.py
│          │  vllm_megatron_weight_loaders.py
│          │  vllm_worker.py
│          │  weight_manager.py
│          │  __init__.py
│          │
│          └─patch
│                  ca_mem_sleep.py
│                  worker_v1_sleep.py
│                  __init__.py
│
└─trainer
    │  main.py
    │  __init__.py
    │
    ├─rollout
    │      rollout_worker.py
    │      __init__.py
    │
    └─train_adapter
        │  __init__.py
        │
        └─mindspeed_rl
            │  agent_grpo_trainer.py
            │  train_agent_grpo.py
            │  __init__.py
            │
            ├─configs
            │      parse_config.py
            │      __init__.py
            │
            ├─patch
            │      compute_utils.py
            │      dataset_mapping.py
            │      get_current_node_ip.py
            │      grpo_actor_loss_func.py
            │      launcher.py
            │      __init__.py
            │
            └─workers
                    actor_hybrid_worker.py
                    integrated_worker.py
                    __init__.py
```

# 安装部署

介绍AgentSDK的安装方式。

## 安装依赖

### 安装Ubuntu系统依赖
| 依赖名称 | 版本建议 | 获取建议 |
| :--- | :--- | :--- |
| Python | 3.10及以上 | 建议通过获取源码包编译安装。 |
| CMake | 4.1.0及以上 | 建议通过包管理模块安装。 |
| Make | 4.3及以上 | 建议通过包管理模块安装。 |
| GCC | 11.4.0及以上 | 建议通过包管理模块安装。 |
| G++ | 11.4.0及以上 | 建议通过包管理模块安装。 |

### 安装NPU驱动固件和CANN

安装前，请参考[CANN 软件安装指南](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1/softwareinst/instg/instg_0000.html?Mode=PmIns&InstallType=local&OS=Ubuntu&Software=cannToolKit)安装CANN开发套件包、昇腾NPU驱动和昇腾NPU固件。
CANN软件提供进程级环境变量设置脚本，供用户在进程中引用，以自动完成环境变量设置。用户进程结束后自动失效。可在程序启动的Shell脚本中使用如下命令设置CANN的相关环境变量，也可通过命令行执行如下命令（以root用户默认安装路径“/usr/local/Ascend”为例）：

```shell
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### 安装开源软件

通过Agent SDK使用mindspeed-rl训练时，需要安装以下开源软件。
参考如下命令，安装指定版本的仓库至指定位置，普通用户需要注意使用拥有权限的路径。

```shell
mkdir -p /home/third-party # 可自定义目录
cd /home/third-party

git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout core_r0.8.0
cd ..

git clone https://github.com/Ascend/MindSpeed.git
cd MindSpeed
git checkout 2.1.0_core_r0.8.0
cd ..

git clone https://github.com/Ascend/MindSpeed-LLM.git
cd MindSpeed-LLM
git checkout 2.1.0
cd ..

git clone https://github.com/Ascend/MindSpeed-RL.git
cd MindSpeed-RL
git checkout v2.2.0
cd ..

git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout v0.9.1
VLLM_TARGET_DEVICE=empty pip3 install -e .
cd ..

pip3 install --ignore-installed --upgrade blinker==1.9.0
git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend
git checkout v0.9.1-dev
pip3 install -e .
cd ..

pip3 install -r MindSpeed/requirements.txt
pip3 install -r MindSpeed-LLM/requirements.txt
pip3 install -r MindSpeed-RL/requirements.txt

# 使能环境变量，根据实际安装的情况调整目录
source /usr/local/Ascend/driver/bin/setenv.bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export PYTHONPATH=$PYTHONPATH:/home/third-party/Megatron-LM/:/home/third-party/MindSpeed/:/home/third-party/MindSpeed-LLM:/home/third-party/MindSpeed-RL
```

### 安装Python软件包依赖

```shell
pip3 install transformers==4.52.3
pip3 install sympy==1.13.1
pip3 install pylatexenc==2.10
pip3 install openai==1.99.6
pip3 install torch==2.5.1
pip3 install torch_npu==2.5.1.post1
pip3 install vertexai==1.64.0
pip3 install sentence_transformers==5.1.0
pip3 install hydra-core==1.3.2
pip3 install regex==2025.8.29
pip3 install tensordict==0.1.2
pip3 install word2number==1.1
pip3 install codetiming==1.4.0
pip3 install torchvision==0.20.1
pip3 install ray==2.42.1
```

## 安装Agent SDK

1.  编译AgentSDK
2.  在根目录执行以下命令获取tar包：

        bash script/compile.sh
3.  解压tar包，执行命令：
    
        tar -zxvf output/Ascend-mindsdk-agentsdk__linux-aarch64.tar.gz -C output
4.  安装whl包

        pip3 install output/agentic_rl-7.0.0-py3-none-any.whl
5.  设置环境变量

        export PATH=$PATH:~/.local/bin

# 快速入门

## 简介

Agent SDK提供agentic_rl命令，本章节通过介绍该命令的使用，帮助用户熟悉本软件。

## 环境准备

安装Agent SDK以及相关依赖，具体安装过程请参见[安装部署](#安装部署)。

## 使用流程

Agent SDK提供了训练模型示例。
- 将训练模型进行转换，具体操作如下：
    ```shell
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
- 将训练数据集进行处理，具体操作如下：
    ```shell
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

- 完成训练模型转换和数据集处理后，创建配置文件：
    ```shell
    # 进入自己的工作目录
    cd /home/work-dir
  
    # 编辑配置文件，按照实际的文件位置进行调整
    vi agentic-parameters.yaml
    
    # 文件内容参考
    tokenizer_name_or_path: /home/models/Qwen2.5-7B-Instruct/
    data_path: /home/datasets/deepscalar/data
    load_params_path: /home/models/Qwen2.5-7B-Instruct-mcore/
    save_params_path: /home/models/Qwen2.5-7B-Instruct-mcore-new/
    train_iters: 1
    agent_name: test_agent
    agent_engine_wrapper_path: /path/to/agent/engine/wrapper
    ```
- 开启训练：
    ```shell
    # 执行命令
    agentic_rl --config-path /home/work-dir/agentic-parameters.yaml
    ```
## 说明

- 请确保模型权重路径，Agent SDK安装路径及所有文件的属主与运行用户一致。
- 请确保路径不为软链接。
- 请确保路径为本地绝对路径。
- 请确保路径权限为750，文件为640。
- 请确保模型文件来源可信，文件未被篡改，且已完成了训练模型转换和数据集处理。如果模型来源不可靠，可能会- 发生torch.load导致的序列化问题。

# 安全声明

- 使用API读取文件时，用户需要保证该文件的owner必须为自己，且权限不高于640，避免发生提权等安全问题。 外部下载的软件代码或程序可能存在风险，功能的安全性需由用户保证。
- 通信矩阵：目前Agent SDK开发套件包不会主动打开或者依赖任意端口，因此不涉及通信矩阵。
- 公网地址详见：Agent SDK的安装包中的网址安装结束后会被清除，并不会访问，不会造成风险.


# 免责声明

- 本仓库代码中包含多个开发分支，这些分支可能包含未完成、实验性或未测试的功能。在正式发布前，这些分支不应被应用于任何生产环境或者依赖关键业务的项目中。请务必使用我们的正式发行版本，以确保代码的稳定性和安全性。
  使用开发分支所导致的任何问题、损失或数据损坏，本项目及其贡献者概不负责。

# License

AgentSDK以Mulan PSL v2许可证许可，对应许可证文本可查阅[LICENSE](LICENSE.md)。

# 建议与交流

欢迎大家为社区做贡献。如果有任何疑问或建议，请提交[issue](https://gitcode.com/Ascend/AgentSDK/issues)，我们会尽快回复。感谢您的支持。
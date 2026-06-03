# 快速入门<a name="ZH-CN_TOPIC_0000002459355024"></a>

## **简介<a name="section142771553125211"></a>**

Agent SDK使用脚本`start_rl_with_verl_vllm.sh`来启动，本章节通过介绍该脚本的使用，帮助用户熟悉本软件。

## **环境准备<a name="section543617275526"></a>**

使用预构建镜像创建容器，具体操作请参见【[容器环境部署](installation_guide.md#ZH-CN_TOPIC_0000002492554173)】。

## **使用流程<a name="section167395353541"></a>**

Agent SDK提供了训练模型示例。

- 准备训练模型和数据集，具体操作请参见【[准备模型权重](installation_guide.md#准备模型权重)
  】与【[准备训练数据](installation_guide.md#准备训练数据)】。
- 根据实际环境修改 YAML 配置文件中的路径参数，完整示例请参见【[配置文件示例](command_api.md#section_config_example)】。
- 根据实际环境修改 hosts.conf 和 base.conf配置文件，完整示例请参见【[配置文件示例](command_api.md)】。

在A3环境上，以模型Qwen3-32B，基于math场景启动分离模式为例，设置配置文件与启动方式如下。

1. hosts.conf

    ```shell
    # host,index,train_master_index,infer_master_index(可选)
    # 如果单机训练+推理共部署, 则需要配置infer_master_index

    # [单机训练+推理]
    # 配置例子1：单机，训推共节点部署, 方便本地调测
    # host,index,train_master_index,infer_master_index(可选)
    # 192.168.0.1,0,1,1

    # [多机训练+推理]
    # 配置例子2：双机, 训推分离, 分节点部署
    # host,index,train_master_index,infer_master_index(可选)
    192.168.0.1,0,0
    192.168.0.2,1,1
    ```

2. base.conf

    ```shell
    # [train]
    # 启动训练相关参数
    # 工作模式：hybrid 共卡模式 | one_step_off 全异步分离模式
    work_mode=one_step_off

    # 共卡和分离模式均需要配置训练yaml文件
    train_config_name=verl_train_async_t16_qwen3_32B_math

    # 分离模式需要单独配置推理yaml文件, 共卡模式该配置不生效
    infer_config_name=vllm_infer_i16_qwen3_32b

    # [resume]
    # 启动断点续训相关参数
    # 需要监控的启动脚本：start_rl_with_msrl_vllm.sh | start_rl_with_verl_vllm.sh
    monitor_cmd=start_rl_with_verl_vllm.sh

    # 断点续训重试次数, 默认100次
    max_retries=100

    # 第一次启动是否需要清空ckpt文件夹: 0 不清理; 1 需要清理
    clean_old_ckpt=0
    ```

3. 启动方式

    ```shell
    # 进入自己的工作目录
    cd /home/work/AgentSDK/aura

    bash scripts/start_rl_with_verl_vllm.sh
    ```

> [!NOTE] 说明
>
>- 请确保模型权重路径，Agent SDK安装路径及所有文件的属主与运行用户一致。
>- 请确保路径不为软链接。
>- 请确保路径为本地绝对路径。
>- 请确保路径权限为750，文件为640。
>- 请确保模型文件来源可信，文件未被篡改，且已完成了训练模型转换和数据集处理。如果模型来源不可靠，可能会发生torch.load导致的序列化问题。
>- 分离多机器模式下请将代码，权重均保存在共享盘内，保证数据可以同时被所有机器获取

## **后续步骤<a name="section167395353541"></a>**

**Agent使用样例请参考[使用指南](user_guide/user_guide.md)**

**Agent Qwen3-4b Math场景一键拉起样例请参考[使用指南](model/qwen3-4b.md)**

**AgentSDK 支持的后端与模型列表请参考[支持推理后端](appendix.md#支持的推理后端)，[支持训练后端](appendix.md#支持的训练后端)，[支持agent后端](appendix.md#支持的agent后端)，[支持模型列表](appendix.md#支持的模型列表)**

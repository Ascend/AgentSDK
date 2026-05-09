# 快速入门<a name="ZH-CN_TOPIC_0000002459355024"></a>

## **简介<a name="section142771553125211"></a>**

Agent SDK使用脚本`run_start_in_local.sh`来启动，本章节通过介绍该脚本的使用，帮助用户熟悉本软件。

## **环境准备<a name="section543617275526"></a>**

使用预构建镜像创建容器，具体操作请参见【[容器环境部署](installation_guide.md#ZH-CN_TOPIC_0000002492554173)】。

## **使用流程<a name="section167395353541"></a>**

Agent SDK提供了训练模型示例。

- 准备训练模型和数据集，具体操作请参见【[准备模型权重](installation_guide.md#准备模型权重)】与【[准备训练数据](installation_guide.md#准备训练数据)】。
- 根据实际环境修改 YAML 配置文件中的路径参数，完整示例请参见【[配置文件示例](command_api.md#section_config_example)】。

完成上述准备操作后，开始进行训练，此处以本地运行单节点的共卡模式为例。

    ```shell
    # 进入自己的工作目录
    cd /home/work/AgentSDK
    
    bash run_start_in_local.sh --config-path base_direct_1node_qwen25_7b_train_hybrid_with_verl_megatron.yaml
    ```

> [!NOTE] 说明
>
>- 请确保模型权重路径，Agent SDK安装路径及所有文件的属主与运行用户一致。
>- 请确保路径不为软链接。
>- 请确保路径为本地绝对路径。
>- 请确保路径权限为750，文件为640。
>- 请确保模型文件来源可信，文件未被篡改，且已完成了训练模型转换和数据集处理。如果模型来源不可靠，可能会发生torch.load导致的序列化问题。

## **后续步骤<a name="section167395353541"></a>**

**Agent使用样例请参考[使用指南](user_guide/user_guide.md)**

**AgentSDK 支持的后端与模型列表请参考[支持推理后端](appendix.md#支持的推理后端)，[支持训练后端](appendix.md#支持的训练后端)，[支持agent后端](appendix.md#支持的agent后端)，[支持模型列表](appendix.md#支持的模型列表)**

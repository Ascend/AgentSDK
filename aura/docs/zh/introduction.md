# 简介<a name="ZH-CN_TOPIC_0000002459514656"></a>

Agent SDK用来帮助用户快速训练AI智能体。

- 多种轨迹生成方法，支持插件。
- 融合调度高效利用显存。

**使用导引<a name="section181186816488"></a>**

如果第一次使用本软件，可以从[快速入门](quick_start.md#快速入门)中的样例开始上手，并确保按照其中步骤准备好相关环境和软件包。

如果对于相关流程已比较熟悉，可以直接跳转到[Python接口说明](./api_python.md#python接口说明)获取需要的函数接口，加速数据处理流程。

# 软件架构<a name="ZH-CN_TOPIC_0000002492554225"></a>

Agent SDK软件架构如[图1](#fig173917397815)所示。

**图 1**  Agent SDK软件架构<a id="fig173917397815"></a>  
![](figures/Agent-SDK软件架构.png "Agent-SDK软件架构")

**表 1**  架构图模块介绍

|模块|说明|
|--|--|
|**外部工具 Plugins**|外部工具插件，如Math、Code等，Agent可调用的外部能力|
|**第三方 Agent 引擎**|支持多种Agent引擎，包括rLLM、Langchain等|
|**Serve 模式**|服务化部署模式，提供HTTP API接口，包含infer_router和agent_router|
|**Rollouter (轨迹生成)**|负责Agent轨迹生成，包含agent_manager、agent_executor、agent_engine_wrapper|
|**Scheduler**|推理请求调度器，支持PD分离和负载均衡|
|**Memory**|轨迹数据持久化存储，使用Episode格式|
|**Trainer (训练任务编排)**|训练任务管理和编排，包含buffer和data_manager|
|**infer_manager / infer_executor**|推理服务管理和执行|
|**rollout_controller**|轨迹生成控制器，协调推理和数据流动|
|**train_manager / train_executor**|训练服务管理和执行|
|**train_controller**|训练控制器，协调训练和数据流动|
|**推理引擎**|支持多种推理引擎，包括vllm-ascend、omni-infer、SGLang|
|**训练引擎**|支持多种训练引擎，包括MindSpeed-RL、verl|
|**RAY 分布式资源管理**|基于Ray的分布式资源管理，支持共卡/分离部署|

# 简介

Agent SDK用来帮助用户快速训练AI智能体。

- 多种轨迹生成方法，支持插件。
- 融合调度高效利用显存。

**使用导引<a name="section181186816488"></a>**

如果第一次使用本软件，可以从[快速入门](03_quick_start.md#快速入门)中的样例开始上手，并确保按照其中步骤准备好相关环境和软件包。

如果对于相关流程已比较熟悉，可以直接跳转到[Python接口说明](05_api_python.md#python接口说明)获取需要的函数接口，加速数据处理流程。

# 软件架构

<a id="fig173917397815"></a>

Agent SDK 软件架构如**图 1**所示。

<div align="center">

**图1 Agent SDK 软件架构**

![](figures/Aura框架架构图.png)

</div>

**表1 架构图模块介绍**

|模块|说明|
|--|--|
|**第三方 Agent 引擎**|支持多种Agent引擎，包括rLLM、Langchain等|
|**Serve (训推通信)**|支持训练侧向推理侧和agent侧的http通信|
|**Rollouter (轨迹生成)**|负责Agent轨迹生成，包含agent_manager、agent_executor、agent_engine_wrapper|
|**Scheduler**|推理请求调度器，支持负载均衡|
|**Memory**|轨迹数据持久化存储，使用Episode格式|
|**Trainer (训练任务编排)**|训练任务管理和编排，通过data_manager协调训练和推理的数据流动。|
|**推理引擎**|支持第三方推理引擎，包括vllm-ascend|
|**训练引擎**|支持第三方训练引擎，包括verl|
|**RAY 分布式资源管理**|基于Ray的分布式资源管理，支持共卡/分离部署|

# 支持特性

- [训推共卡使用指南（On-Policy 策略）](04_user_guide/02_hybrid.md)：训练与推理在同一组卡上通过时分复用协同运行
- [训推分离使用指南（One-Step-Off 策略）](04_user_guide/03_one_step_off.md)：训练与推理在不同节点上并行执行
- [自定义 Agent 接入指南](04_user_guide/04_custom_agent.md)：将自定义 agent 接入 Aura 训练框架

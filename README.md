# AgentSDK

[AgentSDK](https://gitcode.com/Ascend/AgentSDK) 是昇腾亲和的Agent生态仓库，旨在为昇腾NPU基础设施上的Agent应用提供开箱即用的生态工具与框架支持，帮助开发者快速构建和部署AI智能体。

AgentSDK目前涵盖Agentic RL训推调框架、Agent应用开发等多个方向，后续将持续扩展更多Agent生态组件。

## 主要目录结构与说明

| 目录                     | 说明                                                  |
|------------------------|-----------------------------------------------------|
| [aura](./aura)         | 提供Agentic RL训推框架aura，有助于Agentic应用开发者快速构建领域Agentic应用 |
| [presmoke](./presmoke) | 提供AgentSDK的前冒烟测试用例                                   |

### aura 目录

提供Agentic RL训推框架aura，可对接不同训练、推理、agents后端。更多详情请参考 [aura README](./aura/README.md)。

包含以下子目录：

| 子目录                               | 说明                                       |
|-----------------------------------|------------------------------------------|
| [aura](./aura/aura)               | 提供Agentic RL训推框架aura核心代码，可对接不同训练、推理、agents后端 |
| [agents](./aura/agents)           | 提供agent样例（如math_agent、dtn_agent），方便aura对接的agents后端直接使用 |
| [cli](./aura/cli)                 | 提供数据预处理、模型权重转换、推理启动等命令行脚本                |
| [configs](./aura/configs)         | 提供aura使用的各个场景下的配置文件 |
| [dockers](./aura/dockers)         | 提供aura运行所需的Docker镜像环境构建                  |
| [docs](./aura/docs)               | 提供aura相关说明文档，包括安装指南、快速入门、API参考、FAQ等      |
| [scripts](./aura/scripts)         | 提供aura运行和部署相关的辅助脚本，包括日志分析、轨迹可视化等 |
| [tests](./aura/tests)             | 提供aura和agents目录对应的单元测试用例                |
| [third_party](./aura/third_party) | 存放aura所需的三方依赖                            |

### presmoke 目录

提供AgentSDK的前冒烟测试用例，用于在正式部署前验证框架的基本功能和配置正确性。包含以下子目录：

| 子目录                               | 说明                                       |
|-----------------------------------|------------------------------------------|
| [cases](./presmoke/cases)         | 提供冒烟测试用例，包括模块导入验证、配置校验、模式验证等端到端测试        |
| [configs](./presmoke/configs)     | 提供冒烟测试专用的配置文件，覆盖有效和无效配置场景                |

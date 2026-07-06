# AgentSDK

[AgentSDK](https://gitcode.com/Ascend/AgentSDK) 是昇腾亲和的Agent生态仓库，旨在为昇腾NPU基础设施上的Agent应用提供开箱即用的生态工具与框架支持，帮助开发者快速构建和部署AI智能体。

AgentSDK目前涵盖Agentic RL训推调框架、Agent应用开发等多个方向，后续将持续扩展更多Agent生态组件。

## 主要产品

AgentSDK 仓库包含以下两个核心产品，点击进入各自的详细介绍：

| 产品 | 说明 |
|------|------|
| **[Aura](./aura/README.md)** | Agentic RL 训推调一体化框架，支持训推共卡与训推分离模式，对接多种训练、推理、Agent 引擎 |
| **[Openclaw](./openclaw/README.md)** | 基于 OpenClaw 构建的多领域 Agent 框架与服务，集成代码生成、研究分析等垂直领域能力 |

## 其他目录说明

| 目录 | 说明 |
|------|------|
| [docker](./docker) | 提供 AgentSDK 的 Dockerfile 和环境构建脚本，用于快速构建容器镜像和配置运行环境 |
| [docs](./docs) | 提供 AgentSDK 相关说明文档，包括安装指南、快速入门、API 参考、FAQ 等 |
| [pre-commit](./pre-commit) | 提供代码提交前的自动化检查配置，包括拼写检查、代码规范、安全扫描等 |
| [presmoke](./presmoke) | 提供 AgentSDK 的前冒烟测试用例，用于正式部署前验证框架的基本功能和配置正确性 |
| [script](./script) | 提供 AgentSDK 的构建、测试、安装等辅助脚本 |

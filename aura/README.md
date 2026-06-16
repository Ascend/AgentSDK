# AgentSDK

- [AgentSDK](#agentsdk)
- [最新消息](#最新消息)
- [简介](#简介)
- [目录结构](#目录结构)
- [版本说明](#版本说明)
- [兼容性信息](#兼容性信息)
- [环境部署](#环境部署)
- [快速入门](#快速入门)
- [特性介绍](#特性介绍)
  - [训推共卡](#训推共卡)
  - [训推分离](#训推分离)
  - [混合批次调度](#混合批次调度)
- [API参考](#api参考)
- [FAQ](#faq)
- [安全声明](#安全声明)
- [分支维护策略](#分支维护策略)
- [版本维护策略](#版本维护策略)
- [免责声明](#免责声明)
- [License](#license)
- [贡献声明](#贡献声明)
- [建议与交流](#建议与交流)

# 最新消息

- [2026.04.25]: 🚀 发布AgentSDK全新训推调框架aura，支持训推共卡与训推分离模式

# 简介

aura 是一个面向 Agent 场景的训推调一体化框架，通过统一抽象接口兼容多种训练引擎、推理引擎与 Agent 框架，帮助开发者快速构建、训练与部署自己的 Agent。
aura 采用模块化与标准化设计，屏蔽底层框架差异，支持从数据构建、模型训练、能力对齐到在线推理与服务部署的端到端流程。
aura 具备良好的扩展性与生态兼容能力，支持用户灵活接入自定义模型与工具链，加速 Agent 应用落地。

更多详情请查看[简介](docs/zh/introduction.md)。

<div align="center">

[![Zread](https://img.shields.io/badge/Zread-Ask_AI-_.svg?style=flat&color=0052D9&labelColor=000000&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQuOTYxNTYgMS42MDAxSDIuMjQxNTZDMS44ODgxIDEuNjAwMSAxLjYwMTU2IDEuODg2NjQgMS42MDE1NiAyLjI0MDFWNC45NjAxQzEuNjAxNTYgNS4zMTM1NiAxLjg4ODEgNS42MDAxIDIuMjQxNTYgNS42MDAxSDQuOTYxNTZDNS4zMTUwMiA1LjYwMDEgNS42MDE1NiA1LjMxMzU2IDUuNjAxNTYgNC45NjAxVjIuMjQwMUM1LjYwMTU2IDEuODg2NjQgNS4zMTUwMiAxLjYwMDEgNC45NjE1NiAxLjYwMDFaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00Ljk2MTU2IDEwLjM5OTlIMi4yNDE1NkMxLjg4ODEgMTAuMzk5OSAxLjYwMTU2IDEwLjY4NjQgMS42MDE1NiAxMS4wMzk5VjEzLjc1OTlDMS42MDE1NiAxNC4xMTM0IDEuODg4MSAxNC4zOTk5IDIuMjQxNTYgMTQuMzk5OUg0Ljk2MTU2QzUuMzE1MDIgMTQuMzk5OSA1LjYwMTU2IDE0LjExMzQgNS42MDE1NiAxMy43NTk5VjExLjAzOTlDNS42MDE1NiAxMC42ODY0IDUuMzE1MDIgMTAuMzk5OSA0Ljk2MTU2IDEwLjM5OTlaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik0xMy43NTg0IDEuNjAwMUgxMS4wMzg0QzEwLjY4NSAxLjYwMDEgMTAuMzk4NCAxLjg4NjY0IDEwLjM5ODQgMi4yNDAxVjQuOTYwMUMxMC4zOTg0IDUuMzEzNTYgMTAuNjg1IDUuNjAwMSAxMS4wMzg0IDUuNjAwMUgxMy43NTg0QzE0LjExMTkgNS42MDAxIDE0LjM5ODQgNS4zMTM1NiAxNC4zOTg0IDQuOTYwMVYyLjI0MDFDMTQuMzk4NCAxLjg4NjY0IDE0LjExMTkgMS42MDAxIDEzLjc1ODQgMS42MDAxWiIgZmlsbD0iI2ZmZiIvPgo8cGF0aCBkPSJNNCAxMkwxMiA0TDQgMTJaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00IDEyTDEyIDQiIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgo8L3N2Zz4K&logoColor=ffffff)](https://zread.ai/Ascend/AgentSDK)&nbsp;&nbsp;&nbsp;&nbsp;
[![DeepWiki](https://img.shields.io/badge/DeepWiki-Ask_AI-_.svg?style=flat&color=0052D9&labelColor=000000&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAAyCAYAAAAnWDnqAAAAAXNSR0IArs4c6QAAA05JREFUaEPtmUtyEzEQhtWTQyQLHNak2AB7ZnyXZMEjXMGeK/AIi+QuHrMnbChYY7MIh8g01fJoopFb0uhhEqqcbWTp06/uv1saEDv4O3n3dV60RfP947Mm9/SQc0ICFQgzfc4CYZoTPAswgSJCCUJUnAAoRHOAUOcATwbmVLWdGoH//PB8mnKqScAhsD0kYP3j/Yt5LPQe2KvcXmGvRHcDnpxfL2zOYJ1mFwrryWTz0advv1Ut4CJgf5uhDuDj5eUcAUoahrdY/56ebRWeraTjMt/00Sh3UDtjgHtQNHwcRGOC98BJEAEymycmYcWwOprTgcB6VZ5JK5TAJ+fXGLBm3FDAmn6oPPjR4rKCAoJCal2eAiQp2x0vxTPB3ALO2CRkwmDy5WohzBDwSEFKRwPbknEggCPB/imwrycgxX2NzoMCHhPkDwqYMr9tRcP5qNrMZHkVnOjRMWwLCcr8ohBVb1OMjxLwGCvjTikrsBOiA6fNyCrm8V1rP93iVPpwaE+gO0SsWmPiXB+jikdf6SizrT5qKasx5j8ABbHpFTx+vFXp9EnYQmLx02h1QTTrl6eDqxLnGjporxl3NL3agEvXdT0WmEost648sQOYAeJS9Q7bfUVoMGnjo4AZdUMQku50McDcMWcBPvr0SzbTAFDfvJqwLzgxwATnCgnp4wDl6Aa+Ax283gghmj+vj7feE2KBBRMW3FzOpLOADl0Isb5587h/U4gGvkt5v60Z1VLG8BhYjbzRwyQZemwAd6cCR5/XFWLYZRIMpX39AR0tjaGGiGzLVyhse5C9RKC6ai42ppWPKiBagOvaYk8lO7DajerabOZP46Lby5wKjw1HCRx7p9sVMOWGzb/vA1hwiWc6jm3MvQDTogQkiqIhJV0nBQBTU+3okKCFDy9WwferkHjtxib7t3xIUQtHxnIwtx4mpg26/HfwVNVDb4oI9RHmx5WGelRVlrtiw43zboCLaxv46AZeB3IlTkwouebTr1y2NjSpHz68WNFjHvupy3q8TFn3Hos2IAk4Ju5dCo8B3wP7VPr/FGaKiG+T+v+TQqIrOqMTL1VdWV1DdmcbO8KXBz6esmYWYKPwDL5b5FA1a0hwapHiom0r/cKaoqr+27/XcrS5UwSMbQAAAABJRU5ErkJggg==)](https://deepwiki.com/Ascend/AgentSDK)
</div>

# 目录结构

```text
.
├─aura
│  ├─base
│  │  ├─conf
│  │  ├─exceptions
│  │  ├─execution
│  │  ├─log
│  │  ├─misc
│  │  ├─resources
│  │  ├─utils
│  │  └─weight_loaders
│  ├─controllers
│  │  ├─rollout_controller
│  │  ├─train_controller
│  │  └─utils
│  ├─data_manager
│  ├─memory
│  │  └─episode
│  │      └─backend
│  ├─runner
│  │  ├─agent_engine_wrapper
│  │  │  ├─base
│  │  │  │  ├─agent
│  │  │  │  ├─environment
│  │  │  │  └─parser
│  │  │  ├─rllm
│  │  │  │  └─patch
│  │  │  └─smolagent
│  │  ├─agent_service
│  │  ├─infer_adapter
│  │  │  ├─omni_infer
│  │  │  ├─sglang
│  │  │  └─vllm
│  │  │      ├─extensions
│  │  │      └─patch
│  │  │          ├─comm
│  │  │          ├─patch_0_10_2
│  │  │          └─patch_0_9_1
│  │  ├─infer_service
│  │  │  └─infer_server
│  │  └─scheduler
│  ├─serve
│  └─trainer
│      ├─rollout
│      └─train_adapter
│          ├─mindspeed_rl
│          │  ├─config_cls
│          │  ├─hybrid_policy
│          │  ├─one_step_off_policy
│          │  │  ├─rollout
│          │  │  └─train
│          │  ├─patch
│          │  ├─utils
│          │  └─workers
│          ├─omni_rl
│          │  ├─full_async
│          │  ├─hybrid
│          │  └─patches
│          └─verl
│              ├─full_async
│              │  └─workers
│              └─hybrid
├─agents
│  ├─dtn_agent
│  └─math_agent
│      ├─environment
│      │  └─tools
│      ├─parser
│      │  └─tool_parser
│      ├─prompt
│      └─reward
├─cli
├─configs
│  ├─msrl_conf
│  │  ├─datasets
│  │  ├─envs
│  │  ├─evaluation
│  │  └─model
│  └─verl_conf
├─dockers
└─scripts
    └─deploy_vllm
```

# 版本说明

AgentSDK版本配套详情请参考：[版本配套详情](docs/zh/release_notes.md#版本配套说明)。

# 兼容性信息

AgentSDK版本兼容信息请参考：[版本兼容信息](docs/zh/release_notes.md#版本兼容性说明)。

# 环境部署

AgentSDK可通过源码进行安装。详细步骤请遵循[安装指南](docs/zh/installation_guide.md)。

# 快速入门

AgentSDK 通过脚本 `run_start_in_local.sh` 启动，支持 Serve 模式（服务化部署）和 Direct 模式（训练任务执行）。

```bash
bash run_start_in_local.sh --config-name your_config.yaml
```

- 环境准备和详细使用流程请参考[快速入门指南](docs/zh/quick_start.md)。
- 自定义 Agent 开发请参考[使用指南](docs/zh/user_guide/user_guide.md)。

# 特性介绍

## 训推共卡

支持训练和推理在同一张卡上运行，通过共享显存和计算资源，显著提高硬件资源利用率。该特性特别适合资源受限的场景，能够在单卡环境下完成Agent的训练和推理全流程，降低部署成本。

## 训推分离

支持训练和推理在不同卡上独立运行，实现训练和推理的解耦部署。该特性允许训练任务和推理服务分别扩展，提供更灵活的部署方案，适合大规模生产环境，便于独立优化训练性能和推理吞吐量。

## 混合批次调度

支持将计算密集型的预填充（Prefill）请求与访存密集型的解码（Decode）请求动态混合到同一批次中执行。该特性通过智能调度平衡计算与访存负载，显著提升NPU利用率与整体吞吐能力，同时降低解码阶段请求延迟，有效缓解队头阻塞问题。

# API参考

API参考详见：[Python API](docs/zh/api_python.md) 与 [命令行 API](docs/zh/command_api.md)。

# FAQ

相关FAQ请参考：[FAQ](docs/zh/faq.md)。

# 安全声明

- 使用API读取文件时，用户需要保证该文件的owner必须为自己，且权限不高于640，避免发生提权等安全问题。 外部下载的软件代码或程序可能存在风险，功能的安全性需由用户保证。
- 通信矩阵：目前Agent SDK开发套件包不会主动打开或者依赖任意端口，因此不涉及通信矩阵。
- 公网地址详见：[公网地址](docs/zh/resource/AgentSDK_公网地址_0000002516443057.xlsx)，Agent SDK的安装包中的网址安装结束后会被清除，并不会访问，不会造成风险。
- 安全加固指南请参考：[Agent SDK安全加固指南](docs/zh/security_hardening.md)。

# 分支维护策略

版本分支遵循定义的维护阶段：

| 状态 | 时间 | 说明 |
|------|------|------|
| 计划 | 1-3个月 | 特性规划 |
| 开发 | 3个月 | 新特性开发和问题修复，定期发布 |
| 维护 | 3-12个月 | 常规分支维护3个月，长期支持分支维护12个月。仅修复重大BUG，不加入新特性 |
| 生命周期终止（EOL） | N/A | 分支不再接受任何修改 |

# 版本维护策略

| 版本    | 维护策略 | 当前状态 | 发布日期 | 后续状态 | EOL日期 |
|-------|----------|----------|----------|----------|---------|
| master | 长期支持 | 开发 | 在研分支，不发布 | 持续开发 | - |
| v26.0 | 常规分支 | 维护 | 2026-01-28 | 预计2026/4/28起进入无维护状态 | 2026-04-28 |

# 免责声明

- 本仓库代码中包含多个开发分支，这些分支可能包含未完成、实验性或未测试的功能。在正式发布前，这些分支不应被应用于任何生产环境或者依赖关键业务的项目中。请务必使用我们的正式发行版本，以确保代码的稳定性和安全性。
使用开发分支所导致的任何问题、损失或数据损坏，本项目及其贡献者概不负责。

- 版本更新说明请参考：[更新说明](docs/zh/release_notes.md#更新说明)。

# License

AgentSDK以Mulan PSL v2许可证许可，对应许可证文本可查阅[LICENSE](../LICENSE.md)。

AgentSDK docs目录下的文档适用CC-BY 4.0许可证，具体请参见[LICENSE](./docs/LICENSE)文件。

# 贡献声明

- 贡献前，请先签署[开放项目贡献者许可协议（CLA）](https://clasign.osinfra.cn/sign/gitee_ascend-1611222220829317930)。
- 如果您遇到bug，请提交[issue](https://gitcode.com/Ascend/AgentSDK/issues)。
- 如果您计划贡献bug-fixes，请提交Pull Requests，参见[具体要求](../contributing.md)。
- 如果您计划贡献新特性、功能，请先创建issue与我们讨论。写明需求背景/目的，如何设计，对现有API等的影响。未经讨论提交PR可能会导致请求被拒绝，因为项目演进方向可能与您的想法存在偏差。
- 更详细的贡献流程，请参考[贡献指南](../contributing.md)。

# 建议与交流

欢迎大家为社区做贡献。如果有任何疑问或建议，请提交[issue](https://gitcode.com/Ascend/AgentSDK/issues)，我们会尽快回复。感谢您的支持。

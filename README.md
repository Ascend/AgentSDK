<h1 align="center">Agent SDK</h1>

<div align="center">

[![Ascend](https://img.shields.io/badge/Community-MindSDK-blue.svg)](https://www.hiascend.com/cn/developer/software/mindsdk)
[![License](https://badgen.net/badge/License/MulanPSL-2.0/blue)](LICENSE.md)
[![Zread](https://img.shields.io/badge/Zread-Ask_AI-_.svg?style=flat&color=0052D9&labelColor=000000&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQuOTYxNTYgMS42MDAxSDIuMjQxNTZDMS44ODgxIDEuNjAwMSAxLjYwMTU2IDEuODg2NjQgMS42MDE1NiAyLjI0MDFWNC45NjAxQzEuNjAxNTYgNS4zMTM1NiAxLjg4ODEgNS42MDAxIDIuMjQxNTYgNS42MDAxSDQuOTYxNTZDNS4zMTUwMiA1LjYwMDEgNS42MDE1NiA1LjMxMzU2IDUuNjAxNTYgNC45NjAxVjIuMjQwMUM1LjYwMTU2IDEuODg2NjQgNS4zMTUwMiAxLjYwMDEgNC45NjE1NiAxLjYwMDFaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00Ljk2MTU2IDEwLjM5OTlIMi4yNDE1NkMxLjg4ODEgMTAuMzk5OSAxLjYwMTU2IDEwLjY4NjQgMS42MDE1NiAxMS4wMzk5VjEzLjc1OTlDMS42MDE1NiAxNC4xMTM0IDEuODg4MSAxNC4zOTk5IDIuMjQxNTYgMTQuMzk5OUg0Ljk2MTU2QzUuMzE1MDIgMTQuMzk5OSA1LjYwMTU2IDE0LjExMzQgNS42MDE1NiAxMy43NTk5VjExLjAzOTlDNS42MDE1NiAxMC42ODY0IDUuMzE1MDIgMTAuMzk5OSA0Ljk2MTU2IDEwLjM5OTlaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik0xMy43NTg0IDEuNjAwMUgxMS4wMzg0QzEwLjY4NSAxLjYwMDEgMTAuMzk4NCAxLjg4NjY0IDEwLjM5ODQgMi4yNDAxVjQuOTYwMUMxMC4zOTg0IDUuMzEzNTYgMTAuNjg1IDUuNjAwMSAxMS4wMzg0IDUuNjAwMUgxMy43NTg0QzE0LjExMTkgNS42MDAxIDE0LjM5ODQgNS4zMTM1NiAxNC4zOTg0IDQuOTYwMVYyLjI0MDFDMTQuMzk4NCAxLjg4NjY0IDE0LjExMTkgMS42MDAxIDEzLjc1ODQgMS42MDAxWiIgZmlsbD0iI2ZmZiIvPgo8cGF0aCBkPSJNNCAxMkwxMiA0TDQgMTJaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00IDEyTDEyIDQiIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgo8L3N2Zz4K&logoColor=ffffff)](https://zread.ai/Ascend/AgentSDK)
[![DeepWiki](https://img.shields.io/badge/DeepWiki-Ask_AI-_.svg?style=flat&color=0052D9&labelColor=000000&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAAyCAYAAAAnWDnqAAAAAXNSR0IArs4c6QAAA05JREFUaEPtmUtyEzEQhtWTQyQLHNak2AB7ZnyXZMEjXMGeK/AIi+QuHrMnbChYY7MIh8g01fJoopFb0uhhEqqcbWTp06/uv1saEDv4O3n3dV60RfP947Mm9/SQc0ICFQgzfc4CYZoTPAswgSJCCUJUnAAoRHOAUOcATwbmVLWdGoH//PB8mnKqScAhsD0kYP3j/Yt5LPQe2KvcXmGvRHcDnpxfL2zOYJ1mFwrryWTz0advv1Ut4CJgf5uhDuDj5eUcAUoahrdY/56ebRWeraTjMt/00Sh3UDtjgHtQNHwcRGOC98BJEAEymycmYcWwOprTgcB6VZ5JK5TAJ+fXGLBm3FDAmn6oPPjR4rKCAoJCal2eAiQp2x0vxTPB3ALO2CRkwmDy5WohzBDwSEFKRwPbknEggCPB/imwrycgxX2NzoMCHhPkDwqYMr9tRcP5qNrMZHkVnOjRMWwLCcr8ohBVb1OMjxLwGCvjTikrsBOiA6fNyCrm8V1rP93iVPpwaE+gO0SsWmPiXB+jikdf6SizrT5qKasx5j8ABbHpFTx+vFXp9EnYQmLx02h1QTTrl6eDqxLnGjporxl3NL3agEvXdT0WmEost648sQOYAeJS9Q7bfUVoMGnjo4AZdUMQku50McDcMWcBPvr0SzbTAFDfvJqwLzgxwATnCgnp4wDl6Aa+Ax283gghmj+vj7feE2KBBRMW3FzOpLOADl0Isb5587h/U4gGvkt5v60Z1VLG8BhYjbzRwyQZemwAd6cCR5/XFWLYZRIMpX39AR0tjaGGiGzLVyhse5C9RKC6ai42ppWPKiBagOvaYk8lO7DajerabOZP46Lby5wKjw1HCRx7p9sVMOWGzb/vA1hwiWc6jm3MvQDTogQkiqIhJV0nBQBTU+3okKCFDy9WwferkHjtxib7t3xIUQtHxnIwtx4mpg26/HfwVNVDb4oI9RHmx5WGelRVlrtiw43zboCLaxv46AZeB3IlTkwouebTr1y2NjSpHz68WNFjHvupy3q8TFn3Hos2IAk4Ju5dCo8B3wP7VPr/FGaKiG+T+v+TQqIrOqMTL1VdWV1DdmcbO8KXBz6esmYWYKPwDL5b5FA1a0hwapHiom0r/cKaoqr+27/XcrS5UwSMbQAAAABJRU5ErkJggg==)](https://deepwiki.com/Ascend/AgentSDK)

</div>

## ✨ 最新消息

<span style="font-size:14px;">

🔹 **[2026.7.8]**：🚀 **Aura**新增支持 Qwen3-14B模型，并提供[快速拉起指南](./docs/zh/aura/models/qwen3-14b_quick_start/qwen3-14b-hybrid.md)<br>
🔹 **[2026.06.12]**：🚀 **Aura**新增支持 Qwen3-4B, Qwen3-8B, Qwen3-32B, Qwen3-30B-A3B 模型，并提供[快速拉起指南](./docs/zh/aura/models)<br>
🔹 **[2026.04.25]**：🚀 发布 Agent SDK 全新训推调框架 **Aura** ，支持训推共卡与训推分离模式<br>

</span>

## ℹ️ 简介

Agent SDK是昇腾亲和的Agent生态仓库，旨在为昇腾NPU基础设施上的Agent应用提供开箱即用的生态工具与框架支持，帮助开发者快速构建和部署AI智能体。

Agent SDK目前涵盖Agentic RL训推调框架、Agent应用开发等多个方向，后续将持续扩展更多Agent生态组件。

## ⚙️ 功能介绍

**核心产品**

| 功能 | 描述 |
| --- | --- |
| [Aura](./aura/README.md) | Agentic RL 训推调一体化框架，支持训推共卡与训推分离模式，对接多种训练、推理、Agent 引擎 |
| [Openclaw](./openclaw/README.md) | 基于 OpenClaw 构建的多领域 Agent 框架与服务，集成代码生成、研究分析等垂直领域能力 |

## 🗺️ Roadmap

[Roadmap （2026Q3）](https://gitcode.com/Ascend/AgentSDK/issues/85)

## 🔀 版本维护策略

| 版本    | 维护策略 | 当前状态 | 发布日期 | 后续状态 | EOL日期 |
|-------|----------|----------|----------|----------|---------|
| master | 长期支持 | 开发 | 在研分支，不发布 | 持续开发 | - |
| v26.0.0.beta.1 | 常规分支 | EOL | 2026-04-25 | 预计2026-07-25起进入无维护状态 | 2026-07-25 |
| v26.1.0 | 常规分支 | 维护 | 2026-07-10 | 预计2027-01-10起进入无维护状态 | 2027-01-10 |

## 🛠️ 贡献指南

- 贡献前，请先签署[开放项目贡献者许可协议（CLA）](https://clasign.osinfra.cn/sign/gitee_ascend-1611222220829317930)。
- 如果您遇到 bug，请提交[issue](https://gitcode.com/Ascend/AgentSDK/issues)。
- 如果您计划贡献 bug-fixes，请提交 Pull Requests，参见[具体要求](./contributing.md)。
- 如果您计划贡献新特性、功能，请先创建 issue 与我们讨论。写明需求背景/目的，如何设计，对现有 API 等的影响。未经讨论提交 PR 可能会导致请求被拒绝，因为项目演进方向可能与您的想法存在偏差。
- 更详细的贡献流程，请参考[贡献指南](./contributing.md)。

## ⚖️ 相关说明

🔹 《[许可证声明](./LICENSE.md)》<br>
🔹 《[文档许可证声明](./docs/LICENSE)》<br>
🔹 《[第三方开源软件声明](./Third_Party_Open_Source_Software_Notice.md)》<br>

## 🤝 建议与交流

欢迎大家为社区做贡献。如果有任何疑问或建议，请提交[issue](https://gitcode.com/Ascend/AgentSDK/issues)，我们会尽快回复。感谢您的支持。

| 资源 | 说明 |
|:--|:--|
| [FAQ](./docs/zh/aura/07_faq.md) | Aura常见问题解答与使用答疑 |
| [创建Issue](https://gitcode.com/Ascend/AgentSDK/issues/create/choose) | 提交 Bug、需求或建议 |
| [社区任务](https://gitcode.com/Ascend/AgentSDK/issues/59) | 查看和认领社区任务 |
| [会议日历](https://meeting.ascend.osinfra.cn/?sig=sig-AgentSDK) | 社区定期例会与活动日程 |

## 🙏 致谢

Agent SDK 由华为公司的以下部门联合贡献：

🔹 昇腾应用使能开发部
🔹 2012 系统工程实验室
🔹 华为全球技术服务部-AI 计算实验室

感谢来自社区的每一个 PR，欢迎贡献 Agent SDK!

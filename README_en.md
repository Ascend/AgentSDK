<h1 align="center">Agent SDK</h1>

<div align="center">

[![Ascend](https://img.shields.io/badge/Community-MindSDK-blue.svg)](https://www.hiascend.com/eng/developer/software/mindsdk)
[![License](https://badgen.net/badge/License/MulanPSL-2.0/blue)](LICENSE.md)
[![Zread](https://img.shields.io/badge/Zread-Ask_AI-_.svg?style=flat&color=0052D9&labelColor=000000&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQuOTYxNTYgMS42MDAxSDIuMjQxNTZDMS44ODgxIDEuNjAwMSAxLjYwMTU2IDEuODg2NjQgMS42MDE1NiAyLjI0MDFWNC45NjAxQzEuNjAxNTYgNS4zMTM1NiAxLjg4ODEgNS42MDAxIDIuMjQxNTYgNS42MDAxSDQuOTYxNTZDNS4zMTUwMiA1LjYwMDEgNS42MDE1NiA1LjMxMzU2IDUuNjAxNTYgNC45NjAxVjIuMjQwMUM1LjYwMTU2IDEuODg2NjQgNS4zMTUwMiAxLjYwMDEgNC45NjE1NiAxLjYwMDFaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00Ljk2MTU2IDEwLjM5OTlIMi4yNDE1NkMxLjg4ODEgMTAuMzk5OSAxLjYwMTU2IDEwLjY4NjQgMS42MDE1NiAxMS4wMzk5VjEzLjc1OTlDMS42MDE1NiAxNC4xMTM0IDEuODg4MSAxNC4zOTk5IDIuMjQxNTYgMTQuMzk5OUg0Ljk2MTU2QzUuMzE1MDIgMTQuMzk5OSA1LjYwMTU2IDE0LjExMzQgNS42MDE1NiAxMy43NTk5VjExLjAzOTlDNS42MDE1NiAxMC42ODY0IDUuMzE1MDIgMTAuMzk5OSA0Ljk2MTU2IDEwLjM5OTlaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik0xMy43NTg0IDEuNjAwMUgxMS4wMzg0QzEwLjY4NSAxLjYwMDEgMTAuMzk4NCAxLjg4NjY0IDEwLjM5ODQgMi4yNDAxVjQuOTYwMUMxMC4zOTg0IDUuMzEzNTYgMTAuNjg1IDUuNjAwMSAxMS4wMzg0IDUuNjAwMUgxMy43NTg0QzE0LjExMTkgNS42MDAxIDE0LjM5ODQgNS4zMTM1NiAxNC4zOTg0IDQuOTYwMVYyLjI0MDFDMTQuMzk4NCAxLjg4NjY0IDE0LjExMTkgMS42MDAxIDEzLjc1ODQgMS42MDAxWiIgZmlsbD0iI2ZmZiIvPgo8cGF0aCBkPSJNNCAxMkwxMiA0TDQgMTJaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00IDEyTDEyIDQiIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgo8L3N2Zz4K&logoColor=ffffff)](https://zread.ai/Ascend/AgentSDK)
[![DeepWiki](https://img.shields.io/badge/DeepWiki-Ask_AI-_.svg?style=flat&color=0052D9&labelColor=000000&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAAyCAYAAAAnWDnqAAAAAXNSR0IArs4c6QAAA05JREFUaEPtmUtyEzEQhtWTQyQLHNak2AB7ZnyXZMEjXMGeK/AIi+QuHrMnbChYY7MIh8g01fJoopFb0uhhEqqcbWTp06/uv1saEDv4O3n3dV60RfP947Mm9/SQc0ICFQgzfc4CYZoTPAswgSJCCUJUnAAoRHOAUOcATwbmVLWdGoH//PB8mnKqScAhsD0kYP3j/Yt5LPQe2KvcXmGvRHcDnpxfL2zOYJ1mFwrryWTz0advv1Ut4CJgf5uhDuDj5eUcAUoahrdY/56ebRWeraTjMt/00Sh3UDtjgHtQNHwcRGOC98BJEAEymycmYcWwOprTgcB6VZ5JK5TAJ+fXGLBm3FDAmn6oPPjR4rKCAoJCal2eAiQp2x0vxTPB3ALO2CRkwmDy5WohzBDwSEFKRwPbknEggCPB/imwrycgxX2NzoMCHhPkDwqYMr9tRcP5qNrMZHkVnOjRMWwLCcr8ohBVb1OMjxLwGCvjTikrsBOiA6fNyCrm8V1rP93iVPpwaE+gO0SsWmPiXB+jikdf6SizrT5qKasx5j8ABbHpFTx+vFXp9EnYQmLx02h1QTTrl6eDqxLnGjporxl3NL3agEvXdT0WmEost648sQOYAeJS9Q7bfUVoMGnjo4AZdUMQku50McDcMWcBPvr0SzbTAFDfvJqwLzgxwATnCgnp4wDl6Aa+Ax283gghmj+vj7feE2KBBRMW3FzOpLOADl0Isb5587h/U4gGvkt5v60Z1VLG8BhYjbzRwyQZemwAd6cCR5/XFWLYZRIMpX39AR0tjaGGiGzLVyhse5C9RKC6ai42ppWPKiBagOvaYk8lO7DajerabOZP46Lby5wKjw1HCRx7p9sVMOWGzb/vA1hwiWc6jm3MvQDTogQkiqIhJV0nBQBTU+3okKCFDy9WwferkHjtxib7t3xIUQtHxnIwtx4mpg26/HfwVNVDb4oI9RHmx5WGelRVlrtiw43zboCLaxv46AZeB3IlTkwouebTr1y2NjSpHz68WNFjHvupy3q8TFn3Hos2IAk4Ju5dCo8B3wP7VPr/FGaKiG+T+v+TQqIrOqMTL1VdWV1DdmcbO8KXBz6esmYWYKPwDL5b5FA1a0hwapHiom0r/cKaoqr+27/XcrS5UwSMbQAAAABJRU5ErkJggg==)](https://deepwiki.com/Ascend/AgentSDK)

</div>

## ✨ Latest News

<span style="font-size:14px;">

🔹 **[Jul. 8, 2026]**: 🚀 **Aura** adds support for the Qwen3-14B model and provides a [quick start guide](./docs/en/aura/models/qwen3-14b_quick_start/qwen3-14b-hybrid.md).<br>
🔹 **[Jun. 12, 2026]**: 🚀 **Aura** adds support for the Qwen3-4B, Qwen3-8B, Qwen3-32B, and Qwen3-30B-A3B models and provides a [quick start guide](./docs/en/aura/models).<br>
🔹 **[Apr. 25, 2026]**: 🚀 Released **Aura**, a brand-new training, inference, and tuning framework for Agent SDK that supports the co-located and disaggregated training and inference modes.<br>

</span>

## ℹ️ Introduction

Agent SDK is an Ascend-friendly agent ecosystem repository designed to provide out-of-the-box ecosystem tools and framework support for agent applications on Ascend NPU infrastructure, helping you quickly build and deploy AI agents.

Agent SDK currently covers multiple directions, including the Agentic RL training, inference, and tuning framework and agent application development, and will continue to expand with more agent ecosystem components. **Aura** focuses on integrated Agentic RL training, inference, and tuning capabilities, while **OpenClaw** targets multi-domain agent application development and deployment.

## ⚙️ Features

**Core Products**

| Feature | Description |
| --- | --- |
| [Aura](./aura/README.md) | An integrated Agentic RL training, inference, and tuning framework that supports the co-located and disaggregated training and inference modes and connects to various training, inference, and agent engines. |
| [OpenClaw](./openclaw/README.md) | A multi-domain agent framework and services built on OpenClaw, integrating vertical-domain capabilities such as code generation and research analysis. |

## 🗺️ Roadmap

[Roadmap (2026Q3)](https://atomgit.com/Ascend/AgentSDK/issues/85)

## 🔀 Version Maintenance Strategy

| Version   | Maintenance Strategy| Current Status| Release Date| Next Status| EOL Date|
|-------|----------|----------|----------|----------|---------|
| master | Long-term support| Development| Under active development and not yet released| Continuous development| - |
| v26.1.0 | Regular branch| Maintenance| 2026-07-10 | Expected to enter unsupported status on January 10, 2027.| 2027-01-10 |
| v26.0.0.beta.1 | Regular branch| EOL| 2026-04-25 | Expected to enter unsupported status on July 25, 2026.| 2026-07-25 |

## 🛠️ Contribution Guide

- Before contributing, sign the [Open Project Contributor License Agreement (CLA)](https://clasign.osinfra.cn/sign/gitee_ascend-1611222220829317930).
- If you encounter a bug, submit an [issue](https://atomgit.com/Ascend/AgentSDK/issues).
- If you plan to contribute bug fixes, submit a pull request (PR). See [Contribution Requirements](./contributing.md).
- If you plan to contribute new features or functionality, create an issue to discuss it with us first. Describe the background or purpose of the requirement, the design, and its impact on existing APIs. Submitting a PR without prior discussion may lead to rejection, as the evolution direction of the project might differ from your ideas.
- For a detailed contribution process, see the [Contribution Guide](./contributing.md).

## ⚖️ Related Notices

🔹 [License Statement](./LICENSE.md)<br>
🔹 [Documentation License Statement](./docs/LICENSE)<br>
🔹 [Third-Party Open-Source Software Notice](./Third_Party_Open_Source_Software_Notice.md)<br>

## 🤝 Suggestions and Feedback

You are welcome to contribute to the community. If you have any questions or suggestions, please submit an [issue](https://atomgit.com/Ascend/AgentSDK/issues). We will reply as soon as possible. Thank you for your support.

| Resource | Description |
|:--|:--|
| [FAQ](./docs/en/aura/07_faq.md) | Answers to frequently asked questions and usage inquiries about Aura. |
| [Create an Issue](https://atomgit.com/Ascend/AgentSDK/issues/create/choose) | Submit bugs, feature requests, or suggestions. |
| [Community Tasks](https://atomgit.com/Ascend/AgentSDK/issues/59) | View and claim community tasks. |
| [Meeting Calendar](https://meeting.ascend.osinfra.cn/?sig=sig-AgentSDK) | Schedule of regular community meetings and activities. |

## 🙏 Acknowledgments

Agent SDK is jointly contributed by Huawei Technologies Co., Ltd. and developers.

We appreciate every PR from the community. You are welcome to contribute to Agent SDK!

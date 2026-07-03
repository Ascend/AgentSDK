<h1 align="center">Aura: An Integrated Training-Inference-Tuning Framework for AI Agents</h1>

<div align="center">

[![Ascend](https://img.shields.io/badge/Community-MindSDK-blue.svg)](https://www.hiascend.com/cn/developer/software/mindsdk)
[![License](https://badgen.net/badge/License/MulanPSL-2.0/blue)](../LICENSE.md)
[![Zread](https://img.shields.io/badge/Zread-Ask_AI-_.svg?style=flat&color=0052D9&labelColor=000000&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQuOTYxNTYgMS42MDAxSDIuMjQxNTZDMS44ODgxIDEuNjAwMSAxLjYwMTU2IDEuODg2NjQgMS42MDE1NiAyLjI0MDFWNC45NjAxQzEuNjAxNTYgNS4zMTM1NiAxLjg4ODEgNS42MDAxIDIuMjQxNTYgNS42MDAxSDQuOTYxNTZDNS4zMTUwMiA1LjYwMDEgNS42MDE1NiA1LjMxMzU2IDUuNjAxNTYgNC45NjAxVjIuMjQwMUM1LjYwMTU2IDEuODg2NjQgNS4zMTUwMiAxLjYwMDEgNC45NjE1NiAxLjYwMDFaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00Ljk2MTU2IDEwLjM5OTlIMi4yNDE1NkMxLjg4ODEgMTAuMzk5OSAxLjYwMTU2IDEwLjY4NjQgMS42MDE1NiAxMS4wMzk5VjEzLjc1OTlDMS42MDE1NiAxNC4xMTM0IDEuODg4MSAxNC4zOTk5IDIuMjQxNTYgMTQuMzk5OUg0Ljk2MTU2QzUuMzE1MDIgMTQuMzk5OSA1LjYwMTU2IDE0LjExMzQgNS42MDE1NiAxMy43NTk5VjExLjAzOTlDNS42MDE1NiAxMC42ODY0IDUuMzE1MDIgMTAuMzk5OSA0Ljk2MTU2IDEwLjM5OTlaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik0xMy43NTg0IDEuNjAwMUgxMS4wMzg0QzEwLjY4NSAxLjYwMDEgMTAuMzk4NCAxLjg4NjY0IDEwLjM5ODQgMi4yNDAxVjQuOTYwMUMxMC4zOTg0IDUuMzEzNTYgMTAuNjg1IDUuNjAwMSAxMS4wMzg0IDUuNjAwMUgxMy43NTg0QzE0LjExMTkgNS42MDAxIDE0LjM5ODQgNS4zMTM1NiAxNC4zOTg0IDQuOTYwMVYyLjI0MDFDMTQuMzk4NCAxLjg4NjY0IDE0LjExMTkgMS42MDAxIDEzLjc1ODQgMS42MDAxWiIgZmlsbD0iI2ZmZiIvPgo8cGF0aCBkPSJNNCAxMkwxMiA0TDQgMTJaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00IDEyTDEyIDQiIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgo8L3N2Zz4K&logoColor=ffffff)](https://zread.ai/Ascend/AgentSDK)
[![DeepWiki](https://img.shields.io/badge/DeepWiki-Ask_AI-_.svg?style=flat&color=0052D9&labelColor=000000&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAAyCAYAAAAnWDnqAAAAAXNSR0IArs4c6QAAA05JREFUaEPtmUtyEzEQhtWTQyQLHNak2AB7ZnyXZMEjXMGeK/AIi+QuHrMnbChYY7MIh8g01fJoopFb0uhhEqqcbWTp06/uv1saEDv4O3n3dV60RfP947Mm9/SQc0ICFQgzfc4CYZoTPAswgSJCCUJUnAAoRHOAUOcATwbmVLWdGoH//PB8mnKqScAhsD0kYP3j/Yt5LPQe2KvcXmGvRHcDnpxfL2zOYJ1mFwrryWTz0advv1Ut4CJgf5uhDuDj5eUcAUoahrdY/56ebRWeraTjMt/00Sh3UDtjgHtQNHwcRGOC98BJEAEymycmYcWwOprTgcB6VZ5JK5TAJ+fXGLBm3FDAmn6oPPjR4rKCAoJCal2eAiQp2x0vxTPB3ALO2CRkwmDy5WohzBDwSEFKRwPbknEggCPB/imwrycgxX2NzoMCHhPkDwqYMr9tRcP5qNrMZHkVnOjRMWwLCcr8ohBVb1OMjxLwGCvjTikrsBOiA6fNyCrm8V1rP93iVPpwaE+gO0SsWmPiXB+jikdf6SizrT5qKasx5j8ABbHpFTx+vFXp9EnYQmLx02h1QTTrl6eDqxLnGjporxl3NL3agEvXdT0WmEost648sQOYAeJS9Q7bfUVoMGnjo4AZdUMQku50McDcMWcBPvr0SzbTAFDfvJqwLzgxwATnCgnp4wDl6Aa+Ax283gghmj+vj7feE2KBBRMW3FzOpLOADl0Isb5587h/U4gGvkt5v60Z1VLG8BhYjbzRwyQZemwAd6cCR5/XFWLYZRIMpX39AR0tjaGGiGzLVyhse5C9RKC6ai42ppWPKiBagOvaYk8lO7DajerabOZP46Lby5wKjw1HCRx7p9sVMOWGzb/vA1hwiWc6jm3MvQDTogQkiqIhJV0nBQBTU+3okKCFDy9WwferkHjtxib7t3xIUQtHxnIwtx4mpg26/HfwVNVDb4oI9RHmx5WGelRVlrtiw43zboCLaxv46AZeB3IlTkwouebTr1y2NjSpHz68WNFjHvupy3q8TFn3Hos2IAk4Ju5dCo8B3wP7VPr/FGaKiG+T+v+TQqIrOqMTL1VdWV1DdmcbO8KXBz6esmYWYKPwDL5b5FA1a0hwapHiom0r/cKaoqr+27/XcrS5UwSMbQAAAABJRU5ErkJggg==)](https://deepwiki.com/Ascend/AgentSDK)

</div>

# ✨ Latest News

- [2026.06.12]: 🚀 Added support for Qwen3-4B, Qwen3-8B, Qwen3-32B, and Qwen3-30B-A3B models, with [quick start guides](../docs/aura/zh/models) provided.
- [2026.04.25]: 🚀 Released AgentSDK's brand-new training-inference-tuning framework **Aura**, supporting both hybrid and separate deployment modes.

# ℹ️ Introduction

**Aura (Agentic Ultra-fast Reinforcement Architecture)** is an integrated training-inference-tuning framework for foundation models. It continuously optimizes foundation models based on task trajectories and reward signals. Through reinforcement learning and other optimization methods, models gradually acquire Agentic capabilities such as planning, tool use, and long-horizon decision-making via post-training.

**Aura** is compatible with various training engines, inference engines, and Agent frameworks through unified abstract interfaces. It supports users to flexibly integrate custom models and toolchains, helping developers quickly build, train, and deploy their own Agents.

<a id="fig173917397815"></a>

<div align="center">

![](../docs/aura/zh/figures/Aura框架架构图.png)

</div>

# ⚙️ Features

- Supports [hybrid training-inference](../docs/aura/zh/04_user_guide/02_hybrid.md) and [separate training-inference](../docs/aura/zh/04_user_guide/03_one_step_off.md) modes.
- Supports [custom Agent integration](../docs/aura/zh/04_user_guide/04_custom_agent.md).
- Supports [Qwen3-4B](../docs/aura/zh/models/qwen3-4b.md), [Qwen3-8B](../docs/aura/zh/models/qwen3_8b.md), [Qwen3-32B](../docs/aura/zh/models/qwen3_32b.md), and [Qwen3-30B-A3B](../docs/aura/zh/models/qwen3-30b-a3b.md) models.
- Supports verl training engine and vllm inference engine.
- Supports rLLM agent engine.
- Uses TensorBoard to record training metrics.
- Hardware support: Ascend A2/A3 architecture servers.

# 🚀 Quick Start

| Training Framework | Model | Scenario | Server Architecture | Recommended Minimum Compute Resources | Quick Start Guide |
| --- | --- | --- | --- | --- | --- |
| verl+vllm+fsdp2+Hybrid | Qwen3-4B | Math | A3 | Single node 4*64GB memory | [Qwen3-4B Hybrid Mode Quick Start Guide](../docs/aura/zh/models/qwen3-4b_quick_start/qwen3-4b-hybrid.md) |
| verl+vllm+fsdp2+Separate | Qwen3-4B | Math | A3 | Dual node 4*64GB memory | [Qwen3-4B Separate Mode Quick Start Guide](../docs/aura/zh/models/qwen3-4b_quick_start/qwen3-4b-one-step-off.md) |

- For the complete general quick start workflow, please refer to: [Quick Start Guide](../docs/aura/zh/03_quick_start.md).

# 📦 Installation Guide

Currently, Aura only provides environment deployment procedures, including container environment deployment, model weight preparation, and training data preparation.

**Prerequisites**: Python, Docker (for container environment deployment)

**Python Dependencies**: [requirements.txt](requirements.txt), containing the base dependencies required for the Aura core framework to run (such as torch, transformers, ray, etc.).

**Third-party Dependencies**: [third_party/requirements_with_verl_vllm.txt](third_party/requirements_with_verl_vllm.txt), containing third-party repository dependencies such as megatron, mindspeed, and rllm.

For detailed steps, please follow the [Installation Guide](../docs/aura/zh/02_installation_guide.md).

# 📖 Usage Guide

**Prerequisites**: It is recommended to familiarize yourself with the following dependency frameworks before using Aura:

- [vllm-ascend](https://docs.vllm.ai/projects/vllm-ascend-cn/zh-cn/latest/): vLLM inference backend on Ascend NPUs
- [verl](https://verl.org.cn/en/latest/index.html): Reinforcement learning training framework

**Usage Guides**:

- Separate training-inference mode (sufficient resources): Please refer to the [Separate Mode Usage Guide](../docs/aura/zh/04_user_guide/03_one_step_off.md).
- Hybrid training-inference mode (limited resources): Please refer to the [Hybrid Mode Usage Guide](../docs/aura/zh/04_user_guide/02_hybrid.md).
- Defining Agent content and integrating custom data and toolchains: Please refer to the [Custom Agent Integration Guide](../docs/aura/zh/04_user_guide/04_custom_agent.md).

# ❓ FAQ

The FAQ includes error solutions that may be needed during environment deployment or operation. For related FAQs, please refer to: [FAQ](../docs/aura/zh/07_faq.md).

# 🛠️ Contribution Guide

- Before contributing, please sign the [Open Project Contributor License Agreement (CLA)](https://clasign.osinfra.cn/sign/gitee_ascend-1611222220829317930).
- If you encounter a bug, please submit an [issue](https://gitcode.com/Ascend/AgentSDK/issues).
- If you plan to contribute bug fixes, please submit a Pull Request. See [specific requirements](../contributing.md).
- If you plan to contribute new features or functionality, please create an issue to discuss with us first. Describe the requirement background/purpose, design approach, and impact on existing APIs. Submitting a PR without prior discussion may result in rejection, as the project evolution direction may differ from your ideas.
- For a more detailed contribution process, please refer to the [Contribution Guide](../contributing.md).

# ⚖️ Related Information

🔹 《[Release Notes](../docs/aura/zh/08_release_notes.md)》<br>
🔹 《[License Statement](../LICENSE.md)》<br>
🔹 《[Document License Statement](../docs/aura/LICENSE)》<br>
🔹 《[Disclaimer](../docs/aura/zh/09_disclaimer.md)》<br>
🔹 《[Security Statement](../docs/aura/zh/06_security_hardening.md)》<br>
🔹 《[Appendix](../docs/aura/zh/10_appendix.md)》<br>

# 🤝 Suggestions and Feedback

Everyone is welcome to contribute to the community. If you have any questions or suggestions, please submit an [issue](https://gitcode.com/Ascend/AgentSDK/issues), and we will reply as soon as possible. Thank you for your support.

# 🙏 Acknowledgments

Aura is jointly contributed by the following departments of Huawei:

- Ascend Application Enablement Development Department
- 2012 Systems Engineering Laboratory
- Huawei Global Technical Services Department - AI Computing Laboratory

Thank you for every PR from the community. Contributions to Aura are welcome!

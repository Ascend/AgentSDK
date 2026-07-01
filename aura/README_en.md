# Agent SDK

- [Agent SDK](#agent-sdk)
- [Latest News](#latest-news)
- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Release Notes](#release-notes)
- [Compatibility Information](#compatibility-information)
- [Environment Deployment](#environment-deployment)
- [Quick Start](#quick-start)
- [Features](#features)
- [API Reference](#api-reference)
- [FAQ](#faq)
- [Security Statement](#security-statement)
- [Branch Maintenance Strategy](#branch-maintenance-strategy)
- [Version Maintenance Strategy](#version-maintenance-strategy)
- [Disclaimer](#disclaimer)
- [License](#license)
- [Contribution Statement](#contribution-statement)
- [Suggestions and Feedback](#suggestions-and-feedback)

# Latest News

- [2026.01.28]: 🚀 Integrated the MindSpeed-RL training framework and added support for the GRPO algorithm.
- [2026.01.28]: 🚀 Provided the `BaseEngineWrapper` abstract interface to support custom agent logic.

# Overview

AgentSDK provides an enterprise-grade Agentic RL training and inference framework with layered decoupling and Ascend affinity.
It helps you build, run, and scale LLM agents with tool use and multi-step reasoning on Ascend NPU infrastructure.
By integrating agent logic and controllable tool calling, it helps developers build domain-specific agentic applications quickly.

For details, see [Introduction](docs/en/introduction.md).

<div align="center">

[![Zread](https://img.shields.io/badge/Zread-Ask_AI-_.svg?style=flat&color=0052D9&labelColor=000000&logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iMTYiIGhlaWdodD0iMTYiIHZpZXdCb3g9IjAgMCAxNiAxNiIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTQuOTYxNTYgMS42MDAxSDIuMjQxNTZDMS44ODgxIDEuNjAwMSAxLjYwMTU2IDEuODg2NjQgMS42MDE1NiAyLjI0MDFWNC45NjAxQzEuNjAxNTYgNS4zMTM1NiAxLjg4ODEgNS42MDAxIDIuMjQxNTYgNS42MDAxSDQuOTYxNTZDNS4zMTUwMiA1LjYwMDEgNS42MDE1NiA1LjMxMzU2IDUuNjAxNTYgNC45NjAxVjIuMjQwMUM1LjYwMTU2IDEuODg2NjQgNS4zMTUwMiAxLjYwMDEgNC45NjE1NiAxLjYwMDFaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00Ljk2MTU2IDEwLjM5OTlIMi4yNDE1NkMxLjg4ODEgMTAuMzk5OSAxLjYwMTU2IDEwLjY4NjQgMS42MDE1NiAxMS4wMzk5VjEzLjc1OTlDMS42MDE1NiAxNC4xMTM0IDEuODg4MSAxNC4zOTk5IDIuMjQxNTYgMTQuMzk5OUg0Ljk2MTU2QzUuMzE1MDIgMTQuMzk5OSA1LjYwMTU2IDE0LjExMzQgNS42MDE1NiAxMy43NTk5VjExLjAzOTlDNS42MDE1NiAxMC42ODY0IDUuMzE1MDIgMTAuMzk5OSA0Ljk2MTU2IDEwLjM5OTlaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik0xMy43NTg0IDEuNjAwMUgxMS4wMzg0QzEwLjY4NSAxLjYwMDEgMTAuMzk4NCAxLjg4NjY0IDEwLjM5ODQgMi4yNDAxVjQuOTYwMUMxMC4zOTg0IDUuMzEzNTYgMTAuNjg1IDUuNjAwMSAxMS4wMzg0IDUuNjAwMUgxMy43NTg0QzE0LjExMTkgNS42MDAxIDE0LjM5ODQgNS4zMTM1NiAxNC4zOTg0IDQuOTYwMVYyLjI0MDFDMTQuMzk4NCAxLjg4NjY0IDE0LjExMTkgMS42MDAxIDEzLjc1ODQgMS42MDAxWiIgZmlsbD0iI2ZmZiIvPgo8cGF0aCBkPSJNNCAxMkwxMiA0TDQgMTJaIiBmaWxsPSIjZmZmIi8%2BCjxwYXRoIGQ9Ik00IDEyTDEyIDQiIHN0cm9rZT0iI2ZmZiIgc3Ryb2tlLXdpZHRoPSIxLjUiIHN0cm9rZS1saW5lY2FwPSJyb3VuZCIvPgo8L3N2Zz4K&logoColor=ffffff)](https://zread.ai/Ascend/AgentSDK)&nbsp;&nbsp;&nbsp;&nbsp;
[![DeepWiki](https://img.shields.io/badge/DeepWiki-Ask_AI-_.svg?style=flat&color=0052D9&labelColor=000000&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAAyCAYAAAAnWDnqAAAAAXNSR0IArs4c6QAAA05JREFUaEPtmUtyEzEQhtWTQyQLHNak2AB7ZnyXZMEjXMGeK/AIi+QuHrMnbChYY7MIh8g01fJoopFb0uhhEqqcbWTp06/uv1saEDv4O3n3dV60RfP947Mm9/SQc0ICFQgzfc4CYZoTPAswgSJCCUJUnAAoRHOAUOcATwbmVLWdGoH//PB8mnKqScAhsD0kYP3j/Yt5LPQe2KvcXmGvRHcDnpxfL2zOYJ1mFwrryWTz0advv1Ut4CJgf5uhDuDj5eUcAUoahrdY/56ebRWeraTjMt/00Sh3UDtjgHtQNHwcRGOC98BJEAEymycmYcWwOprTgcB6VZ5JK5TAJ+fXGLBm3FDAmn6oPPjR4rKCAoJCal2eAiQp2x0vxTPB3ALO2CRkwmDy5WohzBDwSEFKRwPbknEggCPB/imwrycgxX2NzoMCHhPkDwqYMr9tRcP5qNrMZHkVnOjRMWwLCcr8ohBVb1OMjxLwGCvjTikrsBOiA6fNyCrm8V1rP93iVPpwaE+gO0SsWmPiXB+jikdf6SizrT5qKasx5j8ABbHpFTx+vFXp9EnYQmLx02h1QTTrl6eDqxLnGjporxl3NL3agEvXdT0WmEost648sQOYAeJS9Q7bfUVoMGnjo4AZdUMQku50McDcMWcBPvr0SzbTAFDfvJqwLzgxwATnCgnp4wDl6Aa+Ax283gghmj+vj7feE2KBBRMW3FzOpLOADl0Isb5587h/U4gGvkt5v60Z1VLG8BhYjbzRwyQZemwAd6cCR5/XFWLYZRIMpX39AR0tjaGGiGzLVyhse5C9RKC6ai42ppWPKiBagOvaYk8lO7DajerabOZP46Lby5wKjw1HCRx7p9sVMOWGzb/vA1hwiWc6jm3MvQDTogQkiqIhJV0nBQBTU+3okKCFDy9WwferkHjtxib7t3xIUQtHxnIwtx4mpg26/HfwVNVDb4oI9RHmx5WGelRVlrtiw43zboCLaxv46AZeB3IlTkwouebTr1y2NjSpHz68WNFjHvupy3q8TFn3Hos2IAk4Ju5dCo8B3wP7VPr/FGaKiG+T+v+TQqIrOqMTL1VdWV1DdmcbO8KXBz6esmYWYKPwDL5b5FA1a0hwapHiom0r/cKaoqr+27/XcrS5UwSMbQAAAABJRU5ErkJggg==)](https://deepwiki.com/Ascend/AgentSDK)
</div>

# Directory Structure

```text
│  __init__.py
│
├─base
│  │  __init__.py
│  │
│  ├─log
│  │      loggers.py
│  │      __init__.py
│  │
│  ├─utils
│  │      checker.py
│  │      class_loader.py
│  │      data_loader.py
│  │      file_utils.py
│  │      get_local_rank.py
│  │      logger_patch.py
│  │      ray_secure_init.py
│  │      __init__.py
│  │
│  └─weight_loaders
│          megatron_weight_loaders.py
│          __init__.py
│
├─configs
│      agentic_rl_config.py
│      ray_env_config.py
│      __init__.py
│
├─data_manager
│      data_manager.py
│      data_registry.py
│      data_transform.py
│      mindspeed_rl_data.py
│      __init__.py
│
├─memory
│      constants.py
│      memory_base.py
│      memory_config.py
│      memory_simple.py
│      memory_summary.py
│      prompts.py
│      summary_client.py
│      token_counter.py
│      utils.py
│      __init__.py
│
├─runner
│  │  runner_worker.py
│  │  __init__.py
│  │
│  ├─agent_engine_wrapper
│  │      base.py
│  │      base_engine_wrapper.py
│  │      __init__.py
│  │
│  └─infer_adapter
│      │  async_server.py
│      │  async_server_base.py
│      │  infer_registry.py
│      │  __init__.py
│      │
│      └─vllm
│          │  base_inference_engine.py
│          │  cache_manager.py
│          │  memory_manager.py
│          │  vllm_async_server.py
│          │  vllm_megatron_weight_loaders.py
│          │  vllm_worker.py
│          │  weight_manager.py
│          │  __init__.py
│          │
│          └─patch
│                  ca_mem_sleep.py
│                  worker_v1_sleep.py
│                  __init__.py
│
└─trainer
    │  main.py
    │  __init__.py
    │
    ├─rollout
    │      rollout_worker.py
    │      __init__.py
    │
    └─train_adapter
        │  __init__.py
        │
        └─mindspeed_rl
            │  agent_grpo_trainer.py
            │  train_agent_grpo.py
            │  __init__.py
            │
            ├─configs
            │      parse_config.py
            │      __init__.py
            │
            ├─patch
            │      compute_utils.py
            │      get_current_node_ip.py
            │      grpo_actor_loss_func.py
            │      launcher.py
            |      logprob_computer.py
            │      __init__.py
            │
            └─workers
                    actor_hybrid_worker.py
                    integrated_worker.py
                    __init__.py
```

# Release Notes

For details about the Agent SDK version mapping, see [Version Mapping](docs/en/release_notes.md#version-mapping).

# Compatibility Information

For Agent SDK version compatibility information, see [Version Compatibility](docs/en/release_notes.md#version-compatibility).

# Environment Deployment

You can install Agent SDK from source. For detailed steps, see the [Installation Guide](docs/en/installation_guide.md).

# Quick Start

Get started with Agent SDK by running a complete Agent loop example. The example demonstrates tool definition, agent execution, and trajectory observation. The quick start includes creating a custom `BaseEngineWrapper` implementation, configuring training parameters, and launching the `agentic_rl` command.

- See [Quick Start](docs/en/quick_start.md) for a hands-on tutorial.

- See the [Examples and Guidance](docs/en/user_guide/user_guide.md) for custom agent examples.

# Features

- For Agent SDK environment variables, model support, and backend support, see [Appendix](docs/en/appendix.md).

# API Reference

See [Python API](docs/en/api_python.md) and [CLI API](docs/en/command_api.md) for the API reference.

# FAQ

For FAQ, see [FAQ](docs/en/faq.md).

# Security Statement

- When you use an API to read a file, ensure that you own the file and that its permissions are no more permissive than `640`. This helps prevent privilege escalation and similar security issues. Software code or programs downloaded from external sources may pose risks. You must guarantee the security of their functions.
- Communication matrix: The Agent SDK development kit does not actively open or depend on any port. Therefore, no communication matrix is involved.
- For public network addresses, see [Public network addresses](docs/en/resource/AgentSDK_public_network_addresses_0000002516443057.xlsx). URLs in the Agent SDK installation package are removed after installation, so they are not accessed and do not pose a risk.
- For the security hardening guide, see [Agent SDK Security Hardening Guide](docs/en/security_hardening.md).

# Branch Maintenance Strategy

Version branches follow the defined maintenance phases.

| Status| Time| Description|
|------|------|------|
| Planning| 1 to 3 months| Feature planning|
| Development| 3 months| New feature development and issue fixes, released regularly|
| Maintenance| 3 to 12 months| Regular branches receive 3 months of maintenance. Long-term support branches receive 12 months of maintenance. Only major bugs are fixed. No new features are added.|
| End of life (EOL)| N/A | The branch no longer accepts any changes.|

# Version Maintenance Strategy

| Version   | Maintenance Strategy| Current Status| Release Date| Next Status| EOL Date|
|-------|----------|----------|----------|----------|---------|
| master | Long-term support| Development| Under active development and not yet released| Continuous development| - |
| v26.0 | Regular branch| Maintenance| 2026-01-28 | Expected to enter unsupported status on April 28, 2026.| 2026-04-28 |

# Disclaimer

- This repository contains multiple development branches, which may include unfinished, experimental, or untested features. These branches should not be used in any production environment or service-critical projects before an official release. Ensure you use our official release versions to guarantee stability and security.
This project and its contributors are not responsible for any issues, losses, or data corruption resulting from the use of development branches.

- For version update notes, see [Release Notes](docs/en/release_notes.md#change-description).

# License

Agent SDK is licensed under Mulan PSL v2. The license text can be found in [LICENSE](../LICENSE.md).

Documents in the `docs` directory of Agent SDK are licensed under CC-BY 4.0. For details, see [LICENSE](./docs/LICENSE).

# Contribution Statement

- Before contributing, sign the [Open Project Contributor License Agreement (CLA)](https://clasign.osinfra.cn/sign/gitee_ascend-1611222220829317930).
- If you encounter a bug, submit an [issue](https://gitcode.com/Ascend/AgentSDK/issues).
- If you plan to contribute bug fixes, submit a pull request (PR). See [Contribution Requirements](../contributing.md).
- If you plan to contribute new features or functionality, create an issue to discuss it with us first. Describe the background or purpose of the requirement, the design, and its impact on existing APIs. Submitting a PR without prior discussion may lead to rejection, as the evolution direction of the project might differ from your ideas.
- For a detailed contribution process, see the [Contribution Guide](../contributing.md).

# Suggestions and Feedback

You are welcome to contribute to the community. If you have any questions or suggestions, please submit a [Issues](https://gitcode.com/Ascend/AgentSDK/issues). We will reply as soon as possible. Thank you for your support.

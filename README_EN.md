# Agent SDK

[Agent SDK](https://gitcode.com/Ascend/AgentSDK) is an Ascend-native agent ecosystem repository designed to provide out-of-the-box ecosystem tools and framework support for agent applications on Ascend NPU infrastructure, helping developers quickly build and deploy AI agents.

Agent SDK currently covers multiple areas, including the agentic RL training, inference, and tuning framework and agent application development, and will continue to expand with more agent ecosystem components.

## Main Products

The Agent SDK repository contains the following two core products. Click the links to learn more about each product:

| Product | Description |
|------|------|
| **[Aura](./aura/README.md)** | An integrated agentic RL training, inference, and tuning framework that supports both co-located and separated training and inference modes, integrating with multiple training, inference, and agent engines. |
| **[Openclaw](./openclaw/README.md)** | A multi-domain agent framework and service built on OpenClaw, integrating domain-specific capabilities such as code generation and research analysis. |

## Other Directories

| Directory | Description |
|------|------|
| [docker](./docker) | Provides Dockerfiles and environment setup scripts for Agent SDK to quickly build container images and configure runtime environments |
| [docs](./docs) | Provides documentation for Agent SDK, including installation guides, quick starts, API references, and FAQs |
| [pre-commit](./pre-commit) | Provides automated pre-commit check configurations, including spelling checks, code style checks, and security scans |
| [presmoke](./presmoke) | Provides pre-smoke test cases for Agent SDK to verify basic framework functionality and configuration correctness before formal deployment |
| [script](./script) | Provides auxiliary scripts for building, testing, and installing Agent SDK |

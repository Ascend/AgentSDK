# MindClaw Docker

> English | [中文](./OVERVIEW.zh.md)

## Quick Reference

- MindClaw is maintained by [MindClaw](https://gitcode.com/chadwweng/mindclaw)

- Where to get help

    - [MindClaw Documentation](https://gitcode.com/chadwweng/mindclaw/blob/dev/README.md)
    - [Issue Feedback](https://gitcode.com/chadwweng/mindclaw/issues)

## MindClaw Docker

MindClaw is a highly usable multi-domain Agent framework and service built on OpenClaw, integrating vertical domain Agent capabilities such as code generation, web search, research analysis, and mathematical calculation. This document introduces the Docker image architecture design and usage of MindClaw.

## Supported Tags and Dockerfile Usage

### Tag Specification

Tags follow the format:

```text
<MindClaw Version>-<Architecture>
```

| Field | Example Value | Description |
|-------|---------------|-------------|
| MindClaw Version | 2026.4.11 | Corresponding to the official OpenClaw release version identifier |
| Architecture | x86 / aarch64 | System architecture supported by the image |

### Image Repository Address

MindClaw images are hosted on the Atlas Community Image Repository:

```text
https://www.hiascend.com/developer/ascendhub
```

**Complete Image Examples:**

```text
swr.cn-south-1.myhuaweicloud.com/ascendhub/openclaw:2026.4.11-aarch64
swr.cn-south-1.myhuaweicloud.com/ascendhub/openclaw:2026.4.11-x86
```

### Image Architecture

MindClaw adopts a three-layer image architecture design:

| Layer | Image Name | Description | Dockerfile |
|-------|------------|-------------|------------|
| **Layer 1** | `openclaw-base` | Infrastructure including SSH/uv/bun/pnpm/npm global packages/pip packages/Playwright | [Dockerfile.openclaw-base](./Dockerfile.openclaw-base) |
| **Layer 2** | `openclaw` | Official OpenClaw build artifacts | [Dockerfile.openclaw-app](./Dockerfile.openclaw-app) |
| **Layer 3** | `openclaw-overlay` | Custom layer with claude-mem + ccb + subagent-coordinator + ChromaDB | [Dockerfile.openclaw-overlay](./Dockerfile.openclaw-overlay) |

### Build Parameters

| Parameter | Description | Required | Example Value |
|-----------|-------------|----------|---------------|
| VERSION | OpenClaw version number | No | 2026.4.11 |
| REGISTRY | Image registry prefix | No | localhost |
| SKIP_BASE | Whether to skip base image build (default false) | No | true |
| SKIP_APP | Whether to skip app image build (default false) | No | true |
| SKIP_OVERLAY | Whether to skip overlay image build (default false) | No | true |
| OPENCLAW_SRC | OpenClaw source code directory | No | /path/to/openclaw-src |
| CLAUDE_MEM_SRC | claude-mem source code directory | No | /path/to/claude-mem-src |
| CLAUDE_CODE_SRC | claude-code-best source code directory | No | /path/to/claude-code-best |
| INCLUDE_CLAUDE_CODE_BEST | Whether to include claude-code-best (default false) | No | true |
| DOCKER_REGISTRY_NPM | Docker image registry NPM mirror prefix | No | https://registry.npmmirror.com |

## Quick Start

### Build MindClaw Images

Use the unified build script to build images:

```bash
# Build all three layers of images with one click
cd mindclaw/docker
bash ./build-openclaw.sh

# Only build the final image (skip base and app, use existing images)
bash ./build-openclaw.sh --skip-base --skip-app
```

### Run MindClaw Containers

Use the deployment script to run containers:

```bash
# Quick deployment (single instance)
API_KEY=api-key bash ./scripts/deploy.sh quick -n 1 -m mode-name -u http://xxxx.xx.xx.xx:xxxx -p xxxx -i openclaw:2026.4.11-x86 --skills --name openclaw-test
```

### How to Develop Secondarily

```bash
# Use MindClaw image as base image and add user software
FROM openclaw:2026.4.11-x86

# Install additional dependencies
RUN apt update -y && \
    apt install -y --no-install-recommends \
        extra-package && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install --break-system-packages \
    -i https://mirrors.aliyun.com/pypi/simple \
    extra-python-package

# Copy user-defined skills
COPY --chown=root:root ./custom-skills/ /home/node/.openclaw/skills/
RUN chown -R node:node /home/node/.openclaw/skills/ && \
    chmod -R 755 /home/node/.openclaw/skills/

WORKDIR /app
CMD ["node", "/app/dist/index.js"]
```

## Supported Hardware Architectures

| Architecture | Description |
|--------------|-------------|
| x86_64 | Intel/AMD 64-bit architecture |
| aarch64 | ARM 64-bit architecture |

## License

As with all container images, pre-installed software packages (Python, Node.js, system libraries, etc.) may be subject to their own license restrictions.

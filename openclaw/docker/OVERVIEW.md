# AgentSDK Openclaw Docker

> English | [中文](./OVERVIEW.zh.md)

## Quick Reference

- AgentSDK Openclaw is maintained by [AgentSDK Openclaw](https://gitcode.com/Ascend/AgentSDK/tree/master/openclaw)

- Where to get help

    - [AgentSDK Openclaw Documentation](https://gitcode.com/Ascend/AgentSDK/blob/master/openclaw/README.md)
    - [Issue Feedback](https://gitcode.com/Ascend/AgentSDK/issues)

## AgentSDK Openclaw Docker

AgentSDK Openclaw is a highly usable multi-domain Agent framework and service built on OpenClaw, integrating vertical domain Agent capabilities such as code generation, web search, research analysis, and mathematical calculation. This document introduces the Docker image architecture design and usage of AgentSDK Openclaw.

## Supported Tags and Dockerfile Usage

### Tag Specification

Tags follow the format:

```text
<Openclaw Version>
```

| Field | Example Value | Description |
|-------|---------------|-------------|
| Openclaw Version | 2026.5.22 | Corresponding to the official OpenClaw release version identifier |

### Image Repository Address

AgentSDK Openclaw images are hosted on the Atlas Community Image Repository:

```text
https://www.hiascend.com/developer/ascendhub
```

**Complete Image Examples:**

```text
swr.cn-south-1.myhuaweicloud.com/ascendhub/openclaw:2026.5.22
```

### Image Architecture

AgentSDK Openclaw adopts a three-layer image architecture design:

| Layer | Image Tag | Description | Dockerfile |
|-------|-----------|-------------|------------|
| **Layer 1** | `openclaw:base-{version}` | Infrastructure including SSH/uv/bun/pnpm/npm global packages/pip packages/Playwright/Chromium/gosu/ffmpeg/rsync | [Dockerfile.openclaw-base](./Dockerfile.openclaw-base) |
| **Layer 2** | `openclaw:app-{version}` | Official OpenClaw build artifacts (multi-stage build) | [Dockerfile.openclaw-app](./Dockerfile.openclaw-app) |
| **Layer 3** | `openclaw:{version}` | Custom layer with subagent-coordinator + Hermes + skills | [Dockerfile.openclaw-overlay](./Dockerfile.openclaw-overlay) |

### Build Parameters

| Parameter | Description | Required | Example Value |
|-----------|-------------|----------|---------------|
| VERSION | OpenClaw version number | No | 2026.5.22 |
| REGISTRY | Image registry prefix | No | localhost |
| OFFLINE | Offline mode, use local sources only (default false) | No | true |
| SKIP_BASE | Whether to skip base image build (default false) | No | true |
| SKIP_APP | Whether to skip app image build (default false) | No | true |
| SKIP_OVERLAY | Whether to skip overlay image build (default false) | No | true |
| SKIP_PLUGINS | Whether to skip plugin preparation (default false) | No | true |
| OPENCLAW_SRC | OpenClaw source code directory | No | /path/to/openclaw-src |
| HERMES_SRC | Hermes Agent source code directory | No | /path/to/hermes-src |
| DOCKER_REGISTRY_NPM | NPM mirror URL | No | https://registry.npmmirror.com |

## Quick Start

### Build AgentSDK Openclaw Images

Use the unified build script to build images:

```bash
# Build all three layers of images with one click
cd AgentSDK/openclaw/docker
bash ./build-openclaw.sh

# Only build the final image (skip base and app, use existing images)
bash ./build-openclaw.sh --skip-base --skip-app
```

### Run AgentSDK Openclaw Containers

Use the deployment script to run containers:

```bash
# Quick deployment (single instance)
API_KEY=api-key bash ./scripts/deploy.sh quick -n 1 -m mode-name -u http://xxxx.xx.xx.xx:xxxx -p xxxx -i openclaw:2026.5.22 --skills --name openclaw-test
```

### How to Develop Secondarily

```bash
# Use AgentSDK Openclaw image as base image and add user software
FROM openclaw:2026.5.22

# Install additional dependencies
RUN apt update -y && \
    apt install -y --no-install-recommends \
        extra-package && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install --break-system-packages \
    -i https://mirrors.aliyun.com/pypi/simple \
    extra-python-package

# Copy user-defined skills (deploy script extracts from skills-shared and merge-mounts)
COPY --chown=root:root ./custom-skills/ /home/node/.openclaw/skills-shared/
RUN chown -R node:node /home/node/.openclaw/skills-shared/ && \
    chmod -R 755 /home/node/.openclaw/skills-shared/

WORKDIR /app
CMD ["node", "openclaw.mjs", "gateway"]
```

### Skills Mounting Mechanism

The deploy script automatically merges image built-in skills with host custom skills and bind-mounts the result when starting containers:

| Stage | Path | Description |
|-------|------|-------------|
| In image | `/home/node/.openclaw/skills-shared/` | Skills packaged in the overlay layer (extracted at deploy time, shadowed by bind mount at runtime) |
| Host | `openclaw-configs/skills-merged/` | Merge directory (image skills + host skills union, host takes priority) |
| Container | `/home/node/.openclaw/skills/` | bind mount from `skills-merged/`, where Gateway loads skills |

**Adding skills dynamically** (after container is running, no image rebuild needed):

```bash
# Option 1: Operate directly on the host merge directory
cp -r ./my-skill openclaw-configs/skills-merged/my-skill/
chmod -R 755 openclaw-configs/skills-merged/my-skill/

# Option 2: Operate via container path (equivalent, since it's a bind mount)
docker cp ./my-skill openclaw-1:/home/node/.openclaw/skills/my-skill/
```

Restart Gateway after adding skills: enter `/restart` in the WebUI, or run `docker exec openclaw-1 pkill -u node -f gateway` (the health monitor will auto-restart it).

## Supported Hardware Architectures

| Architecture | Description |
|--------------|-------------|
| x86_64 | Intel/AMD 64-bit architecture |
| aarch64 | ARM 64-bit architecture |

## License

As with all container images, pre-installed software packages (Python, Node.js, system libraries, etc.) may be subject to their own license restrictions.

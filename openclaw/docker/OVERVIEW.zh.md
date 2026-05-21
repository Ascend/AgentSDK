# MindClaw Docker

> [English](./OVERVIEW.md) | 中文

## 快速参考

- MindClaw 由 [MindClaw](https://gitcode.com/chadwweng/mindclaw) 维护

- 从哪里获取帮助

    - [MindClaw 文档](https://gitcode.com/chadwweng/mindclaw/blob/dev/README.md)
    - [问题反馈](https://gitcode.com/chadwweng/mindclaw/issues)

## MindClaw Docker

MindClaw 是基于 OpenClaw 构建的高易用性多领域 Agent 框架与服务，集成代码生成、网络搜索、研究分析、数学计算等垂直领域 Agent 能力。本文档介绍 MindClaw 的 Docker 镜像架构设计与使用方法。

## 支持的 Tags 及 Dockerfile 使用方法

### Tag 规范

Tag 遵循以下格式：

```text
<MindClaw 版本号>-<架构>
```

| 字段          | 示例值          | 说明                             |
|-------------|---------------|--------------------------------|
| MindClaw 版本号 | 2026.4.11         | 对应 OpenClaw 官方发布版本标识        |
| 架构   | x86 / aarch64        | 镜像支持的系统架构                |

### 镜像仓库地址

MindClaw 镜像托管在昇腾社区镜像仓库：

```text
https://www.hiascend.com/developer/ascendhub
```

**完整镜像示例：**

```text
swr.cn-south-1.myhuaweicloud.com/ascendhub/openclaw:2026.4.11-aarch64
swr.cn-south-1.myhuaweicloud.com/ascendhub/openclaw:2026.4.11-x86
```

### 镜像架构

MindClaw 采用三层镜像架构设计：

| 层级 | 镜像名称 | 说明 | 构建文件 |
|------|---------|------|----------|
| **Layer 1** | `openclaw-base` | SSH/uv/bun/pnpm/npm全局包/pip包/Playwright 等基础设施 | [Dockerfile.openclaw-base](./Dockerfile.openclaw-base) |
| **Layer 2** | `openclaw` | 官方 OpenClaw 构建产物 | [Dockerfile.openclaw-app](./Dockerfile.openclaw-app) |
| **Layer 3** | `openclaw-overlay` | claude-mem + ccb + subagent-coordinator + ChromaDB 定制层 | [Dockerfile.openclaw-overlay](./Dockerfile.openclaw-overlay) |

### 构建参数

| 参数               | 说明                               | 必填 | 示例值                                                |
|------------------|----------------------------------|----|----------------------------------------------------|
| VERSION     | OpenClaw 版本号                    | 否  | 2026.4.11                                              |
| REGISTRY | 镜像仓库前缀 | 否 | localhost |
| SKIP_BASE | 是否跳过基础镜像构建（默认 false） | 否 | true |
| SKIP_APP | 是否跳过应用镜像构建（默认 false） | 否 | true |
| SKIP_OVERLAY | 是否跳过扩展镜像构建（默认 false） | 否 | true |
| OPENCLAW_SRC | OpenClaw 源码目录 | 否 | /path/to/openclaw-src |
| CLAUDE_MEM_SRC       | claude-mem 源码目录                 | 否  | /path/to/claude-mem-src                           |
| CLAUDE_CODE_SRC | claude-code-best 源码目录                       | 否  | /path/to/claude-code-best                                              |
| INCLUDE_CLAUDE_CODE_BEST | 是否包含 claude-code-best（默认 false） | 否  | true |
| DOCKER_REGISTRY_NPM | Docker 镜像仓库 NPM 镜像前缀 | 否 | https://registry.npmmirror.com |

## 快速开始

### 构建 MindClaw 镜像

使用统一构建脚本构建镜像：

```bash
# 一键构建全部三层镜像
cd mindclaw/docker
bash ./build-openclaw.sh

# 仅构建最终镜像（跳过 base 和 app，使用已有镜像）
bash ./build-openclaw.sh --skip-base --skip-app
```

### 运行 MindClaw 容器

使用部署脚本运行容器：

```bash
# 快速部署（单实例）
API_KEY=api-key bash ./scripts/deploy.sh quick -n 1 -m mode-name -u http://xxxx.xx.xx.xx:xxxx -p xxxx -i openclaw:2026.4.11-x86 --skills --name openclaw-test
```

### 如何二次开发

```bash
# 以 MindClaw 镜像为基础镜像，叠加用户软件
FROM openclaw:2026.4.11-x86

# 安装额外依赖
RUN apt update -y && \
    apt install -y --no-install-recommends \
        extra-package && \
    apt clean && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
RUN pip3 install --break-system-packages \
    -i https://mirrors.aliyun.com/pypi/simple \
    extra-python-package

# 复制用户自定义技能
COPY --chown=root:root ./custom-skills/ /home/node/.openclaw/skills/
RUN chown -R node:node /home/node/.openclaw/skills/ && \
    chmod -R 755 /home/node/.openclaw/skills/

WORKDIR /app
CMD ["node", "/app/dist/index.js"]
```

## 支持的硬件架构

| 架构 | 说明 |
|------|------|
| x86_64 | Intel/AMD 64位架构 |
| aarch64 | ARM 64位架构 |

## 许可证

与所有容器镜像一样，预装软件包（Python、Node.js、系统库等）可能受其自身许可证约束。

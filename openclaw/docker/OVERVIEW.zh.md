# AgentSDK Openclaw Docker

> [English](./OVERVIEW.md) | 中文

## 快速参考

- AgentSDK Openclaw 由 [AgentSDK Openclaw](https://gitcode.com/Ascend/AgentSDK/tree/master/openclaw) 维护

- 从哪里获取帮助

    - [AgentSDK Openclaw 文档](https://gitcode.com/Ascend/AgentSDK/blob/master/openclaw/README.md)
    - [问题反馈](https://gitcode.com/Ascend/AgentSDK/issues)

## AgentSDK Openclaw Docker

AgentSDK Openclaw 是基于 OpenClaw 构建的高易用性多领域 Agent 框架与服务，集成代码生成、网络搜索、研究分析、数学计算等垂直领域 Agent 能力。本文档介绍 AgentSDK Openclaw 的 Docker 镜像架构设计与使用方法。

## 支持的 Tags 及 Dockerfile 使用方法

### Tag 规范

Tag 遵循以下格式：

```text
<Openclaw 版本号>
```

| 字段          | 示例值          | 说明                             |
|-------------|---------------|--------------------------------|
| Openclaw 版本号 | 2026.5.22         | 对应 OpenClaw 官方发布版本标识        |

### 镜像仓库地址

AgentSDK Openclaw 镜像托管在昇腾社区镜像仓库：

```text
https://www.hiascend.com/developer/ascendhub
```

**完整镜像示例：**

```text
swr.cn-south-1.myhuaweicloud.com/ascendhub/openclaw:2026.5.22
```

### 镜像架构

AgentSDK Openclaw 采用三层镜像架构设计：

| 层级 | 镜像标签 | 说明 | 构建文件 |
|------|---------|------|----------|
| **Layer 1** | `openclaw:base-{版本号}` | SSH/uv/bun/pnpm/npm全局包/pip包/Playwright/Chromium/gosu/ffmpeg/rsync 等基础设施 | [Dockerfile.openclaw-base](./Dockerfile.openclaw-base) |
| **Layer 2** | `openclaw:app-{版本号}` | 官方 OpenClaw 构建产物（多阶段构建） | [Dockerfile.openclaw-app](./Dockerfile.openclaw-app) |
| **Layer 3** | `openclaw:{版本号}` | subagent-coordinator + Hermes + skills 定制层 | [Dockerfile.openclaw-overlay](./Dockerfile.openclaw-overlay) |

### 构建参数

| 参数               | 说明                               | 必填 | 示例值                                                |
|------------------|----------------------------------|----|----------------------------------------------------|
| VERSION     | OpenClaw 版本号                    | 否  | 2026.5.22                                              |
| REGISTRY | 镜像仓库前缀 | 否 | localhost |
| OFFLINE | 离线模式，仅使用本地源码（默认 false） | 否 | true |
| SKIP_BASE | 是否跳过基础镜像构建（默认 false） | 否 | true |
| SKIP_APP | 是否跳过应用镜像构建（默认 false） | 否 | true |
| SKIP_OVERLAY | 是否跳过扩展镜像构建（默认 false） | 否 | true |
| SKIP_PLUGINS | 是否跳过插件准备（默认 false） | 否 | true |
| OPENCLAW_SRC | OpenClaw 源码目录 | 否 | /path/to/openclaw-src |
| HERMES_SRC | Hermes Agent 源码目录 | 否 | /path/to/hermes-src |
| DOCKER_REGISTRY_NPM | NPM 镜像地址 | 否 | https://registry.npmmirror.com |

## 快速开始

### 构建 AgentSDK Openclaw 镜像

使用统一构建脚本构建镜像：

```bash
# 一键构建全部三层镜像
cd AgentSDK/openclaw/docker
bash ./build-openclaw.sh

# 仅构建最终镜像（跳过 base 和 app，使用已有镜像）
bash ./build-openclaw.sh --skip-base --skip-app
```

### 运行 AgentSDK Openclaw 容器

使用部署脚本运行容器：

```bash
# 快速部署（单实例）
API_KEY=api-key bash ./scripts/deploy.sh quick -n 1 -m mode-name -u http://xxxx.xx.xx.xx:xxxx -p xxxx -i openclaw:2026.5.22 --skills --name openclaw-test
```

### 如何二次开发

```bash
# 以 AgentSDK Openclaw 镜像为基础镜像，叠加用户软件
FROM openclaw:2026.5.22

# 安装额外依赖
RUN apt update -y && \
    apt install -y --no-install-recommends \
        extra-package && \
    apt clean && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
RUN pip3 install --break-system-packages \
    -i https://mirrors.aliyun.com/pypi/simple \
    extra-python-package

# 复制用户自定义技能（部署脚本会从 skills-shared 提取并合并挂载）
COPY --chown=root:root ./custom-skills/ /home/node/.openclaw/skills-shared/
RUN chown -R node:node /home/node/.openclaw/skills-shared/ && \
    chmod -R 755 /home/node/.openclaw/skills-shared/

WORKDIR /app
CMD ["node", "openclaw.mjs", "gateway"]
```

### 技能挂载机制

部署脚本启动容器时，会自动将镜像内置技能与宿主机自定义技能取并集后挂载：

| 阶段 | 路径 | 说明 |
|------|------|------|
| 镜像内 | `/home/node/.openclaw/skills-shared/` | 覆盖层打包的技能（部署时提取，运行时被 bind mount 遮蔽） |
| 宿主机 | `openclaw-configs/skills-merged/` | 合并目录（镜像技能 + 宿主机技能取并集，宿主机优先） |
| 容器内 | `/home/node/.openclaw/skills/` | bind mount 自 `skills-merged/`，Gateway 加载技能的路径 |

**动态添加技能**（容器运行后，无需重建镜像）：

```bash
# 方式一：宿主机直接操作合并目录
cp -r ./my-skill openclaw-configs/skills-merged/my-skill/
chmod -R 755 openclaw-configs/skills-merged/my-skill/

# 方式二：通过容器内路径操作（等价，因是 bind mount）
docker cp ./my-skill openclaw-1:/home/node/.openclaw/skills/my-skill/
```

放入后重启 Gateway 使其生效：在 WebUI 中输入 `/restart`，或执行 `docker exec openclaw-1 pkill -u node -f gateway`（健康探针会自动拉起）。

## 支持的硬件架构

| 架构 | 说明 |
|------|------|
| x86_64 | Intel/AMD 64位架构 |
| aarch64 | ARM 64位架构 |

## 许可证

与所有容器镜像一样，预装软件包（Python、Node.js、系统库等）可能受其自身许可证约束。

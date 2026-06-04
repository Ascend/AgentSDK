# subagent-coordinator 插件

`subagent-coordinator` 是一组面向 OpenClaw 的主子代理协同插件，并配套提供同名 Skill。它用于让主代理在处理复杂任务时先进行任务分析、复杂度评分和 L1-L5 算子分级，再把适合委派的任务交给 worker subagent 或 ACP runtime 执行。

本文档说明如何安装这组插件，以及在 OpenClaw 中如何使用它们。

## 组件说明

本目录包含两部分内容：

| 组件 | 路径 | 作用 |
|------|------|------|
| Skill | `skill/` | 提供任务分解、算子分级、路由策略和使用说明，是用户在 OpenClaw 中触发 subagent-coordinator 的入口。 |
| 插件 | `plugins/subagent-taskr/` | 提供层级任务规划、任务持久化、复杂度评分和任务分解工具。 |
| 插件 | `plugins/subagent-exec-monitor/` | 提供执行前后质量门控、检查点和重试策略相关能力。 |
| 插件 | `plugins/subagent-observability/` | 提供指标、追踪、成本、限流状态和数据脱敏等可观测性能力。 |
| 类型包 | `packages/types/` | 提供插件间共享的事件、payload 和服务类型定义。 |

Skill 是使用入口；插件是增强层。即使插件没有全部加载，Skill 仍可提供基础的任务分析和路由逻辑；插件加载后会通过事件与工具能力增强任务持久化、质量门控和可观测性。

## 前置条件

安装前需要满足以下条件：

- OpenClaw 命令可用，即可以执行 `openclaw`。
- 如需从源码构建，需安装 Node.js、pnpm 和 TypeScript 相关依赖。
- OpenClaw 中需要有可被主代理调用的 worker subagent。
- 如需执行 `--build`，构建过程需要能解析 OpenClaw `plugin-sdk`。在完整 OpenClaw 源码树中它通常位于 `packages/plugin-sdk`；在容器镜像中可能位于 `/app/packages/plugin-sdk`；宿主机全局安装的 OpenClaw 通常只提供编译后 SDK，例如 `$(dirname $(dirname $(readlink -f $(command -v openclaw))))/lib/node_modules/openclaw/dist/plugin-sdk`。安装脚本会自动尝试这些路径；如果无法发现，可通过 `PLUGIN_SDK_SRC=/path/to/plugin-sdk` 显式指定。

## 快速安装

在本目录执行：

```bash
./install-sc-local.sh --build
```

该脚本会自动完成以下操作：

1. 检查 `openclaw` 命令和 OpenClaw 主目录。
2. 执行 `pnpm install` 和 `pnpm build`（因指定了 `--build`）。
3. 为每个插件生成/修正根 `index.js` 入口，使其指向编译后的 `dist/` 路径。
4. 将 `plugins/` 下的插件安装到 `~/.openclaw/workspace/plugins/`。
5. 将 `skill/` 安装到 `~/.openclaw/skills/subagent-coordinator/`，确保 `openclaw skills info` 可发现。
6. 调用 `openclaw plugins install --link` 把插件注册到 OpenClaw。
7. 若 OpenClaw 服务已部署，尝试创建 `worker` 子代理。
8. 在 `~/.openclaw/openclaw.json` 中配置 `main -> worker` 的调用权限。
9. 为 `worker` 配置 exec allowlist。
10. 重新启动 Gateway。

默认安装方式是**复制文件**到 `~/.openclaw/workspace/plugins/`，修改源码后需重新执行脚本。

> **注意**：如果目标文件系统不支持 POSIX `chmod`（例如 WSL2 以 9p/drvfs 挂载 Windows 目录的环境），安装脚本会在开始时自动探测并切换为符号链接模式，以避免被 OpenClaw 安全扫描以 `world-writable path` 拦截。脚本会输出一行 WARN 提示该自动切换。详见下文"容器环境的注意事项"。

如果希望在开发调试时使用符号链接（修改源码后无需重新安装即可生效），执行：

```bash
./install-sc-local.sh --build --symlink
```

如果 OpenClaw 主目录不是默认的 `~/.openclaw`，使用：

```bash
./install-sc-local.sh --openclaw-home /path/to/.openclaw --build
```

也可以通过环境变量指定：

```bash
OPENCLAW_HOME=/path/to/.openclaw ./install-sc-local.sh --build
```

如果宿主机不是完整 OpenClaw 源码树，且脚本无法自动找到 `plugin-sdk`，可显式指定全局 OpenClaw 安装中的 SDK：

```bash
PLUGIN_SDK_SRC=/path/to/openclaw/dist/plugin-sdk ./install-sc-local.sh --build
```

如果 `pnpm install` 提示 lockfile 与 `package.json` 不一致，应先更新并提交 `pnpm-lock.yaml`，确保依赖声明和锁文件同步；不要在构建产物缺失时继续安装。

### 安装选项

| 选项 | 说明 |
|------|------|
| `--build` | 安装前执行 `pnpm install` 和 `pnpm build`。 |
| `--symlink` | 使用符号链接代替复制（开发场景；默认是复制）。 |
| `--openclaw-home <path>` | 指定 OpenClaw 主目录。 |
| `--skip-install` | 跳过 `openclaw plugins install` 调用，仅链接/复制插件文件。 |

## 安装到容器环境

如果 OpenClaw 运行在容器中，需要先把本目录挂载到容器内，然后在容器内执行安装脚本。例如：

```bash
docker run -d \
  --name openclaw \
  -v /path/to/subagent-coordinator:/workspace/subagent-coordinator \
  openclaw-image

docker exec openclaw bash -c "cd /workspace/subagent-coordinator && ./install-sc-local.sh --build"
```

如果容器内 OpenClaw 主目录不是 `~/.openclaw`，需要显式指定：

```bash
docker exec openclaw bash -c \
  "cd /workspace/subagent-coordinator && ./install-sc-local.sh --openclaw-home /home/node/.openclaw --build"
```

### 容器环境的注意事项

容器中安装和重启 OpenClaw 存在以下常见问题：
1.**脚本行尾格式**：在 Windows 上编辑后的脚本可能出现 CRLF 行尾，导致容器内 `bash` 报 `set -e` 失败或命令找不到。安装前请确保脚本使用 LF 行尾：

```bash
sed -i 's/\r$//' install-sc-local.sh
```

2.**plugin-sdk 链接**：`pnpm install` 需要解析 `openclaw/plugin-sdk`，如果 OpenClaw 镜像内置了 SDK，脚本会自动链接 `/app/packages/plugin-sdk`；如果不在镜像内且本目录不在 OpenClaw 源码树中，可通过 `PLUGIN_SDK_SRC` 环境变量显式指定：

```bash
PLUGIN_SDK_SRC=/path/to/openclaw/packages/plugin-sdk \
  ./install-sc-local.sh --build
```

3.**插件目录权限**：容器中挂载的目录可能被设置为 `777`（world-writable）且由非 root 用户拥有，OpenClaw 会拒绝加载并提示 `blocked plugin candidate: world-writable path` 或 `suspicious ownership`。

   - **常规文件系统（ext4、tmpfs 等）**：安装脚本会自动修复权限（`chmod 755`、`chown root:root`），但仅在以 root 运行时 `chown` 生效。

   - **WSL2 以 9p/drvfs 挂载 Windows 目录**：此环境下 `chmod` 会被文件系统静默忽略（退出码 0 但 mode 不变），权限修复无效；此时 OpenClaw 会以 `world-writable path` 拦截所有复制出来的插件目录。

     安装脚本在 `check_environment` 之后会执行一次 chmod 探针：在 `$OPENCLAW_HOME` 下建临时目录并 `chmod 755`，若读到的 mode 末位仍为 7，则判定当前文件系统忽略 chmod，自动把 `USE_SYMLINK` 置为 `true`，让插件以符号链接方式部署到 `~/.openclaw/workspace/plugins/`，从根上避开 world-writable 拦截。脚本会输出一行 WARN 提示该自动切换，例如：

     ```text
     [WARN] 检测到目标文件系统不支持 POSIX chmod (probe mode=777，可能为 WSL2 9p/drvfs)，自动切换为 --symlink 模式以避开 'world-writable path' 拦截
     ```

     符号链接在 OpenClaw 中只会产生 `untracked local code` 警告（不是 block），可通过 `openclaw.json` 中的 `plugins.allow` 或 `openclaw plugins install --link` 注册记录来固定信任（参见"重要：宿主机上必须设置 `OPENCLAW_HOME`"一节）。

     如果手动运行的宿主机既不是 9p 又没有合适的修复手段，可显式加 `--symlink` 走符号链接路径绕开此问题。

4.**node_modules 符号链接**：`pnpm` 在 `node_modules/` 中创建的符号链接会指向 `.pnpm/` 内容存储目录，被 OpenClaw 安全扫描判定为"install root 之外的依赖"。安装脚本会在调用 `openclaw plugins install --link` 前移除插件目录中的 pnpm 符号链接，并复制插件运行时依赖（`@sinclair/typebox`、`@subagent-coordinator/types` 与 `openclaw` 包），避免安全扫描失败和运行时依赖缺失。在 `--symlink` 模式下，这些清理会通过符号链接作用于源码目录的 `node_modules/`（`openclaw plugins install --link` 才能在 link 模式下通过安全扫描）。

5.**Gateway 重启**：容器中通常没有 systemd，`openclaw gateway restart` 会失败。安装脚本会自动降级为 `pkill` + `nohup openclaw gateway` 后台启动的方式。如果手动重启，使用：

```bash
pkill -f openclaw-gateway
OPENCLAW_HOME=/home/node/.openclaw nohup openclaw gateway > /tmp/openclaw-gateway.log 2>&1 &
```

## 构建到 OpenClaw 镜像

OpenClaw 镜像构建流程可通过构建脚本调用本目录的 `install_to_image.sh`：

1. 准备 OpenClaw `plugin-sdk`。
2. 执行 `pnpm install`。
3. 执行 `pnpm build`。
4. 将每个插件的构建产物整理到 `dist/plugins/`。
5. 将共享类型包嵌入每个插件的 `node_modules/@subagent-coordinator/types/` 中。

如果需要单独准备镜像产物，可以在本目录执行：

```bash
./install_to_image.sh
```

产物会输出到：

```text
openclaw/plugins/subagent-coordinator/dist/
```

随后 Dockerfile 把 `dist/plugins/` 复制到镜像内的 OpenClaw 扩展目录即可，此外需要注意`subagent-coordinator/skill`也需要拷贝到镜像内对应路径（例如 `~/.openclaw/skills/`）。

## 验证安装

安装后先确认 Gateway 已启动（脚本会自动重启），然后检查插件是否加载：

```bash
openclaw plugins list | grep subagent
```

应能看到以下三个插件，且三项都必须出现才表示插件安装通过：

```text
@subagent-coordinator/taskr
@subagent-coordinator/exec-monitor
@subagent-coordinator/observability
```

检查 Skill 是否可用：

```bash
openclaw skills info subagent-coordinator
```

### 关于 OpenClaw 状态目录

OpenClaw 的 canonical config 位于 `$OPENCLAW_STATE_DIR/openclaw.json`（默认 `~/.openclaw/openclaw.json`）。宿主机 Gateway、`openclaw` CLI 和 `openclaw plugins install --link` 都读写这一份文件。`install-sc-local.sh` 内部以 `OPENCLAW_STATE_DIR` 指向安装位置（由 `--openclaw-home` 控制，默认 `~/.openclaw`），因此安装产物与 Gateway 配置始终一致，不会出现 entries 缺失或静默丢弃插件的情况。

`openclaw.json` 中与本插件相关的字段由以下来源维护：

| 字段 | 维护方式 |
|------|----------|
| `plugins.entries.<id>` | 由 `openclaw plugins install --link` 写入（CLI 行为） |
| `plugins.installs.<id>` | 由 `openclaw plugins install --link` 写入（CLI 行为），含 `sourcePath`/`installPath`/`version`/`installedAt` |
| `plugins.load.paths` | 由 `openclaw plugins install --link` 写入（CLI 行为）。`install-sc-local.sh` 不再单独维护此字段，以避免与 CLI 写入路径产生冲突 |
| `agents.list[main].subagents.allowAgents` | 由 `install-sc-local.sh` 的 `add_main_worker_permission` 维护，确保 `main` 可调用 `worker` |

如需将整套 OpenClaw 状态目录迁到其他位置，设置 `OPENCLAW_STATE_DIR=/custom/path` 后重跑安装脚本即可。

## 在 OpenClaw 中如何使用

安装后，用户不需要直接调用插件目录中的 TypeScript 代码。正常使用方式是在 OpenClaw 会话中启用或触发 `subagent-coordinator` Skill，让主代理根据任务复杂度决定是否拆分和委派。

典型使用方式是在 OpenClaw 中提出适合拆分的任务，例如：

```text
请使用 subagent-coordinator 分析这个任务：批量检查项目中的测试失败原因，拆分可并行处理的子任务，并把适合的部分委派给 worker。
```

或更直接地要求使用该 Skill：

```text
使用 subagent-coordinator 帮我把这个多步骤重构任务拆分、分级并委派给合适的子代理执行。
```

Skill 会根据任务特征进行以下决策：

| 等级 | 适用任务 | 默认路由 |
|------|----------|----------|
| L1 | 简单、单步、低风险任务，例如创建目录、复制单文件。 | `worker` subagent |
| L2 | 批处理任务，例如批量重命名、批量文本替换。 | `worker` subagent |
| L3 | 处理型任务，例如日志分析、格式化、数据处理。 | `worker` subagent，带检查点或拆分策略 |
| L4 | 分析型任务，例如代码审查、依赖分析、架构分析。 | 优先使用 ACP runtime 或受监督委派 |
| L5 | 复杂任务，例如系统设计、跨模块调试、复杂方案制定。 | 通常由主代理主导，必要时使用 ACP |

插件会在 Skill 的关键事件上自动工作，例如：

- `subagent-taskr` 在任务分析和拆分时记录任务结构、评分和分解结果。
- `subagent-exec-monitor` 在委派前后执行质量门控、检查点记录和重试策略选择。
- `subagent-observability` 记录执行指标、追踪信息、token 使用、成本和限流状态。

因此，用户侧主要关注任务描述和委派目标，不需要手动调用插件 hook。

## 推荐任务描述方式

为了让 subagent-coordinator 更准确地分解任务，建议在任务中写清楚：

- 目标：希望最终得到什么结果。
- 范围：涉及哪些目录、文件、模块或数据。
- 约束：哪些文件不能改，是否只允许分析，是否需要先确认方案。
- 验证方式：需要运行哪些测试、检查哪些输出。
- 并行性：哪些部分可以独立处理，哪些部分必须按顺序执行。

示例：

```text
使用 subagent-coordinator 处理以下任务：
目标是修复 openclaw/scripts 下健康检查脚本的失败项。
范围只包括 openclaw/scripts 和 openclaw/docker。
先分析失败项并按模块拆分，能并行调查的部分委派给 worker。
修改前先汇总方案，修改后运行相关 shellcheck 或健康检查命令验证。
```

## 常见问题

### 插件安装了但没有生效

先确认 `~/.openclaw/openclaw.json` 中包含插件 entries。运行 `openclaw plugins list | grep subagent` 应能看到三个插件。也可以手动确认 `plugins.load.paths` 中包含 `~/.openclaw/workspace/plugins/<plugin-name>`。

### 主代理无法调用 worker

确认 `main` agent 配置中包含：

```json
{
  "subagents": {
    "allowAgents": ["worker"]
  }
}
```

修改配置后需要执行：

```bash
openclaw gateway restart
```

### worker 不存在

可以手动创建：

```bash
openclaw agents add worker \
  --model <MODEL_NAME> \
  --non-interactive \
  --workspace ~/.openclaw/workspace/worker
```

然后把 `worker` 加入 `main` 的 `subagents.allowAgents`。

### 本地开发时修改插件代码没有生效

默认安装是复制文件，如果希望源码修改后立即生效，安装时应使用 `--symlink`：

```bash
./install-sc-local.sh --build --symlink
```

如果使用复制方式安装，修改源码后需要重新执行安装脚本复制最新文件，或重新构建并重启 Gateway。

### `pnpm build` 报类型错误但产物已生成

`openclaw/plugin-sdk` 的源码通过相对路径引用了 OpenClaw 内部 TypeScript 源码，其中一些类型错误来自 OpenClaw 上游而非本插件代码。TypeScript 仍然会输出 JS 产物，因此脚本中 `pnpm build` 使用 `|| true` 忽略这些错误。如果需要严格检查本插件代码的类型，可单独运行：

```bash
cd plugins/subagent-taskr && pnpm typecheck
```

### 容器中报 "blocked plugin candidate: world-writable path"

容器挂载的插件目录权限是 `777`，OpenClaw 拒绝加载。安装脚本会自动修复（`chmod 755`），但如果脚本以非 root 用户运行无法修复，可手动执行：

```bash
chmod 755 ~/.openclaw/workspace/plugins/*
```

### 容器中报 "suspicious ownership"

插件目录的拥有者与运行 gateway 的用户（通常是 root）不同。安装脚本会自动 `chown -R root:root`，但如果脚本以非 root 运行无法修复，可手动执行：

```bash
chown -R root:root ~/.openclaw/workspace/plugins/subagent-*
```

### 容器中 `openclaw gateway restart` 失败

容器中通常没有 systemd user instance。可手动重启：

```bash
pkill -f openclaw-gateway
OPENCLAW_HOME=/home/node/.openclaw nohup openclaw gateway > /tmp/openclaw-gateway.log 2>&1 &
```

## 卸载

如需移除 subagent-coordinator，可按以下步骤操作：

### 1. 通过 openclaw plugins uninstall 卸载插件

```bash
openclaw plugins uninstall --force @subagent-coordinator/taskr
openclaw plugins uninstall --force @subagent-coordinator/exec-monitor
openclaw plugins uninstall --force @subagent-coordinator/observability
```

### 2. 删除插件和 Skill 目录

```bash
# 删除插件
rm -rf ~/.openclaw/workspace/plugins/subagent-taskr
rm -rf ~/.openclaw/workspace/plugins/subagent-exec-monitor
rm -rf ~/.openclaw/workspace/plugins/subagent-observability

# 删除 Skill
rm -rf ~/.openclaw/skills/subagent-coordinator
```

如果安装时使用了 `--symlink`，删除的只是符号链接，不会影响源码目录。

### 3. 清理 OpenClaw 配置

`install-sc-local.sh` 内部以 `OPENCLAW_STATE_DIR` 指向 `~/.openclaw`，因此 `openclaw plugins install --link` 与宿主机 Gateway 始终读写同一个 `~/.openclaw/openclaw.json`。

编辑 `~/.openclaw/openclaw.json`（若 `--openclaw-home` 指定了其他位置则为对应路径），执行以下清理：

- 从 `plugins.load.paths` 中移除已删除的插件路径。
- 如果不再需要 worker 子代理，从 `main` agent 的 `subagents.allowAgents` 中移除 `"worker"`。
- 如果不再需要 worker 子代理本身，删除 `~/.openclaw/agents/worker/` 目录。

### 4. 重启 Gateway

```bash
openclaw gateway restart
```

### 5. 验证移除

```bash
openclaw skills info subagent-coordinator
# 应提示 skill 不存在

openclaw plugins list | grep subagent
# 应无输出
```

---

## 相关文档

更多细节可参考 Skill 自带文档：

- `skill/references/USER_GUIDE.md`：完整用户指南。
- `skill/references/OPENCLAW_SC_PLUGIN_INSTALL.md`：安装脚本说明。
- `skill/references/ARCHITECTURE.md`：架构说明。
- `skill/references/PLUGIN_API.md`：插件事件和 API 说明。
- `skill/references/MIGRATION.md`：迁移说明。

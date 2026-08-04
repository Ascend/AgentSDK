# Orchestrator 配置参考

> 完整 workflow.md 配置说明。首次配置时先读此文件了解各字段含义。

## 必选参数（按 tracker 场景）

不同 `tracker.kind` 的必选字段不同。缺任意一项编排器都无法正常工作——配置硬必选项缺失会让 daemon 启动直接失败（`TrackerConfigError`），运行级必选项缺失会让首次 LLM 调用失败。

| tracker.kind | 配置硬必选（缺则 daemon 启动失败） | API Token 环境变量 |
|---|---|---|
| `local` | `tracker.issues_path` + `workspace.repo_clone_url` | 无 |
| `linear` | `tracker.project_slug` + `workspace.repo_clone_url` | `LINEAR_API_KEY` |
| `github` | `tracker.owner` + `tracker.repo` + `workspace.repo_clone_url` | `GITHUB_TOKEN` |
| `gitcode` | `tracker.owner` + `tracker.repo` + `workspace.repo_clone_url` | `GITCODE_TOKEN` |
| `gitee` | `tracker.owner` + `tracker.repo` + `workspace.repo_clone_url` | `GITEE_TOKEN` |

**运行级必选（所有 kind 通用，缺则首次 LLM 调用失败）**：

- `agent.provider`（默认 `anthropic`，必须确认与实际可用提供商一致）
- `agent.model`（必须显式指定真实 model id，如 `claude-sonnet-4-20250514`；默认 `None` 可能命中无效占位符）

**非必选（自动解析，无需用户设置）**：

- provider API key：由 provider config（`clawcodex login` 写入）或环境变量（如 `ANTHROPIC_API_KEY`）自动解析；本地 provider（ollama/vllm 等）根本不需要 key。仅当首次 LLM 调用失败时排查。

**要点**：

- `workspace.repo_clone_url` 对**所有 tracker 统一必选**。`local` 指 Issue 来源是本地文件，不是指目标仓库本地——agent 仍需 clone 目标仓库才能 commit/push/PR。留空则 daemon 能启动但失去整条 Issue→PR 流水线。
- `workspace.root` 有默认值 `/tmp/symphony_workspaces`，非硬必选，但推荐显式设置（默认是共享临时目录，多项目会冲突）。
- `agent.permission_mode` 无需手动设：orchestrator 加载时自动把 `dontAsk` 提升为 `bypassPermissions`（无人值守必需）。仅当想更严格时才显式覆盖。
- `agent.provider` / `agent.model` 为运行级必选，必须填；provider API key 非必选，由 `clawcodex login` 或 env var 自动解析。
- `linear` 的 `active_states` / `terminal_states` 默认值与其他 tracker 不同（见下文各段）。

## 文件结构

workflow.md 由两部分组成：

1. **YAML frontmatter**（`---` 之间）— 编排器配置
2. **Prompt 模板**（第二个 `---` 之后）— 每个 Issue 下发时用 Jinja2 渲染的 agent prompt

## 配置段速查

### tracker — Issue 跟踪器

```yaml
tracker:
  kind: gitcode              # github | gitcode | gitee | linear | local
  endpoint: https://api.gitcode.com/api/v5   # API 端点，留空用默认
  owner: chadwweng           # 仓库 owner
  repo: AgentSDK             # 仓库名
  clone_url: https://gitcode.com/chadwweng/AgentSDK.git
  api_key: $GITCODE_TOKEN    # 环境变量引用
  assignee: chadwweng         # 过滤指定用户，留空跟踪所有
  branch_prefix: clawcodex   # Issue 分支前缀
  active_states: [open]      # 活跃状态列表
  terminal_states: [closed]  # 终态列表
```

各 tracker 需要的 API 密钥环境变量：

- `github` → `GITHUB_TOKEN`
- `gitcode` → `GITCODE_TOKEN`
- `gitee` → `GITEE_TOKEN`
- `linear` → `LINEAR_API_KEY`
- `local` → 不需要

### polling — 轮询间隔

```yaml
polling:
  interval_ms: 60000         # 60 秒轮询一次 Issue 列表
```

### workspace — 工作区配置

```yaml
workspace:
  root: /tmp/symphony_workspaces   # 工作区根目录
  strategy: isolated               # isolated | shared | sequential
  base_branch: main                # 主分支
  clone_depth: 1                   # 浅克隆深度
  checkout_issue_branch: true      # 是否自动检出 issue 分支
  require_clean_start: true        # 启动前要求工作区干净
  require_clean_between_issues: true  # Issue 间要求干净
  preserve_on_terminal: true       # 完成后保留工作区
  preserve_on_failure: true        # 失败后保留工作区
  preserve_on_timeout: true        # 超时后保留工作区
```

### agent — Agent 执行参数

```yaml
agent:
  max_concurrent_agents: 1         # 全局并发上限
  max_turns: 200                   # 每个 Issue 最大 turn 数
  max_retry_attempts: 6            # 失败后最大重试次数
  max_retry_backoff_ms: 300000     # 重试退避（5 分钟）
  provider: anthropic              # 模型提供商
  model: claude-sonnet-4-20250514  # 模型名（可选）
  permission_mode: bypassPermissions  # 权限模式
  test_command: "pytest -q"        # 验证命令
  build_command: ""                # 构建命令（可选）
  lint_command: "ruff check ."     # 检查命令（可选）
  # 环境变量注入（合并到每个 Bash 子进程）
  # env:
  #   PATH: "/custom/bin:$PATH"
  #   MY_VAR: "value"
```

**permission_mode 说明**：

- `bypassPermissions`（推荐）— 无人值守模式，所有工具自动批准
- `dontAsk` — 仍触发 ApprovalPolicy 检查，可能阻塞无人值守运行

### sandbox — 运行沙箱

```yaml
sandbox:
  thread_sandbox: workspace-write
  turn_timeout_ms: 3600000        # 每 turn 超时（1 小时）
  read_timeout_ms: 5000           # 读取超时
  stall_timeout_ms: 300000        # 空闲超时（5 分钟）
  approval_policy:
    reject:
      sandbox_approval: true
      rules: true
      mcp_elicitations: true
```

### hooks — 生命周期钩子

```yaml
hooks:
  before_run: "echo 'starting $ISSUE_IDENTIFIER'"   # Issue 开始前
  after_run:  "echo 'finished $ISSUE_IDENTIFIER'"    # Issue 结束后
  pre_commit: "ruff check ."                         # commit 前
  pre_push: ""                                        # push 前
  post_sync: "git push -u origin HEAD:review/$ID"     # 同步后
  timeout_ms: 60000                                  # 钩子超时
```

### modes — 协作模式

```yaml
modes:
  default: single              # 默认模式
  # pipeline 阶段定义
  # pipeline:
  #   stages:
  #     - analyzer
  #     - implementer
  #     - tester
```

### rules — 规则提取

```yaml
rules:
  enabled: true                # 启用 PR 审查规则提取
  max_rules: 50                # 最大规则数
  auto_extract: true           # 自动提取
  scoring:
    min_confidence: 0.6        # 最低置信度
```

### review_feedback — 审查反馈

```yaml
review_feedback:
  enabled: false               # 启用 PR 评论自动处理
  mode: manual                 # manual | auto
  poll_interval_ms: 60000      # 轮询间隔
  reply_to_comments: true      # 是否自动回复评论
  max_followup_attempts_per_pr: 5  # 每个 PR 最大 follow-up 次数
```

### observability — 可观测性

```yaml
observability:
  dashboard_enabled: true      # 启用 TUI 仪表盘
  refresh_ms: 1000
  render_interval_ms: 16
```

### server — HTTP API

```yaml
server:
  host: 127.0.0.1
  port: 8765
```

## 各 tracker 默认状态

| tracker.kind | 活跃状态 | 终态 |
|-------------|---------|------|
| github/gitcode/gitee | `open` | `closed` |
| linear | `backlog`, `todo`, `in_progress`, `in_review` | `done`, `canceled`, `duplicate` |
| local | `open`, `ready` | `completed`, `closed`, `cancelled`, `failed`, `abandoned` |

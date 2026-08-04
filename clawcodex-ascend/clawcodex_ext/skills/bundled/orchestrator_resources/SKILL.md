---
name: orchestrator
description: 配置、启动、控制、观察 ClawCodex 编排器（自治 Issue→PR 流水线）。用户可以用口语化方式直接掌控编排器运行的方方面面：初始化配置、启停守护进程、跟踪 Issue 执行、控制运行中的 Issue、管理审查规则、清理工作区。当用户提到"编排器""orchestrator""自动跑 Issue""流水线""PR 自动创建"等概念时触发。
aliases: [orch]
user_invocable: true
allowed-tools: [Bash, Read, Grep, Glob]
---

# /orchestrator — 编排器管家

## 核心原则

1. **先检测，后行动**：每次调用先执行状态检测，自动判断当前所处的阶段。
2. **口语化翻译**：把用户的自然语言需求翻译成 `clawcodex-dev orchestrator` 系列命令，执行并报告结果。
3. **安全确认**：停止、重试、变基、清理、删除等破坏性操作必须经用户确认。
4. **不确定就问**：配置项缺失、Issue ID 模糊、操作有歧义时，询问用户，不要猜测。
5. **命令参考见 `references/command-reference.md`**（其中命令正确可信，可直接使用），配置参考见 `references/workflow-config-reference.md`，workflow 模板见 `references/workflow.template.md`。仅当对某子命令的具体参数/用法拿不准时，才跑 `clawcodex-dev orchestrator <command> -h` 核实——无需每次执行前都查，那样太慢。

## 虚拟环境

clawcodex 可能运行在 Python 虚拟环境中。如果用户使用了虚拟环境，在执行 `clawcodex-dev` 命令前需要先激活：

```bash
source .venv/bin/activate
# 或直接使用虚拟环境中的 cli
.venv/bin/clawcodex-dev orchestrator server status
```

**注意**：虚拟环境不是必须的。如果用户未使用虚拟环境，直接使用 `clawcodex-dev` 即可。如果用户不确定，先尝试执行 `clawcodex-dev`，命令不存在时再提示激活虚拟环境。

## 工作目录

编排器操作默认在**当前工作目录**进行。但用户可能想操作其他目录下的编排器实例：

```bash
# 方式 1：用户指定工作目录
cd /path/to/project && clawcodex-dev orchestrator server status

# 方式 2：通过 --workflow 参数指定 workflow.md 路径，自动推导工作目录
clawcodex-dev orchestrator server status --workflow /path/to/project/workflow.md

# 方式 3：通过 --workspace 参数指定工作区根目录
clawcodex-dev orchestrator issue list --workspace /path/to/workspaces
```

**处理方式**：

- 用户说"帮我看看 /path/to/project 下的编排器" → 先 `cd /path/to/project`，再执行状态检测
- 用户说"在 /tmp/myproject 里启动编排器" → 先 `cd /tmp/myproject`，再执行阶段 1/2 的逻辑
- 所有命令执行前，先 `cd <工作目录>`，再激活虚拟环境，最后执行命令
- 如果用户未指定工作目录，使用当前目录（`pwd`）
- 状态检测阶段在没有 `workflow.md` 时提示用户当前目录，并询问是否需要切换到其他目录

---

## 状态检测（入口）

每次用户调用 `/orch` 或 `/orchestrator` 时，先执行以下检测，缓存结果供本次对话使用：

```bash
ls -la ./workflow.md 2>/dev/null || echo "MISSING"
clawcodex-dev orchestrator server status 2>&1 || echo "DAEMON_STOPPED"
clawcodex-dev orchestrator issue list 2>&1 || echo "NO_ISSUES"
clawcodex-dev orchestrator workspace list 2>&1 || echo "NO_WORKSPACES"
```

根据检测结果展示简洁状态摘要，然后引导用户到对应阶段。例：

```text
编排器：● 运行中（PID 12345，已运行 2 小时）
配置：workflow.md 已存在（tracker: gitcode）

正在运行的 Issue：
  ISSUE-42 │ 运行中 │ 34 turns │ Write src/main.py

需要我做什么？"停掉 ISSUE-42" / "看看状态" / "启动" / "查看规则"
```

---

## 阶段 1：首次配置

**触发条件**：无 `workflow.md`，用户说"启动编排器"。

### 必选参数（按 tracker 场景）

不同 tracker.kind 的必选字段不同，缺任意一项编排器都无法正常工作。完整说明见 `references/workflow-config-reference.md`。

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

- provider API key：由 provider config（`clawcodex login` 写入）或环境变量（如 `ANTHROPIC_API_KEY`）自动解析；本地 provider（ollama/vllm 等）根本不需要 key。skill 不要求用户 export，仅当首次 LLM 调用失败时排查。

**说明**：

- `workspace.repo_clone_url` 对**所有 tracker 统一必选**。`local` 指 Issue 来源是本地文件，不是指目标仓库本地——agent 仍需 clone 目标仓库才能 commit/push/PR。留空则 daemon 能启动但失去整条 Issue→PR 流水线。
- `workspace.root` 有默认值 `/tmp/symphony_workspaces`，非硬必选，但推荐显式设置（默认是共享临时目录，多项目会冲突）。
- `agent.permission_mode` 无需手动设：orchestrator 加载时自动把 `dontAsk` 提升为 `bypassPermissions`（无人值守必需）。仅当想更严格时才显式覆盖。
- `agent.provider` / `agent.model` 为运行级必选，必须填；provider API key 非必选，由 `clawcodex login` 或 env var 自动解析。

### 交互流程（tracker-kind 感知决策树）

1. 询问 `tracker.kind`（`github` / `gitcode` / `gitee` / `linear` / `local`）
2. **按 kind 分支收集硬必选字段**：
   - `local` → 问 `tracker.issues_path`（默认 `.issues`）
   - `linear` → 问 `tracker.project_slug`（如 `my-team/my-project`）
   - `github` / `gitcode` / `gitee` → 问 `tracker.owner` + `tracker.repo`
3. 问 `workspace.repo_clone_url`（所有 kind 必选；repo 类可由 owner+repo 推导为 `https://<domain>/<owner>/<repo>.git`）。**此字段用纯文本提问**：直接输出"请提供目标仓库的 clone URL"并结束本轮，等用户下一条消息回复完整 URL。**勿用 AskUserQuestion/选项式收集**——URL 是自由文本，选项式无法捕获，且 headless（`clawcodex-dev -p`）下 AskUserQuestion 是 no-op。
4. 问 `workspace.root`（推荐显式设，默认 `/tmp/symphony_workspaces/<repo>`）
5. 问 `agent.provider`（默认 `anthropic`，必须确认与实际可用提供商一致）+ `agent.model`（必选，显式指定真实 model id，如 `claude-sonnet-4-20250514`）
6. 提示用户 export 对应 API Token 环境变量（按上表，tracker token 是必选）。provider API key 不必设置——由 `clawcodex login` 或 env var 自动解析
7. 其余配置用默认值，执行 `workflow init --non-interactive --kind <kind> --owner <o> --repo <r> --workspace-root <path>` 生成配置；provider/model 需随后编辑 workflow.md 的 `agent:` 段填入
8. 报告结果：列出已填字段 + 待 export 的 tracker token 环境变量清单

**用户说"我随便看看"** → 生成最小模板，告知还需编辑哪些必选字段。

**用户说"修改配置"** → 引导编辑 workflow.md 中的 YAML frontmatter，按上表核对必选项是否齐全。

---

## 阶段 2：启动编排器

**触发条件**：用户说"启动" / "开始运行" / "跑起来"。

**决策流程**：

- 已在运行？→ 报告状态，询问是否需要重启
- 无 workflow.md？→ 跳到阶段 1
- 启动前预检失败？→ 列出缺失项，引导补全后再启动

### 启动前预检

执行 `server start` 前，读 workflow.md 并按当前 `tracker.kind` 校验必选项是否就绪，避免 daemon 启动失败后才发现缺字段：

1. **配置硬必选**（按阶段 1 的矩阵）：读 workflow.md 的 YAML frontmatter，确认该 kind 对应的硬必选字段均已填充（非空、非占位符 `{{...}}`）。
2. **API Token 环境变量**：确认对应 token（如 `GITCODE_TOKEN`）已在当前 shell export 且非空。
3. **provider + model（运行级必选）**：确认 `agent.provider` 与实际可用提供商一致、`agent.model` 为真实可用 id（非空、非占位符）。provider API key 非必选，不列入预检——由 `clawcodex login` 或 env var 自动解析，仅首次 LLM 调用失败时排查。
4. **`workspace.repo_clone_url`**：确认已填（所有 kind 必选）。

任一缺失 → 列出缺失项 + 补全方式（编辑 workflow.md / export 环境变量），询问用户是否补全后继续，**不直接启动**。全部就绪 → 进入启动命令。

**启动命令**（`server start` 是长驻前台进程，**必须后台运行**否则 Bash 无法返回）：

```bash
# 后台启动，日志重定向到文件
nohup clawcodex-dev orchestrator server start --workflow ./workflow.md > /tmp/orchestrator.log 2>&1 &
# 可选 --gateway（IM 通知）或 --workflow-yaml（声明式工作流）
# 稍候用 status 确认启动成功
sleep 2 && clawcodex-dev orchestrator server status
```

**启动报告**：PID、工作区根目录、轮询间隔、并发配置、常用命令提示。

---

## 阶段 3：查看状态

**触发条件**：用户说"看看怎么样了" / "状态" / "在跑什么" / "打开仪表盘"。

**展示**：

- 守护进程状态（`server status`）
- 运行中的 Issue 列表（`issue list`）
- 已完成 / 失败 / 重试队列统计
- 规则统计（如已启用）

**Web 仪表盘**：用户说"打开仪表盘"/"可视化"时启动 Web 界面：

```bash
clawcodex-dev orchestrator dashboard                          # 默认端口 8765
clawcodex-dev orchestrator dashboard --port 8080              # 指定端口
```

**引导**：询问用户是否想了解某个 Issue 的详情。

---

## 阶段 4：跟踪 Issue

**触发条件**：用户说"看看 XXXX 怎么样了" / "跟踪 XXXX"。

**工作流**：

1. 从 `issue list` 查找目标 Issue（模糊匹配 ID 或标题）
2. 展示 Issue 详情（`issue show --id <id>`）
3. 提供选项：实时跟踪（`tail`）、完整对话（`transcript`）、代码变更（`diff`）

**常用命令**：

```bash
clawcodex-dev orchestrator issue tail --id <issue-id>              # 实时跟踪
clawcodex-dev orchestrator issue transcript --id <issue-id>        # 完整对话
clawcodex-dev orchestrator issue diff --id <issue-id>              # 代码变更
clawcodex-dev orchestrator issue workspace --id <issue-id> --ls    # 工作区文件
```

---

## 阶段 5：控制 Issue

**触发条件**：用户说"停掉" / "暂停" / "恢复" / "重试" / "注入" / "接管"。

**支持的操作**：

| 用户说 | 命令 |
|--------|------|
| "停掉 ISSUE-42" | `issue stop --id ISSUE-42` |
| "暂停 ISSUE-42" | `issue pause --id ISSUE-42 --reason "..."` |
| "恢复 ISSUE-42" | `issue resume --id ISSUE-42` |
| "重试 ISSUE-42"（重新开始） | `issue retry --id ISSUE-42 --mode reset` |
| "在 ISSUE-42 基础上追加" | `issue retry --id ISSUE-42 --mode followup` |
| "解封 ISSUE-42" | `issue retry --id ISSUE-42 --mode unblock` |
| "变基 ISSUE-42" | `issue rebase --id ISSUE-42` |
| "注入提示给 ISSUE-42" | `issue inject --id ISSUE-42 "提示内容"` |
| "接管 ISSUE-42" | `issue takeover --id ISSUE-42` |
| "回答澄清 ISSUE-42" | `issue clarify --id ISSUE-42 --answer "..."` |
| "连接 ISSUE-42" | `issue attach --id ISSUE-42` |

**安全设计**：停止、重试（reset）、变基、解封等操作**必须**向用户展示确认信息（Issue ID、进度、后果），确认后方可执行。强制重试需用户明确说"强制"。

---

## 阶段 6：停止编排器

**触发条件**：用户说"停掉" / "关机" / "不跑了"。

**决策流程**：

- 有正在运行的 Issue？→ 列出并警告，确认后先停 Issue 再停守护进程
- 无运行中的 Issue？→ 直接停止

```bash
clawcodex-dev orchestrator server stop --all
clawcodex-dev orchestrator server stop --workspace /tmp/workspaces
```

**停止报告**：本轮运行统计（已完成/失败/总 Token）+ 保留工作区列表 + 询问是否清理。

---

## 阶段 7：管理规则

**触发条件**：用户说"看看规则" / "提取规则" / "删除规则" / "规则统计"。

**前置检查**：检查 `workflow.md` 中 `rules.enabled: true`。未启用时提示用户配置。

**常用命令**：

```bash
clawcodex-dev orchestrator rules stats                                # 规则统计
clawcodex-dev orchestrator rules list                                 # 列出所有规则
clawcodex-dev orchestrator rules review --id 1                        # 查看单条详情
clawcodex-dev orchestrator rules delete --id 1                        # 删除（需确认）
clawcodex-dev orchestrator rules extract                              # 提取规则
clawcodex-dev orchestrator rules extract --dry-run                    # 预览模式
```

---

## 阶段 8：管理保留的工作区

**触发条件**：用户说"看看工作区" / "清理工作区"。

**常用命令**：

```bash
clawcodex-dev orchestrator workspace list                             # 列出
clawcodex-dev orchestrator workspace list --status completed          # 筛选
clawcodex-dev orchestrator workspace show --id <issue-id>             # 详情
clawcodex-dev orchestrator workspace cd --id <issue-id>               # 输出路径
clawcodex-dev orchestrator workspace cleanup --id <issue-id>          # 清理单个（需确认）
clawcodex-dev orchestrator workspace cleanup --all-completed          # 清理所有已完成（需确认）
clawcodex-dev orchestrator workspace verify --id <issue-id>           # 运行验证
```

---

## 阶段 9：审查与反馈

**触发条件**：用户说"处理 PR 评论" / "看看反馈" / "批准反馈"。

**前置检查**：检查 `review_feedback.enabled` 是否开启。

**常用命令**：

```bash
clawcodex-dev orchestrator issue feedback --id <issue-id> --list     # 列出待处理
clawcodex-dev orchestrator issue feedback --id <issue-id> --approve  # 批准（触发 follow-up）
clawcodex-dev orchestrator issue feedback --id <issue-id> --dismiss  # 驳回
clawcodex-dev orchestrator issue review --id <issue-id> --approve    # 审查批准
clawcodex-dev orchestrator issue review --id <issue-id> --reject --feedback "..."  # 审查驳回
```

---

## 通用约束

1. **命令前缀**：所有命令使用 `clawcodex-dev orchestrator`
2. **破坏性操作需确认**：`stop`、`retry --mode reset`、`rebase`、`rules delete`、`workspace cleanup`、`server stop`（有运行中 Issue 时）
3. **不确定就问**：Issue ID 模糊时列出匹配项；配置缺失时给出默认值选项；意图不明确时列出可能的操作
4. **退出场景**：用户说"算了"/"取消" → 礼貌结束，告知随时可用 `/orch` 回来
5. **命令参考**：完整命令列表见 `references/command-reference.md`（其中命令正确可信，可直接使用；仅对具体参数/用法拿不准时才用 `clawcodex-dev orchestrator <command> -h` 核实）
6. **配置参考**：workflow.md 各字段说明见 `references/workflow-config-reference.md`
7. **模板文件**：workflow 模板见 `references/workflow.template.md`、`references/workflow-local.template.md` 和 `references/workflow.yaml.template`；LocalTracker Issue 卡片模板见 `references/issue-card.template.md`
8. **拿不准才查 `-h`**：速查表命令可直接信任执行，无需每次都跑 `-h`（太慢）。仅当对不熟悉的子命令具体参数不确定时，才 `clawcodex-dev orchestrator <command> -h` 核实（如 `clawcodex-dev orchestrator issue retry -h`）

---
name: subagent-coordinator
description: "Enable main agent to intelligently decompose complex tasks using operator grading system and delegate to appropriate worker subagents. Features: (1) Automatic task complexity analysis, (2) 5-level operator grading (L1-L5), (3) Smart routing (subagent vs ACP), (4) Parallel execution planning, (5) Quality gates and checkpoint support."
metadata:
  version: 0.0.4
  author: Claw
  platform: openclaw
---

# Subagent Coordinator Skill

Intelligently decompose complex tasks using the **operator grading system** and delegate to appropriate worker subagents with optimized execution planning.

## New Features in v3.0

- **Task Decomposition Engine**: Automatically analyzes and breaks down complex tasks
- **5-Level Operator Grading**: L1 (Simple) → L5 (Complex) classification
- **Smart Routing**: Automatically selects subagent vs ACP runtime based on task complexity
- **🚀 ACP Runtime Support**: Complex tasks now delegate to ACP (openclaw, opencode, claude, etc.)
- **Parallel Execution Planning**: Groups independent subtasks for concurrent execution
- **Quality Gates**: Pre/post execution validation and checkpoint support

## New Features in v3.1

- **⏱️ 超时配置 (Timeout Scaling)**: 新增 `timeout` / `timeoutSeconds` 参数，支持为不同复杂度的任务设置合适的超时时间，超时设置应随任务复杂度提升而增加（L1: 30秒 → L5: 10分钟）

## Prerequisites (Must Configure First)

Before using this skill, you must configure the subagent system. If not configured, users will see an error message prompting them to set it up.

### 1. Create Worker Subagent

```bash
openclaw agents add worker --model {MODEL_NAME} --non-interactive --workspace ~/.openclaw/workspace
```

Replace `{MODEL_NAME}` with your preferred model (e.g., deepseek/deepseek-chat, anthropic/claude-3-haiku, etc.).

### 2. Configure Main Agent Subagent Permissions

In `~/.openclaw/openclaw.json`, add to main agent:

```json
{
  "agents": {
    "list": [
      {
        "id": "main",
        "subagents": {
          "allowAgents": ["worker"]
        }
      },
      {
        "id": "worker",
        "model": "{MODEL_NAME}"
      }
    ]
  }
}
```

Replace `{MODEL_NAME}` with the same model you specified in step 1.

### 3. Restart Gateway

```bash
openclaw gateway restart
```

### 4. Verify Configuration

```bash
# Check worker subagent exists
ls ~/.openclaw/agents/worker/

# Check main agent config
cat ~/.openclaw/openclaw.json | grep -A5 '"id": "main"'
```

### Error Handling

If subagent is not configured, respond with:

> "Subagent system is not configured. Please follow these steps:
> 1.Run: openclaw agents add worker --model {MODEL_NAME}
> 2.Add subagents.allowAgents: [\"worker\"] to main agent in openclaw.json
> 3.Restart: openclaw gateway restart"

---

## 🚀 ACP Runtime Support (v3.0 新增)

ACP (Agent Client Protocol) 是处理复杂研究任务的推荐运行时，特别适合需要深度推理和多步骤分析的任务。

### 支持的 ACP Agents

通过 `acpx` 命令可用：

| Agent | 用途 | 命令 |
|-------|------|------|
| **openclaw** | OpenClaw 主代理（推荐） | `acpx openclaw` |
| opencode | AI 编程助手 | `acpx opencode` |
| claude | Anthropic Claude | `acpx claude` |
| codex | OpenAI Codex | `acpx codex` |
| gemini | Google Gemini CLI | `acpx gemini` |
| kimi | 月之暗面 Kimi | `acpx kimi` |

> **注意**: 实际可用的 ACP agents 取决于 `acpx --help` 列出的内容和 `openclaw agents list` 中配置的子代理。

### 配置 ACP 子代理

#### 1. 创建 ACP 子代理

```bash
# 在 OpenClaw 中添加 researcher 子代理（使用 openclaw 作为 ACP Agent）
openclaw agents add researcher --model openclaw --runtime acp --non-interactive --workspace ~/.openclaw/workspace
```

#### 2. 更新 openclaw.json 配置

```json
{
  "agents": {
    "list": [
      {
        "id": "main",
        "subagents": {
          "allowAgents": ["worker", "researcher"]
        }
      },
      {
        "id": "worker",
        "model": "deepseek/deepseek-chat",
        "runtime": "subagent"
      },
      {
        "id": "researcher",
        "agent": "openclaw",
        "runtime": "acp"
      }
    ]
  }
}
```

#### 3. 重启 Gateway

```bash
openclaw gateway restart
```

### 验证 ACP 可用性

```bash
# 测试 acpx 是否可用
acpx --version

# 列出可用 agents
acpx --help | grep "Commands:" -A 20
```

---

## Operator Grading System (算子分级系统)

Tasks are classified into 5 levels based on complexity:

| Level | Name | Complexity | Examples | Runtime | Delegation Rule |
|-------|------|------------|----------|---------|-----------------|
| **L1** | 简单算子 | Score 1-2 | 复制单个文件、创建目录、单文件文本替换 | `subagent` | `ALWAYS_DELEGATE` |
| **L2** | 批处理算子 | Score 3-4 | 批量复制/重命名、目录同步、批量转换 | `subagent` | `DELEGATE_WITH_SPLIT` |
| **L3** | 处理算子 | Score 5-6 | CSV数据处理、日志分析、代码格式化 | `subagent` | `DELEGATE_WITH_CHECKPOINT` |
| **L4** | 分析算子 | Score 7-8 | 代码审查、架构分析、依赖分析 | **`ACP优先`** | `DELEGATE_WITH_SUPERVISION` |
| **L5** | 复杂算子 | Score 9-10 | 系统设计、复杂调试、跨模块协调 | **`ACP`** | `MAIN_AGENT_ONLY` |

### 🏆 复杂任务优先选择 ACP

> **重要**: 对于 **L4 (分析算子)** 和 **L5 (复杂算子)** 任务，**强烈建议使用 ACP 运行时**，因为：
>
> - ACP 支持深度推理和上下文理解
> - 可保持多轮对话上下文
> - 支持复杂研究任务（如文献调研、论文撰写）

### Complexity Scoring Factors

- **Step count**: More steps = higher complexity
- **File scope**: Single < Multi < Project-wide
- **Context dependency**: None < Partial < Full
- **Determinism**: High (deterministic) < Medium < Low (creative)
- **Special keywords**: Analysis, logic, batch operations

### Decomposition Triggers

Tasks are automatically decomposed when:

- Step count > 5
- File count > 10
- Complexity score > 5
- Estimated execution time > 5 minutes

### Decomposition Strategies

1. **By File**: Split multi-file operations into file batches
2. **By Step**: Break sequential tasks into ordered subtasks
3. **By Domain**: Separate by functional areas (data collection → processing → output)

---

## Quick Reference

| Scenario | Action |
|----------|--------|
| Complex task received | Analyze if it can be decomposed |
| Decomposable task | Use sessions_spawn to delegate to worker |
| Worker completes | Synthesize results and report to user |
| Direct execution needed | Execute directly if delegation overhead > benefit |

## Available Subagents

### Subagent Runtime (简单任务)

| Agent ID | Model | Purpose |
|----------|-------|---------|
| worker | deepseek/deepseek-chat | Execute simple, deterministic tasks |

### ACP Runtime (复杂任务 - 优先选择 🏆)

| Agent ID | Agent | Purpose |
|----------|-------|---------|
| researcher | openclaw | 复杂研究任务、文献调研 |
| coder | opencode | 代码开发、调试 |
| analyst | claude | 深度分析、架构设计 |

## How to Delegate

### Using sessions_spawn Tool

#### 简单任务 → subagent (worker)

```javascript
// 简单确定性任务使用 worker
sessions_spawn({
  agentId: "worker",
  task: "复制 /path/a 到 /path/b",
  mode: "run",
  timeout: 30000  // L1: 30秒超时
})
```

#### 复杂任务 → ACP (researcher/coder) 🏆 优先

```javascript
// 复杂研究任务使用 ACP (openclaw)
sessions_spawn({
  agentId: "researcher",
  runtime: "acp",
  agent: "openclaw",
  task: "调研急性肾损伤预测的最新研究进展",
  mode: "run",  // 或 "session" 保持上下文
  timeoutSeconds: 300  // L4: 5分钟超时
})

// 代码任务使用 opencode
sessions_spawn({
  agentId: "coder",
  runtime: "acp",
  agent: "opencode",
  task: "实现一个预测模型",
  mode: "run",
  timeoutSeconds: 180  // L3-L4: 3分钟超时
})
```

### Response Handling

When worker completes, it announces back with:

- Result (assistant reply or toolResult)
- Status (completed successfully / failed / timed out)
- Token stats

Synthesize these results into your response to the user.

## Decision Framework

### 🏆 复杂任务优先选择 ACP

| 任务类型 | 推荐运行时 | 说明 |
|----------|------------|------|
| 文件复制/批量操作 | `worker` (subagent) | 简单确定性任务 |
| 数据格式化/处理 | `worker` (subagent) | L3 以下任务 |
| **文献调研/论文撰写** | **`researcher` (ACP)** 🏆 | L4+ 复杂任务 |
| **代码开发/调试** | **`coder` (ACP)** 🏆 | 需要深度推理 |
| **架构设计/系统分析** | **`analyst` (ACP)** 🏆 | L5 复杂任务 |

### 运行时选择规则

```javascript
// 决策逻辑
if (complexity <= 3) {
  // L1-L3: 使用 subagent (worker)
  useRuntime("subagent", "worker")
} else {
  // L4-L5: 优先使用 ACP
  useRuntime("acp", "researcher")  // 或 coder/analyst
}
```

### Delegate to Worker (subagent) When

- Task involves file copy/move/rename
- Batch operations (create multiple files, batch rename)
- Simple data processing (formatting, filtering)
- Directory operations (create, list)
- Any independent, deterministic subtask
- Complexity score ≤ 3

### 🏆 Delegate to ACP When

- **Research tasks**: Literature review, paper writing, trend analysis
- **Complex analysis**: Architecture design, system analysis, code review
- **Multi-step reasoning**: Requires keeping context across steps
- **Creative tasks**: Generating novel solutions, writing reports
- **Long-running tasks**: >5 minutes estimated execution time
- **Complexity score ≥ 4**: L4 (分析算子) 和 L5 (复杂算子)

### Execute Directly When

- Task requires complex context understanding
- Task involves multi-step reasoning
- Delegation overhead > execution time
- Task is already optimal for main agent

## Examples

### Example 1: Simple Task (L1 - Always Delegate)

**User Request:**
> "Copy all Markdown files from {USER_HOME}/workspace/docs to {USER_HOME}/workspace/docs-backup"

**Analysis:**

- Complexity: L2 (Batch Operator)
- Steps: 2 (find files + copy)
- Files: Multiple
- Determinism: High
- **Decision**: Delegate with batch splitting

**Execution:**

```javascript
sessions_spawn({
  agentId: "worker_l1",
  task: "Copy all *.md files from {USER_HOME}/workspace/docs to {USER_HOME}/workspace/docs-backup, creating the destination directory if needed",
  mode: "run"
})
```

**Response Synthesis:**
> "I've delegated the file copy task to the worker subagent (L2-Batch). The files will be copied to the backup directory."

### 🏆 Example 2: Complex Task (L4/L5 - ACP 优先)

**User Request:**
> "调研急性肾损伤(AKI)预测的最新研究进展，撰写一份报告"

**Analysis:**

- Complexity: L4-L5 (Complex Operator)
- Steps: 5+ (搜索 → 筛选 → 分析 → 整理 → 撰写)
- Context: 需要保持多轮对话上下文
- 任务类型: 研究调研
- **Decision**: 使用 ACP 运行时 (researcher)

**Execution:**

```javascript
// 使用 ACP (openclaw) 进行研究任务
sessions_spawn({
  agentId: "researcher",
  runtime: "acp",
  agent: "openclaw",
  task: "调研急性肾损伤(AKI)预测的最新研究进展，包括：1) 2024-2026年最新论文 2) 主要方法(SOTA) 3) 数据集和基准 4) 研究趋势。撰写一份技术报告。",
  mode: "session"  // 保持上下文
})
```

**Response Synthesis:**
> "我已将研究任务委托给 researcher (ACP) 子代理。由于任务需要深度调研和多轮分析，已使用 session 模式保持对话上下文。报告将在完成后呈现。"

### Example 3: Main Agent Only (L5)

**User Request:**
> "Analyze all Python files in the project, generate a dependency graph, and create a refactoring plan"

**Analysis:**

- Complexity: L4 (Analysis Operator)
- Steps: 3 major phases
- Files: Project-wide (20+ files)
- Context: Required
- **Decision**: Decompose by domain

**Decomposition Plan:**

```text

Phase 1: Data Collection (L2)
  ├─ Scan project structure
  ├─ Identify all Python files
  └─ Extract imports and dependencies

Phase 2: Analysis (L3) [Depends: Phase 1]
  ├─ Build dependency graph
  ├─ Detect circular dependencies
  └─ Identify code smells

Phase 3: Report Generation (L2) [Depends: Phase 2]
  ├─ Generate visualization
  ├─ Create refactoring recommendations
  └─ Output summary document

```

**Execution:**

```javascript
// Phase 1: Parallel subtasks for file scanning
sessions_spawn({
  agentId: "worker_l1",
  task: "Scan /project and list all Python files with their import statements",
  mode: "run"
})

// Phase 2: Processing with checkpoint
sessions_spawn({
  agentId: "worker_l2",
  task: "Analyze dependencies from collected data, build dependency graph",
  mode: "run"
})

// Phase 3: Final output
sessions_spawn({
  agentId: "worker_l1",
  task: "Generate Markdown report with findings and recommendations",
  mode: "run"
})
```

### Example 3: Main Agent Only (L5)

**User Request:**
> "Design a new microservices architecture for our e-commerce platform"

**Analysis:**

- Complexity: L5 (Complex Operator)
- Context: Full project context required
- Reasoning: Multi-step architectural decisions
- **Decision**: Main agent only - DO NOT DELEGATE

**Execution:**
Execute directly using main agent's full context and reasoning capabilities.

**Response:**
> "Based on the project requirements and constraints, I recommend the following architecture..."

## Tool Availability

Main agent (depth 0) has access to:

- sessions_spawn (for spawning subagents)
- All standard tools (read, write, exec, etc.)

Worker subagent has access to:

- All tools EXCEPT session tools (sessions_list, sessions_history, sessions_send, sessions_spawn)

## Limitations

- Worker cannot spawn its own subagents (maxSpawnDepth: 1 by default)
- Worker sessions auto-archive after 60 minutes
- Announce delivery is best-effort
- Task decomposition is heuristic-based (not NLP-perfect)
- L5 tasks cannot be delegated (main agent only)

---

## Fallback: Direct sessions_spawn Call

When user explicitly requires subagent coordination, use this fallback to bypass model's decision-making.

### When to Use

- User explicitly says "must use worker subagent" or "use subagent"
- User explicitly says "use sessions_spawn"
- Model refuses to delegate to worker despite skill guidance
- Guaranteed subagent execution is required

### How to Call Directly

```javascript
// Direct sessions_spawn call
sessions_spawn({
  agentId: "worker",
  task: "<task description>",
  mode: "run"  // One-shot execution
})
```

### Parameters

| Parameter | Required | Description |
|-----------|----------|-------------|
| agentId | Yes | Subagent ID (e.g., "worker") |
| task | Yes | Task description passed to subagent |
| mode | Yes | "run" (one-shot) or "session" (persistent) |
| timeout | No | 超时时间（毫秒），默认 `60000` (1分钟) |
| timeoutSeconds | No | 超时时间（秒），与 `timeout` 二选一 |

#### 超时设置建议 (Timeout Scaling)

> **重要**: 超时设置应随着任务复杂度提升而增加。以下是建议值：

| 任务等级 | 复杂度 | 建议超时 | 示例任务 |
|----------|--------|----------|----------|
| **L1** | 简单算子 (Score 1-2) | `30000` (30秒) | 单文件复制、创建目录 |
| **L2** | 批处理算子 (Score 3-4) | `60000` (1分钟) | 批量复制、目录同步 |
| **L3** | 处理算子 (Score 5-6) | `120000` (2分钟) | CSV处理、日志分析、代码格式化 |
| **L4** | 分析算子 (Score 7-8) | `300000` (5分钟) | 代码审查、架构分析、依赖分析 |
| **L5** | 复杂算子 (Score 9-10) | `600000` (10分钟) | 系统设计、复杂调试、跨模块协调 |

#### 超时配置示例

```javascript
// L1 简单任务 - 30秒超时
sessions_spawn({
  agentId: "worker",
  task: "复制单个文件到目标目录",
  mode: "run",
  timeout: 30000
})

// L3 处理任务 - 2分钟超时
sessions_spawn({
  agentId: "worker",
  task: "分析并转换 CSV 数据格式",
  mode: "run",
  timeoutSeconds: 120
})

// L4/L5 复杂任务 (ACP) - 5-10分钟超时
sessions_spawn({
  agentId: "researcher",
  runtime: "acp",
  agent: "openclaw",
  task: "调研急性肾损伤预测的最新研究进展",
  mode: "session",
  timeoutSeconds: 300  // 5分钟
})
```

### Return Value

```json
{
  "status": "accepted",
  "childSessionKey": "agent:worker:subagent:uuid",
  "runId": "uuid",
  "mode": "run"
}
```

### Response Handling

When worker completes, it announces back with:

- Result (assistant reply or toolResult)
- Status (completed successfully / failed / timed out)
- Token stats

Synthesize these results into your response to the user.

### Example: Forced Subagent Usage

**User Request:**
> "Must use worker subagent to copy files to specified directory"

**Execution:**

```javascript
sessions_spawn({
  agentId: "worker",
  task: "Copy /path/to/source to /path/to/destination",
  mode: "run"
})
```

**Response:**
> "I have delegated this task to the worker subagent for execution. The task has been accepted and is being processed."

---

This skill enables efficient task distribution between main agent and worker subagent.

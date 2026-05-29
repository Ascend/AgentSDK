# Subagent Coordinator — 使用指南

> 完整安装、配置与使用手册。面向希望集成 subagent-coordinator 到 OpenClaw 工作流的开发者与高级用户。

---

## 目录

1. [系统概览](#1-系统概览)
2. [安装](#2-安装)
3. [架构详解](#3-架构详解)
4. [Skill 与插件职责矩阵](#4-skill-与插件职责矩阵)
5. [完整事件流程](#5-完整事件流程)
6. [使用示例](#6-使用示例)
7. [配置参考](#7-配置参考)
8. [故障排查](#8-故障排查)

---

## 1. 系统概览

### 是什么

**subagent-coordinator** 是一个 OpenClaw Skill，实现基于 **L1-L5 算子分级**的智能任务分解与委派系统。它在关键决策点触发标准事件，由插件提供增强功能（性能指标、可视化追踪、任务持久化、质量门控等）。

### 核心能力

| 能力 | 说明 |
|------|------|
| **L1-L5 算子分级** | 根据任务复杂度将任务分为 5 级，决定委派策略 |
| **智能路由** | L1-L3 → `worker` subagent，L4-L5 → ACP runtime |
| **事件驱动架构** | 8 个标准事件，插件可监听并提供增强功能 |
| **内置后备** | 无插件时所有功能仍有基础实现可用 |
| **4 个官方插件** | 16+ tools, 10 services, 13 hooks |

### 组件关系图

```text

┌──────────────────────────────────────────────────────────────┐
│  你 (主会话)                                                │
│    ↕ 发送任务                                               │
│  ┌────────────────────────────────────────────────────┐   │
│  │  Skill Layer: subagent-coordinator                  │   │
│  │  • L1-L5 算子分级规则                               │   │
│  │  • 路由策略 (subagent vs ACP)                      │   │
│  │  • 内置后备实现                                     │   │
│  │  • 事件触发点                                       │   │
│  └──────────────┬─────────────────────────────────────┘   │
│                  │ 事件驱动 (8 个标准事件)                  │
│                  ▼                                          │
│  ┌────────────────────────────────────────────────────┐   │
│  │  Plugin Layer (全部可选)                            │   │
│  │                                                    │   │
│  │  📊 subagent-telemetry — 性能指标收集               │   │
│  │  📈 subagent-trace       — 执行追踪与可视化         │   │
│  │  📋 subagent-taskr       — 任务分解与持久化         │   │
│  │  🔧 subagent-exec-monitor — 质量门控与检查点       │   │
│  └────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘

```

---

## 2. 安装

### 2.1 前置条件

- OpenClaw 2026.3.28+
- 已配置主 agent (model API key)
- Worker subagent 已创建（用于 L1-L3 任务）

### 2.2 安装 Skill

**方式 A：ClawHub（推荐）**

```bash
openclaw skills install subagent-coordinator

```

**方式 B：本地安装**

```bash
# 将 skill 目录复制到 workspace
cp -r /path/to/subagent-coordinator ~/.openclaw/workspace/skills/

```

### 2.3 安装插件

将插件目录放入 `~/.openclaw/extensions/` 或 skill 的 `plugins/` 子目录：

```bash
# 复制所有 4 个插件
for plugin in subagent-telemetry subagent-trace subagent-taskr subagent-exec-monitor; do
  cp -r /path/to/$plugin ~/.openclaw/extensions/
done

# 重启 gateway
openclaw gateway restart

```

### 2.4 验证安装

```bash
# 检查 skill
openclaw skills info subagent-coordinator

# 检查插件
openclaw plugins list | grep subagent

```

**预期输出**：

```bash

📦 subagent-coordinator ✓ Ready
Hook Runner: 13 registered hooks
Plugins: 4/4 loaded

```

---

## 3. 架构详解

### 3.1 L1-L5 算子分级

| 等级 | 名称 | 复杂度 | 示例 | 运行时 | 委派策略 |
|------|------|--------|------|--------|----------|
| **L1** | Simple | 1-2 | 复制单个文件、创建目录 | `subagent` | ALWAYS_DELEGATE |
| **L2** | Batch | 3-4 | 批量复制/重命名、目录同步 | `subagent` | DELEGATE_WITH_SPLIT |
| **L3** | Processing | 5-6 | CSV 处理、日志分析 | `subagent` | DELEGATE_WITH_CHECKPOINT |
| **L4** | Analysis | 7-8 | 代码审查、架构分析 | `ACP` | DELEGATE_WITH_SUPERVISION |
| **L5** | Complex | 9-10 | 系统设计、复杂调试 | `ACP` | MAIN_AGENT_ONLY |

### 3.2 复杂度评分因子

| 因子 | 权重 | 说明 |
|------|------|------|
| Step count | 高 | 步骤越多越复杂 |
| File scope | 高 | 单文件 < 多文件 < 项目范围 |
| Context dependency | 中 | 无 < 部分 < 完全 |
| Determinism | 中 | 高确定性 < 低确定性 |
| Special keywords | 低 | 含 analysis/logic/batch 等关键词 |

### 3.3 路由决策矩阵

```javascript
// 决策逻辑
if (complexity <= 3) {
  // L1-L3: 使用 worker subagent
  return { runtime: "subagent", agentId: "worker" };
} else {
  // L4-L5: 使用 ACP runtime
  return { runtime: "acp", agentId: "researcher" };
}

```

---

## 4. Skill 与插件职责矩阵

### 4.1 subagent-coordinator (Skill)

**职责**：决策引擎，负责核心分级与路由逻辑

| 功能 | 说明 |
|------|------|
| 任务分析 | 接收任务，提取描述、步骤数、文件列表 |
| 复杂度评分 | 根据 5 个因子计算 1-10 分 |
| 算子分级 | 根据复杂度映射到 L1-L5 |
| 路由决策 | 根据等级决定 subagent/ACP/main-agent |
| 质量门控（内置） | 基础参数检查（描述非空、无危险命令） |
| 任务分解（内置） | 按步骤/文件/领域拆分为子任务 |
| 事件触发 | 在关键点触发 8 个标准事件 |

### 4.2 subagent-telemetry (插件)

**职责**：性能指标与资源管理

| 功能 | 说明 |
|------|------|
| Tool Call 捕获 | 记录所有工具调用的成功/失败/耗时 |
| Token 使用统计 | 按模型/会话统计 token 消耗与成本 |
| Agent 生命周期追踪 | 记录 agent 启动/结束/成功率 |
| 速率限制 | Token Bucket 算法防止失控代理 |
| 敏感数据脱敏 | 自动检测并脱敏 API Key/Token/密码 |
| SIEM 集成 | JSONL/CSV/JSON 导出 |

**提供服务**：

- `metrics_collector` — 核心指标存储与检索
- `rate_limiter` — Token Bucket 速率限制
- `sanitiser` — 敏感信息脱敏

**监听事件**：`BEFORE_DELEGATION`, `AFTER_EXECUTION`, `TASK_ANALYZED`, `ROUTE_DECISION`

### 4.3 subagent-trace (插件)

**职责**：执行追踪与成本分析

| 功能 | 说明 |
|------|------|
| 轨迹记录 | 记录完整执行步骤、工具调用、LLM 调用 |
| 成本追踪 | 每日/每月限额 + 超限预警 |
| 趋势分析 | 7 天趋势（耗时、成本、成功率） |
| 缓存分析 | 命中率与预估节省 |
| 轨迹对比 | A/B 测试、效率评分 |
| 仪表板 | HTTP Web UI（自动刷新） |

**提供服务**：

- `trace_recorder` — 轨迹生命周期管理
- `cost_tracker` — 成本预算与预警
- `trend_analyzer` — 7 天趋势分析
- `dashboard_server` — HTTP 仪表板

**监听事件**：`AFTER_EXECUTION`, `TASK_ANALYZED`, `QUALITY_GATE`

### 4.4 subagent-taskr (插件)

**职责**：任务分解与跨会话持久化

| 功能 | 说明 |
|------|------|
| 层级任务结构 | 父子任务关系、支持嵌套 |
| 持久化存储 | 本地 JSON 文件、跨会话持续 |
| 任务笔记 | 附加 context/finding/progress/error 笔记 |
| 状态管理 | open → wip → done/skipped |
| 依赖图分析 | 循环检测、关键路径、并行组 |
| 跨代理连续性 | 任何代理可接续工作 |

**提供服务**：

- `task_store` — 任务 CRUD + 笔记 + 依赖
- `task_graph` — 依赖图构建与分析
- `persistence` — 本地 JSON 存储 + 备份

**监听事件**：`DECOMPOSITION_REQUESTED`, `CHECKPOINT_SAVE`, `CHECKPOINT_RESTORE`, `TASK_ANALYZED`

### 4.5 subagent-exec-monitor (插件)

**职责**：质量门控与检查点管理

| 功能 | 说明 |
|------|------|
| 质量门控 | 描述非空、长度限制、危险命令检测 |
| 检查点管理 | 保存/恢复任务状态与结果 |
| 重试策略 | 根据任务类型选择 aggressive/conservative/exponential |
| 复杂度评分增强 | 交叉系统操作、安全敏感、数据转换评分 |
| 任务分解策略 | 按文件/步骤/领域分解建议 |

**提供服务**：

- `checkpoint_manager` — 检查点保存/恢复/列表

**监听事件**：`QUALITY_GATE`, `BEFORE_DELEGATION`, `AFTER_EXECUTION`, `TASK_ANALYZED`, `ROUTE_DECISION`

### 4.6 职责全景图

| 功能 | Skill | telemetry | trace | taskr | exec-monitor |
|------|-------|-----------|-------|-------|-------------|
| 任务分析 | ✅ | — | — | — | — |
| 复杂度评分 | ✅ 内置 | ✅ 增强 | — | — | ✅ 增强 |
| 算子分级 | ✅ | — | — | — | — |
| 路由决策 | ✅ 内置 | ✅ 建议 | — | — | ✅ 检查 |
| 质量门控 | ✅ 基础 | — | ✅ 记录 | — | ✅ 完整 |
| 任务分解 | ✅ 基础 | — | — | ✅ 策略 | ✅ 建议 |
| 指标收集 | — | ✅ | — | — | — |
| 速率限制 | — | ✅ | — | — | — |
| 敏感脱敏 | — | ✅ | — | — | — |
| 执行轨迹 | — | — | ✅ | — | — |
| 成本追踪 | — | — | ✅ | — | — |
| 趋势分析 | — | — | ✅ | — | — |
| 仪表板 | — | — | ✅ | — | — |
| 任务持久化 | — | — | — | ✅ | — |
| 检查点 | — | — | — | ✅ | ✅ |
| 重试策略 | — | — | — | — | ✅ |

---

## 5. 完整事件流程

```text

┌─────────────────────────────────────────────────────────────────┐
│                      TASK RECEIVED                               │
│              (你发送了一个任务请求)                               │
└────────────────────────────┬────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      TASK ANALYSIS                              │
│  • 提取描述、步骤数、文件列表                                   │
│  • 计算复杂度评分 (1-10)                                        │
│  • 映射到算子等级 (L1-L5)                                       │
│  • Skill 内置实现，或由 telemetry/exec-monitor 增强             │
└────────────────────────────┬────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              │ 触发 TASK_ANALYZED 事件     │
              │ telemetry   ✓               │
              │ taskr       ✓               │
              │ exec-monitor ✓              │
              └───────────────┬───────────────┘
                              ▼
              ┌───────────────────────────────┐
              │  复杂度评分 (可能已增强)       │
              └───────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │  需要分解？                    │
              │  触发 DECOMPOSITION_REQUESTED  │
              │  taskr ✓                      │
              └───────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │  质量门控 (QUALITY_GATE)       │
              │  exec-monitor ✓                │
              │  trace ✓                      │
              └───────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │  路由决策 (ROUTE_DECISION)     │
              │  telemetry ✓ (建议)            │
              │  exec-monitor ✓ (检查)         │
              └───────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │  委派前 (BEFORE_DELEGATION)    │
              │  telemetry ✓ (速率限制)         │
              │  exec-monitor ✓ (质量检查)     │
              └───────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │       EXECUTE TASK            │
              │  sessions_spawn()               │
              │  worker (L1-L3)                │
              │  ACP runtime (L4-L5)          │
              └───────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │  执行完成 (AFTER_EXECUTION)     │
              │  telemetry ✓ (记录指标)        │
              │  trace ✓ (记录轨迹)            │
              │  taskr ✓ (更新任务状态)         │
              │  exec-monitor ✓ (检查点)       │
              └───────────────────────────────┘

```

### 8 个标准事件

| 事件 | 触发时机 | 谁监听 | 提供什么 |
|------|----------|--------|---------|
| `TASK_ANALYZED` | 任务分析完成 | telemetry, taskr, exec-monitor | 复杂度增强 |
| `DECOMPOSITION_REQUESTED` | 需要任务分解时 | taskr | 分解策略 |
| `QUALITY_GATE` | 质量门控检查 | exec-monitor, trace | 验证结果 |
| `ROUTE_DECISION` | 路由决策时 | telemetry, exec-monitor | 运行时建议 |
| `BEFORE_DELEGATION` | 任务委派前 | telemetry, exec-monitor | 速率限制/检查 |
| `AFTER_EXECUTION` | 任务执行完成 | 所有插件 | 记录结果 |
| `CHECKPOINT_SAVE` | 检查点保存时 | taskr, exec-monitor | 持久化 |
| `CHECKPOINT_RESTORE` | 检查点恢复时 | taskr, exec-monitor | 恢复决策 |

---

## 6. 使用示例

### 6.1 通过主会话委托任务

**L1 简单任务**（自动委派给 worker）：

```text

你: "Copy all Markdown files from docs/ to docs-backup/"
系统: 识别为 L2, 使用 subagent(worker) 执行

```

**L4 复杂任务**（使用 ACP runtime）：

```text

你: "Analyze the architecture of this codebase and write a report"
系统: 识别为 L4, 使用 ACP runtime + session mode 执行

```

### 6.2 通过 Skill API 直接调用

```javascript
// 任务分析
const complexity = calculateComplexity({
  description: "Read all files in /project and analyze dependencies",
  steps: 5,
  files: ["src/**/*.js"]
});
// → { total: 6, level: "L3" }

// 路由决策
const route = makeRoutingDecision(complexity, "L3");
// → { runtime: "subagent", agentId: "worker" }

```

### 6.3 使用插件工具

```javascript
// 获取 token 使用统计
const usage = await getTokenUsage({
  sessionId: "session-123",
  timeRange: { start: Date.now() - 86400000, end: Date.now() }
});
// → { totalInputTokens: 50000, totalCost: 0.15, byModel: {...} }

// 获取执行轨迹
const trace = await getTrace({
  sessionId: "session-123",
  includeSteps: true,
  includeToolCalls: true
});
// → { steps: [...], toolCalls: [...], duration: 1234ms }

// 创建任务
const task = await createTask({
  description: "Review PR #42",
  priority: "high"
});
// → { id: "task_001", status: "open", ... }

// 任务分解
const { subtasks, strategy } = await decomposeTask({
  taskId: "task_001",
  strategy: "by_file",
  maxSubtasks: 5
});
// → { subtasks: [...], parallelGroups: [...] }

```

### 6.4 会话模式（长任务）

```javascript
// L4/L5 任务使用 session mode
sessions_spawn({
  agentId: "researcher",
  runtime: "acp",
  agent: "openclaw",
  task: "Research latest AKI prediction methods and write a report",
  mode: "session"  // 保持上下文
})

```

---

## 7. 配置参考

### 7.1 插件配置

在 `openclaw.json` 的 `plugins.entries` 中：

```json
{
  "plugins": {
    "entries": {
      "subagent-telemetry": {
        "enabled": true,
        "config": {}
      },
      "subagent-trace": {
        "enabled": true,
        "config": {}
      },
      "subagent-taskr": {
        "enabled": true,
        "config": {}
      },
      "subagent-exec-monitor": {
        "enabled": true,
        "config": {}
      }
    }
  }
}

```

### 7.3 速率限制配置（telemetry）

```javascript
// rate_limiter 默认配置
{
  maxEventsPerSecond: 10,  // 最大持续速率
  burstSize: 20,           // 突发容量
  windowMs: 1000            // 填充窗口
}

```

### 7.4 成本追踪配置（trace）

```javascript
// cost_tracker 预算配置
{
  dailyLimit: 10.00,      // 每日限额 (USD)
  monthlyLimit: 100.00,    // 每月限额 (USD)
  alertThreshold: 0.8     // 预警阈值 (80%)
}

```

---

## 8. 故障排查

### 8.1 Skill 未加载

```bash
# 检查 skill 状态
openclaw skills info subagent-coordinator

# 如果 not found，重新安装
openclaw skills install subagent-coordinator

```

### 8.2 插件未加载

```bash
# 检查插件状态
openclaw plugins list | grep subagent

# 检查具体错误
openclaw plugins inspect subagent-telemetry

# 常见问题
# 1. plugin.json 缺少 configSchema
# 2. index.js id 与目录名不匹配
# 3. 权限问题 → chown -R node:node ~/.openclaw/extensions/subagent-*
```

### 8.3 Hook 未触发

```bash
# 检查 hook 注册数量
# 预期: 13 registered hooks
openclaw plugins list | grep "hook runner"

# 检查具体插件的 hook
openclaw plugins inspect subagent-exec-monitor | grep hooks

```

### 8.4 插件冲突（hook 重复注册）

```bash

ERROR: hook already registered: task_analyzed (subagent-taskr)

```

**说明**：多个插件注册了同一个 hook。这是正常的——hook runner 支持多 handler，按注册顺序依次调用。

### 8.5 性能问题

```javascript
// 使用 telemetry 检查 token 使用
const usage = await getTokenUsage({ sessionId: "your-session" });

// 使用 trace 检查执行瓶颈
const breakdown = await getCostBreakdown({
  sessionId: "your-session",
  groupBy: "tool"
});

```

---

## 附录：完整文件结构

```text

~/.openclaw/workspace/skills/subagent-coordinator/
├── SKILL.md                        # Skill 定义
├── events.ts                       # 事件契约 (8 个事件)
└── references/                     # 文档与资源
    ├── ARCHITECTURE.md             # 架构文档
    ├── PLUGIN_API.md               # 插件开发 API
    ├── MIGRATION.md                # 迁移指南
    ├── USER_GUIDE.md               # 本文档
    └── OPENCLAW_SC_PLUGIN_INSTALL.md  # 安装指南

```

---

*最后更新：2026-04-05 | OpenClaw 2026.3.28*

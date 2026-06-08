# @subagent-coordinator/taskr — 任务规划与分解

为 **Subagent Coordinator** 提供层次化任务规划、持久化状态管理、复杂度评分和跨代理连续性支持。

## 概述

**Taskr** 是 Subagent Coordinator 插件系统中的任务编排核心。它提供了结构化的任务生命周期——创建、分解、执行和追踪——使主代理能够将复杂工作流分解为带有显式依赖图的可管理子任务。

完整架构设计参见 [RFC: Agent调度插件支持大小模型调度](../../../../../openclaw/docs/rfc/26.1.0/Agent调度插件支持大小模型调度.md)。

## 功能特性

### 工具（MCP）

| 工具 | 描述 |
|------|------|
| `create_task` | 创建新任务，支持可选父任务、优先级、步骤和文件元数据 |
| `get_task` | 按 ID 检索任务（包含备注、依赖关系、子任务以及从根路径的路径） |
| `update_task` | 更新任务属性并切换状态（`open` → `wip` → `done` / `skipped`） |
| `list_tasks` | 按状态、父任务、优先级查询任务，支持排序和分页 |
| `decompose_task` | 使用基于文件、基于步骤或基于领域策略将任务分解为子任务 |
| `add_note` | 向任务添加备注（上下文、发现、进度、错误）以支持执行可追溯性 |
| `score_complexity` | 从五个启发式维度（步骤数、文件数、依赖深度、确定性、关键词）对任务复杂度评分（1–10 分） |
| `classify_operator` | 将复杂度分数映射到操作员等级（L1–L5），并推荐处理策略 |

### 钩子（事件）

| 钩子 | 触发时机 | 用途 |
|------|---------|------|
| `task_analyzed` | 复杂度分析后 | 将分析结果记录为任务备注；生成处理建议 |
| `decomposition_requested` | 请求任务分解时 | 选择最优分解策略；生成子任务列表 |
| `checkpoint_save` | 保存检查点前 | 验证数据完整性；记录进度备注 |
| `checkpoint_restore` | 恢复检查点前 | 定位检查点；验证新鲜度和完整性；计算恢复状态 |

### 服务

- **TaskStore** — 内存中的 CRUD 存储，支持可选的持久化钩子；支持层次化父子查询
- **TaskGraph** — 任务依赖关系的有向图，具备循环检测（DFS）、拓扑排序、关键路径计算和并行组识别
- **ComplexityScorer（启发式）** — 零 LLM 开销的 5 维评分器；无需额外 API 调用
- **MemoryPersistenceService** — 基于快照的持久化，支持跨会话连续性

## 架构

```text
plugins/subagent-taskr/
├── openclaw.plugin.json       # 插件元数据（@subagent-coordinator/taskr）
├── src/
│   ├── index.ts               # 插件入口：注册 8 个工具 + 4 个钩子
│   ├── hooks/
│   │   ├── checkpoint_restore.ts
│   │   ├── checkpoint_save.ts
│   │   ├── decomposition_requested.ts
│   │   └── task_analyzed.ts
│   ├── services/
│   │   ├── complexity_scorer.ts
│   │   ├── memory_store.ts
│   │   ├── memory_task_store.ts
│   │   ├── task_graph.ts
│   │   └── task_store.ts
│   └── tools/
│       ├── add_note.ts
│       ├── classify_operator.ts
│       ├── create_task.ts
│       ├── decompose_task.ts
│       ├── get_task.ts
│       ├── list_tasks.ts
│       ├── score_complexity.ts
│       └── update_task.ts
├── package.json
└── tsconfig.json
```

## 使用方式

当 **subagent-coordinator** 技能激活时，本插件将自动加载。工具可通过标准 MCP 工具调用机制进行调用。

```javascript
// 创建任务并评估复杂度
const task = await callTool("create_task", {
  description: "重构认证模块",
  steps: 12,
  files: ["auth.ts", "session.ts", "middleware.ts"],
  priority: "high"
});

// 分解为子任务
const subtasks = await callTool("decompose_task", {
  taskId: task.id,
  strategy: "by_file",   // "by_file" | "by_step" | "by_domain"
  maxSubtasks: 8
});

// 复杂度评分（无需 LLM 调用）
const score = await callTool("score_complexity", {
  task: { id: task.id, description: "重构认证模块", steps: 12, files: [...] }
});

// 分类操作员等级
const level = await callTool("classify_operator", {
  complexity: { total: 7.5, breakdown: { steps: 3, files: 2, dependency: 1.5, determinism: 1 }, keywords: ["refactor"] }
});
```

## 事件

| 消费事件 | 产生事件 |
|---------|---------|
| `subagent-coordinator:task_analyzed` | —（内部发出 `checkpoint_save` / `checkpoint_restore`） |
| `subagent-coordinator:decomposition_requested` | — |
| `subagent-coordinator:checkpoint_save` | — |
| `subagent-coordinator:checkpoint_restore` | — |

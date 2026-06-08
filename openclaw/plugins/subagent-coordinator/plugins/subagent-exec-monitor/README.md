# @subagent-coordinator/exec-monitor — 执行质量与恢复

为 **Subagent Coordinator** 提供质量门禁、检查点保存/恢复和重试策略。

## 概述

**执行监控（Execution Monitor）** 通过执行前后质量门禁检查、管理检查点以实现容错恢复，以及基于错误分类提供重试策略，来保障子代理（subagent）任务的执行质量。它与 Taskr（任务分解）和 Observability（指标/链路追踪）协同工作，形成完整的执行生命周期。

完整架构设计参见 [RFC: Agent调度插件支持大小模型调度](../../../../../openclaw/docs/rfc/26.1.0/Agent调度插件支持大小模型调度.md)。

## 功能特性

### 工具（MCP）

| 工具 | 描述 |
|------|------|
| `quality_gate_check` | 执行前验证（描述非空、步骤/文件数量在范围内）和执行后结果检查 |
| `retry_strategy_selector` | 基于错误类型（超时、限流、认证、临时、未知）和执行历史选择最优重试策略 |
| `save_checkpoint` | 持久化当前子任务执行状态（已完成、待处理、结果）以便后续恢复 |
| `restore_checkpoint` | 恢复保存的检查点，从最后一个已知状态继续执行 |

### 钩子（事件）

| 钩子 | 触发时机 | 用途 |
|------|---------|------|
| `before_delegation` | 子代理委派前 | 执行委派前质量检查并提供路由建议 |
| `after_execution` | 任务执行后 | 记录执行结果；触发检查点保存 |
| `quality_gate` | 质量门禁检查时 | 执行前验证任务合理性，执行后验证结果正确性 |

### 服务

- **CheckpointManager** — 内存中的检查点存储，支持保存、恢复和列表操作；具备新鲜度验证和完整性检查功能

## 架构

```text
plugins/subagent-exec-monitor/
├── openclaw.plugin.json            # 插件元数据（@subagent-coordinator/exec-monitor）
├── src/
│   ├── index.ts                    # 插件入口：注册 4 个工具 + 3 个钩子
│   ├── hooks/
│   │   ├── after_execution.ts
│   │   ├── before_delegation.ts
│   │   └── quality_gate.ts
│   ├── services/
│   │   └── checkpoint_manager.ts
│   └── tools/
│       └── retry_strategy.ts
├── package.json
├── README.md
└── tsconfig.json
```

## 使用方式

当 **subagent-coordinator** 技能激活时，本插件将自动加载。工具可通过标准 MCP 工具调用机制进行调用。

```javascript
// 执行前质量门禁
const preCheck = await callTool("quality_gate_check", {
  task: { id: "1", description: "重构认证模块", steps: 12 },
  preExecution: true
});

// 任务执行中保存检查点
const cp = await callTool("save_checkpoint", {
  taskId: "task-1",
  subtasks: [{ id: "a", description: "数据层" }],
  completedSubtasks: [],
  results: {}
});

// 超时后选择重试策略
const strategy = await callTool("retry_strategy_selector", {
  error: "timeout",
  history: [{ timestamp: Date.now(), error: "timeout", strategy: "linear_backoff" }]
});

// 恢复检查点以继续执行
const resume = await callTool("restore_checkpoint", {
  checkpointId: cp.checkpointId
});
```

## 事件

| 消费事件 | 产生事件 |
|---------|---------|
| `subagent-coordinator:before_delegation` | — |
| `subagent-coordinator:after_execution` | — |
| `subagent-coordinator:quality_gate` | — |

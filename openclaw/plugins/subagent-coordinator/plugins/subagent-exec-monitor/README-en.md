# @subagent-coordinator/exec-monitor — Execution Quality & Recovery

Quality gates, checkpoint save/restore, and retry strategies for **Subagent Coordinator**.

## Overview

**Execution Monitor** safeguards subagent task execution by enforcing pre- and post-execution quality gates, managing checkpoints for fault-tolerant resumption, and providing retry strategies based on error classification. It works in concert with Taskr (task decomposition) and Observability (metrics/tracing) to form a complete execution lifecycle.

For the full architecture design, see [RFC: Agent调度插件支持大小模型调度](../../../../../openclaw/docs/rfc/26.1.0/Agent调度插件支持大小模型调度.md).

## Features

### Tools (MCP)

| Tool | Description |
|------|-------------|
| `quality_gate_check` | Pre-execution validation (description not empty, step/file count within bounds) and post-execution pass check |
| `retry_strategy_selector` | Select optimal retry strategy based on error type (timeout, rate-limit, auth, transient, unknown) and execution history |
| `save_checkpoint` | Persist current subtask execution state (completed, pending, results) for later resumption |
| `restore_checkpoint` | Restore a saved checkpoint and resume execution from the last known state |

### Hooks (Events)

| Hook | Trigger | Purpose |
|------|---------|---------|
| `before_delegation` | Before subagent delegation | Performs pre-delegation quality checks and routing suggestions |
| `after_execution` | After task execution | Records execution result; triggers checkpoint save |
| `quality_gate` | At quality gate check | Validates task reasonableness pre-execution and result correctness post-execution |

### Services

- **CheckpointManager** — In-memory checkpoint store with save, restore, and listing operations; supports freshness validation and integrity checks

## Architecture

```text
plugins/subagent-exec-monitor/
├── openclaw.plugin.json            # Plugin metadata (@subagent-coordinator/exec-monitor)
├── src/
│   ├── index.ts                    # Plugin entry: registers 4 tools + 3 hooks
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

## Usage

This plugin is automatically loaded when the **subagent-coordinator** skill is active. Tools are callable via the standard MCP tool invocation mechanism.

```javascript
// Pre-execution quality gate
const preCheck = await callTool("quality_gate_check", {
  task: { id: "1", description: "Refactor auth module", steps: 12 },
  preExecution: true
});

// Save a checkpoint mid-task
const cp = await callTool("save_checkpoint", {
  taskId: "task-1",
  subtasks: [{ id: "a", description: "Data layer" }],
  completedSubtasks: [],
  results: {}
});

// Select a retry strategy after a timeout
const strategy = await callTool("retry_strategy_selector", {
  error: "timeout",
  history: [{ timestamp: Date.now(), error: "timeout", strategy: "linear_backoff" }]
});

// Restore a checkpoint to resume
const resume = await callTool("restore_checkpoint", {
  checkpointId: cp.checkpointId
});
```

## Events

| Consumed Event | Produced Event |
|----------------|----------------|
| `subagent-coordinator:before_delegation` | — |
| `subagent-coordinator:after_execution` | — |
| `subagent-coordinator:quality_gate` | — |

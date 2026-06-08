# @subagent-coordinator/taskr — Task Planning & Decomposition

Hierarchical task planning, persistent state management, complexity scoring, and cross-agent continuity plugin for **Subagent Coordinator**.

## Overview

**Taskr** is the task orchestration core of the Subagent Coordinator plugin system. It provides a structured task lifecycle — create, decompose, execute, and track — enabling the main agent to break complex workflows into manageable subtasks with explicit dependency graphs.

For the full architecture design, see [RFC: Agent调度插件支持大小模型调度](../../../../../openclaw/docs/rfc/26.1.0/Agent调度插件支持大小模型调度.md).

## Features

### Tools (MCP)

| Tool | Description |
|------|-------------|
| `create_task` | Create a new task with optional parent, priority, steps, and file metadata |
| `get_task` | Retrieve a task by ID (includes notes, dependencies, children, and path from root) |
| `update_task` | Update task properties and transition status (`open` → `wip` → `done` / `skipped`) |
| `list_tasks` | Query tasks by status, parent, priority, with sorting and pagination |
| `decompose_task` | Decompose a task into subtasks using file-based, step-based, or domain-based strategies |
| `add_note` | Add notes (context, finding, progress, error) to a task for execution traceability |
| `score_complexity` | Score task complexity (1–10) from five heuristic dimensions: steps, files, dependency depth, determinism, and keywords |
| `classify_operator` | Map a complexity score to an operator level (L1–L5) with recommended handling strategy |

### Hooks (Events)

| Hook | Trigger | Purpose |
|------|---------|---------|
| `task_analyzed` | After complexity analysis | Records analysis result as a task note; generates handling suggestions |
| `decomposition_requested` | When task decomposition is requested | Selects optimal decomposition strategy; generates subtask list |
| `checkpoint_save` | Before saving a checkpoint | Validates data integrity; records progress note |
| `checkpoint_restore` | Before restoring a checkpoint | Locates checkpoint; validates freshness and integrity; computes resume state |

### Services

- **TaskStore** — In-memory CRUD storage with optional persistence hooks; supports hierarchical parent-child queries
- **TaskGraph** — Directed graph of task dependencies with cycle detection (DFS), topological sort, critical path calculation, and parallel-group identification
- **ComplexityScorer (Heuristic)** — Zero-LLM-cost scoring across 5 dimensions; no extra API calls needed
- **MemoryPersistenceService** — Snapshot-based persistence for cross-session continuity

## Architecture

```text
plugins/subagent-taskr/
├── openclaw.plugin.json       # Plugin metadata (@subagent-coordinator/taskr)
├── src/
│   ├── index.ts               # Plugin entry: registers 8 tools + 4 hooks
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

## Usage

This plugin is automatically loaded when the **subagent-coordinator** skill is active. Tools are callable via the standard MCP tool invocation mechanism.

```javascript
// Create a task with complexity assessment
const task = await callTool("create_task", {
  description: "Refactor authentication module",
  steps: 12,
  files: ["auth.ts", "session.ts", "middleware.ts"],
  priority: "high"
});

// Decompose into subtasks
const subtasks = await callTool("decompose_task", {
  taskId: task.id,
  strategy: "by_file",   // "by_file" | "by_step" | "by_domain"
  maxSubtasks: 8
});

// Score complexity (no LLM call needed)
const score = await callTool("score_complexity", {
  task: { id: task.id, description: "Refactor auth module", steps: 12, files: [...] }
});

// Classify operator level
const level = await callTool("classify_operator", {
  complexity: { total: 7.5, breakdown: { steps: 3, files: 2, dependency: 1.5, determinism: 1 }, keywords: ["refactor"] }
});
```

## Events

| Consumed Event | Produced Event |
|----------------|----------------|
| `subagent-coordinator:task_analyzed` | — (emits `checkpoint_save` / `checkpoint_restore` internally) |
| `subagent-coordinator:decomposition_requested` | — |
| `subagent-coordinator:checkpoint_save` | — |
| `subagent-coordinator:checkpoint_restore` | — |

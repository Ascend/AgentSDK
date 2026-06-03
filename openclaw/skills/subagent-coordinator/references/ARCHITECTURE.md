# Subagent Coordinator — Architecture

**Project path**: `/Users/chad/workspace/agent-skills/subagent-coordinator/`
**Version**: 4.0.0

---

## 1. System Overview

The Subagent Coordinator is an event-driven task decomposition and delegation system built on a **Skill + 3 Plugins** architecture.

### 1.1 Component Diagram

```text

┌─────────────────────────────────────────────────────────────────────────────┐
│                         MAIN AGENT (OpenClaw)                                │
│                                                                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                      SKILL: subagent-coordinator                      │  │
│  │                                                                        │  │
│  │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │  │
│  │   │  Task Input  │─▶│   Analyzer   │─▶│  Decomposer  │               │  │
│  │   └──────────────┘  └──────┬───────┘  └──────┬───────┘               │  │
│  │                            │                  │                         │  │
│  │                            ▼                  ▼                         │  │
│  │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │  │
│  │   │   Router     │◀─│   Quality    │◀─│  Complexity  │               │  │
│  │   │              │  │    Gate      │  │    Scorer    │               │  │
│  │   └──────┬───────┘  └──────────────┘  └──────────────┘               │  │
│  │          │                                                          │  │
│  │          ▼                                                          │  │
│  │   ┌──────────────┐                                                  │  │
│  │   │  Delegator   │                                                  │  │
│  │   └──────┬───────┘                                                  │  │
│  │          │                                                          │  │
│  └──────────┼──────────────────────────────────────────────────────────┘  │
│             │ Events                                                     │
└─────────────┼─────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           PLUGIN LAYER                                      │
│                                                                              │
│  ┌────────────────────┐  ┌────────────────────┐  │
│  │    observability   │  │       taskr         │  │
│  │                     │  │                    │  │
│  │  • Metrics capture  │  │  • Task persistence│  │
│  │  • Token tracking   │  │  • Decomposition   │  │
│  │  • Rate limiting    │  │  • Task graph      │  │
│  │  • Sanitisation     │  │  • Sync            │  │
│  │  • Trace recording  │  │                     │  │
│  │  • Cost analysis    │  │                     │  │
│  │  • Trend analysis   │  │                     │  │
│  └────────────────────┘  └────────────────────┘  │
│                                                   │
│  ┌────────────────────┐                          │
│  │    exec-monitor    │                          │
│  │                     │                          │
│  │  • Quality gates   │                          │
│  │  • Checkpoints     │                          │
│  │  • Retry strategies │                         │
│  │  • Complexity score │                         │
│  └────────────────────┘                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

```

### 1.2 Component Responsibilities

| Component | Type | Responsibility |
|-----------|------|---------------|
| **subagent-coordinator** | Skill | Core orchestration: task analysis, complexity scoring, routing decisions, delegation |
| **observability** | Plugin | Metrics collection, token usage tracking, rate limiting, data sanitisation, execution trace recording, cost breakdown, trend analysis |
| **taskr** | Plugin | Hierarchical task planning, persistence, cross-session continuity, task notes |
| **exec-monitor** | Plugin | Quality gates, checkpoint management, retry strategies, task decomposition |

---

## 2. Event Flow

Events flow from the Skill to Plugins, enabling loose coupling and optional plugin participation.

### 2.1 Event Flow Diagram

```text

Task Received
      │
      ▼
┌─────────────────┐
│   TASK_ANALYZED │  ──────────────────────────────┐
└────────┬────────┘                               │
         │                                       ▼
         │                        ┌─────────────────────────┐
         │                        │  observability │
         │                        │  (enhances complexity   │
         │                        │   score with ML model)  │
         │                        └─────────────────────────┘
         │
         ▼
┌─────────────────────────┐
│ DECOMPOSITION_REQUESTED │ ────────────────────────┐
└────────────┬────────────┘                           │
             │                                        ▼
             │                        ┌─────────────────────────┐
             │                        │       taskr             │
             │                        │  (provides strategic     │
             │                        │   decomposition plan)   │
             │                        └─────────────────────────┘
             │
             ▼
┌─────────────────┐
│  ROUTE_DECISION │  ────────────────────────────────┐
└────────┬────────┘                                   │
         │                                            ▼
         │                        ┌─────────────────────────┐
         │                        │   exec-monitor          │
         │                        │  (suggests optimal      │
         │                        │   runtime selection)     │
         │                        └─────────────────────────┘
         │
         ▼
┌─────────────────┐
│  QUALITY_GATE   │  ─────── (pre-execution validation)
└────────┬────────┘
         │
         │  Pass?
         ├─ No ──▶ Return error to main agent
         │
         ▼  Yes
┌─────────────────────┐
│  BEFORE_DELEGATION  │  ────────────────────────────┐
└─────────┬───────────┘                               │
          │                                            ▼
          │                        ┌─────────────────────────┐
          │                        │  observability │
          │                        │  (records delegation    │
          │                        │   metrics)              │
          │                        └─────────────────────────┘
          │
          ▼
┌─────────────────┐
│    DELEGATE     │  ──▶ worker (L1-L3) or ACP (L4-L5)
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│   AFTER_EXECUTION   │  ───┬────────────────────────────┐
└─────────────────────┘      │                            │
                            ▼                            ▼
          ┌─────────────────────────┐
          │  observability │
          │  (records performance   │
          │   metrics & execution   │
          │   trace)                │
          └─────────────────────────┘
                        │
                        ▼
          ┌─────────────────────────┐
          │   exec-monitor         │
          │  (checkpoint save if   │
          │   long-running task)   │
          └─────────────────────────┘

```

### 2.2 Event Data Payloads

#### TASK_ANALYZED

```typescript
interface TaskAnalyzedEvent {
  task: {
    id: string;
    description: string;
    steps?: number;
    files?: string[];
    estimatedDuration?: number;
    priority?: "low" | "normal" | "high" | "urgent";
  };
  complexity: {
    total: number;        // 1-10
    breakdown: {
      steps: number;
      files: number;
      dependency: number;
      determinism: number;
    };
    keywords: string[];
  };
  operatorLevel: "L1" | "L2" | "L3" | "L4" | "L5";
  decompositionTriggered: boolean;
  timestamp: number;
}

```

#### DECOMPOSITION_REQUESTED

```typescript
interface DecompositionRequestedEvent {
  task: Task;
  complexity: ComplexityScore;
  suggestedStrategy?: "by_file" | "by_step" | "by_domain";
  timestamp: number;
}

```

#### BEFORE_DELEGATION

```typescript
interface BeforeDelegationEvent {
  task: Task;
  complexity: ComplexityScore;
  operatorLevel: OperatorLevel;
  routingDecision: {
    runtime: "subagent" | "acp";
    agentId: string;
    reason: string;
  };
  timestamp: number;
}

```

#### AFTER_EXECUTION

```typescript
interface AfterExecutionEvent {
  task: Task;
  result: {
    taskId: string;
    success: boolean;
    output?: unknown;
    error?: string;
    duration: number;      // milliseconds
    tokensUsed?: number;
  };
  complexity: ComplexityScore;
  operatorLevel: OperatorLevel;
  timestamp: number;
}

```

---

## 3. Plugin Responsibilities Matrix

| Event | observability | taskr | exec-monitor |
|-------|------------------------|-------|--------------|
| `task_analyzed` | Enhance complexity score, record analysis trace | — | — |
| `decomposition_requested` | — | Provide decomposition strategy | — |
| `route_decision` | Log routing metrics | — | Suggest runtime |
| `quality_gate` | — | — | **Primary handler** |
| `before_delegation` | Record delegation metrics & trace | — | — |
| `after_execution` | Record performance metrics & execution trace | — | Manage checkpoints |
| `checkpoint_save` | — | Persist task state | Save checkpoint |
| `checkpoint_restore` | — | Restore task state | Restore checkpoint |

---

## 4. Decision Flow

### 4.1 Task Processing Pipeline

```text

Task Description
      │
      ▼
┌─────────────────┐     No      ┌──────────────────────┐
│  Has steps > 5  │───────────▶│  Continue pipeline    │
│  or files > 10  │            └──────────────────────┘
│  or score > 5   │
└────────┬────────┘
         │ Yes
         ▼
┌─────────────────┐
│  DECOMPOSE TASK │  ──▶ Split into subtasks
└────────┬────────┘      with dependency graph
         │
         ▼
┌─────────────────┐
│  CALCULATE      │
│  COMPLEXITY     │  ──▶ Score 1-10 based on:
│  SCORE          │      - Step count (weight: high)
└────────┬────────┘      - File scope (weight: high)
         │               - Context dependency (weight: medium)
         ▼               - Determinism (weight: medium)
┌─────────────────┐
│  CLASSIFY       │
│  OPERATOR LEVEL │  ──▶ L1 (1-2), L2 (3-4), L3 (5-6),
└────────┬────────┘      L4 (7-8), L5 (9-10)
         │
         ▼
┌─────────────────┐
│  RUN QUALITY    │  ──▶ Validate task parameters
│  GATE           │      and prerequisites
└────────┬────────┘
         │ Fail?
         ├─ No ──▶ Continue
         │
         ▼
┌─────────────────┐
│  MAKE ROUTING   │
│  DECISION       │  ──▶ subagent (L1-L3) or ACP (L4-L5)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  BEFORE         │
│  DELEGATION     │  ──▶ Final hook before execution
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  DELEGATE TO    │
│  TARGET AGENT   │  ──▶ worker (subagent) or
└────────┬────────┘      researcher/coder (ACP)
         │
         ▼
┌─────────────────┐
│  CAPTURE RESULT │
│  + METRICS      │
└─────────────────┘

```

### 4.2 Routing Decision Matrix

| Operator Level | Complexity Score | Task Type | Runtime | Agent | Delegation Mode |
|---------------|-----------------|-----------|---------|-------|----------------|
| **L1** | 1-2 | Simple file ops | `subagent` | `worker` | `ALWAYS_DELEGATE` |
| **L2** | 3-4 | Batch operations | `subagent` | `worker` | `DELEGATE_WITH_SPLIT` |
| **L3** | 5-6 | Data processing | `subagent` | `worker` | `DELEGATE_WITH_CHECKPOINT` |
| **L4** | 7-8 | Analysis, code review | `acp` | `researcher` | `DELEGATE_WITH_SUPERVISION` |
| **L5** | 9-10 | System design, complex debugging | `acp` | `analyst` | `MAIN_AGENT_ONLY` |

---

## 5. Fallback Mechanism

When plugins are unavailable, the Skill uses built-in fallback implementations.

### 5.1 Fallback Chain

```text

Plugin Available?
      │
      ├── Yes ──▶ Use Plugin Implementation
      │
      └── No ──▶ Use Built-in Fallback
                    │
                    ├── Complexity Scoring ──▶ Heuristic scoring (keywords, step count)
                    ├── Operator Classification ──▶ Threshold-based (L1-L5)
                    ├── Task Decomposition ──▶ Split by "then"/steps
                    ├── Routing Decision ──▶ Static rules (L1-L3 vs L4-L5)
                    ├── Quality Gate ──▶ Basic parameter validation
                    └── Checkpoint ──▶ In-memory storage

```

### 5.2 Fallback vs Plugin Comparison

| Function | Built-in Fallback | Plugin Enhancement |
|----------|-------------------|-------------------|
| **Complexity Scoring** | Keyword-based heuristic | ML model or refined heuristics |
| **Task Decomposition** | Split by "then"/step count | Strategic planning (by_file/by_domain) |
| **Routing Decision** | Static threshold rules | Dynamic runtime analysis |
| **Quality Gate** | Parameter validation only | Deep validation, security checks |
| **Checkpoint** | In-memory (lost on restart) | Persistent storage (SQLite/JSON) |
| **Metrics** | None | Full telemetry capture |

### 5.3 Plugin Detection

```typescript
async function detectPlugins() {
  return {
    hasObservability: await checkPluginAvailable("subagent-observability"),
    hasTaskr: await checkPluginAvailable("openclaw-taskr"),
    hasExecMonitor: await checkPluginAvailable("subagent-exec-monitor"),
  };
}

```

---

## 6. State Management

### 6.1 State Scope

| State Type | Scope | Persistence | Managed By |
|------------|-------|-------------|------------|
| **Task State** | Per-task | Optional | taskr or in-memory |
| **Checkpoint Data** | Per-session | Optional | exec-monitor or memory |
| **Metrics & Trace Data** | Global + Per-task | Optional | observability |

### 6.2 State Flow

```text

Main Agent Session
      │
      ▼
┌─────────────────┐
│  Task Created   │  ──▶ Task State (taskr or memory)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Decomposition  │  ──▶ Task Graph (taskr)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Delegation     │  ──▶ Delegation Metrics (observability)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Execution      │  ──▶ Execution Trace (observability)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Checkpoint     │  ──▶ Checkpoint Data (exec-monitor)
└─────────────────┘

```

### 6.3 Checkpoint Data Structure

```typescript
interface CheckpointData {
  taskId: string;
  subtasks: {
    id: string;
    description: string;
    dependsOn?: string[];
    estimatedDuration?: number;
    parallelGroup?: string;
  }[];
  completedSubtasks: string[];
  results: Map<string, {
    taskId: string;
    success: boolean;
    output?: unknown;
    error?: string;
    duration: number;
  }>;
  timestamp: number;
}

```

---

## 7. Error Handling Strategy

### 7.1 Error Categories

| Category | Code | Recovery Action |
|----------|------|----------------|
| **Validation Error** | `VALIDATION_*` | Block execution, return error to main agent |
| **Plugin Error** | `PLUGIN_*` | Fallback to built-in, log warning |
| **Delegation Error** | `DELEGATION_*` | Retry with exponential backoff |
| **Checkpoint Error** | `CHECKPOINT_*` | Continue without checkpointing |
| **Timeout Error** | `TIMEOUT_*` | Trigger quality gate, assess failure |

### 7.2 Error Handling Flow

```text

Error Occurred
      │
      ▼
┌─────────────────┐
│  Categorize     │
│  Error          │
└────────┬────────┘
         │
         ├─ Validation ──▶ Block & Return Error
         │
         ├─ Plugin ──▶ Log Warning & Use Fallback
         │
         ├─ Delegation ──▶ Retry (max 3 attempts)
         │                    │
         │                    ├── Success ──▶ Continue
         │                    └── Failure ──▶ Quality Gate
         │
         └─ Timeout ──▶ Quality Gate Assessment
                          │
                          ├── Recoverable ──▶ Resume from checkpoint
                          └── Unrecoverable ──▶ Return Error

```

### 7.3 Error Codes

```typescript
const ErrorCodes = {
  // Validation
  VALIDATION_EMPTY_TASK: "VALIDATION_EMPTY_TASK",
  VALIDATION_INVALID_PARAMS: "VALIDATION_INVALID_PARAMS",

  // Plugin
  PLUGIN_NOT_AVAILABLE: "PLUGIN_NOT_AVAILABLE",
  PLUGIN_HOOK_FAILED: "PLUGIN_HOOK_FAILED",
  PLUGIN_SERVICE_FAILED: "PLUGIN_SERVICE_FAILED",

  // Delegation
  DELEGATION_TIMEOUT: "DELEGATION_TIMEOUT",
  DELEGATION_REJECTED: "DELEGATION_REJECTED",
  DELEGATION_MAX_RETRIES: "DELEGATION_MAX_RETRIES",

  // Checkpoint
  CHECKPOINT_SAVE_FAILED: "CHECKPOINT_SAVE_FAILED",
  CHECKPOINT_RESTORE_FAILED: "CHECKPOINT_RESTORE_FAILED",

  // Quality
  QUALITY_GATE_FAILED: "QUALITY_GATE_FAILED",
  QUALITY_GATE_BLOCKED: "QUALITY_GATE_BLOCKED",
};

```

---

## 8. Configuration

### 8.1 Skill Configuration

```yaml
operators:
  levels:
    L1:
      name: "Simple"
      score_range: [1, 2]
      runtime: "subagent"
      delegation: "ALWAYS_DELEGATE"
    L2:
      name: "Batch"
      score_range: [3, 4]
      runtime: "subagent"
      delegation: "DELEGATE_WITH_SPLIT"
    L3:
      name: "Processing"
      score_range: [5, 6]
      runtime: "subagent"
      delegation: "DELEGATE_WITH_CHECKPOINT"
    L4:
      name: "Analysis"
      score_range: [7, 8]
      runtime: "acp"
      delegation: "DELEGATE_WITH_SUPERVISION"
    L5:
      name: "Complex"
      score_range: [9, 10]
      runtime: "acp"
      delegation: "MAIN_AGENT_ONLY"

decomposition:
  triggers:
    step_count: 5
    file_count: 10
    complexity_score: 5
    estimated_time: 300  # seconds

routing:
  subagent:
    default_agent: "worker"
    max_retries: 3
  acp:
    default_agents:
      research: "researcher"
      coding: "coder"
      analysis: "analyst"

```

---

## 9. File Structure

```text

subagent-coordinator/
├── skill/
│   ├── SKILL.md                  # Main skill documentation
│   ├── events.ts                 # Event contract (types + constants)
│   └── references/
│       ├── ARCHITECTURE.md       # This document
│       ├── PLUGIN_API.md         # Plugin developer reference
│       ├── MIGRATION.md          # Migration guide (Phase 3 → Phase 4+)
│       ├── USER_GUIDE.md         # User guide
│       └── OPENCLAW_SC_PLUGIN_INSTALL.md  # Install guide
├── install-sc-local.sh           # Local installation script
│
└── plugins/
    ├── exec-monitor/             # Quality gates, checkpoints
    │   ├── plugin.json
    │   ├── src/
    │   │   ├── index.ts
    │   │   ├── hooks/
    │   │   ├── tools/
    │   │   └── services/
    │   └── README.md
    │
    ├── observability/  # Observability (metrics, tracing, cost tracking, rate limiting, data sanitisation)
    │   ├── plugin.json
    │   ├── src/
    │   └── README.md
    │
    └── taskr/                   # Task persistence
        ├── plugin.json
        ├── src/
        └── README.md

```

---

## 10. Dependencies

### 10.1 Between Components

```text

Main Agent
    │
    ├──▶ Skill (subagent-coordinator)
    │        │
    │        ├──▶ Events (events.ts)
    │        │
    │        ├──▶ Plugin: observability (optional)
    │        │        └──▶ Metrics Collector Service
    │        │        └──▶ Rate Limiter Service
    │        │        └──▶ Sanitiser Service
    │        │        └──▶ Trace Recorder Service
    │        │        └──▶ Cost Tracker Service
    │        │
    │        ├──▶ Plugin: taskr (optional)
    │        │        └──▶ Task Store Service
    │        │        └──▶ Task Graph Service
    │        │
    │        └──▶ Plugin: exec-monitor (optional)
    │                 └──▶ Checkpoint Manager Service
    │
    ├──▶ Worker (subagent) ────▶ L1-L3 Task Execution
    │
    └──▶ ACP Runtime ──────────▶ L4-L5 Task Execution

```

---

*Document version: 4.0.0 — Part of subagent-coordinator skill documentation.*

# Subagent Coordinator — Plugin API

**Project path**: `/Users/chad/workspace/agent-skills/subagent-coordinator/`
**Version**: 4.0.0

Plugin API reference for developers building plugins compatible with the subagent-coordinator event system.

---

## 1. Plugin Structure

A plugin is a directory with a standard structure:

```text

plugins/
└── {plugin-name}/
    ├── plugin.json          # Plugin metadata (required)
    ├── README.md            # Documentation (required for public plugins)
    └── src/
        └── index.ts        # Plugin entry point (required)

```

### 1.1 Required Files

| File | Purpose | Format |
|------|---------|--------|
| `plugin.json` | Plugin metadata, events, tools, services declaration | JSON |
| `src/index.ts` | Plugin entry point with `register()` callback | TypeScript/JavaScript |

### 1.2 plugin.json Schema

```json
{
  "id": "unique-plugin-id",
  "name": "Plugin Display Name",
  "version": "0.0.4",
  "runtime": "openclaw",
  "entry": "src/index.ts",
  "description": "What this plugin does",
  "events": [
    "subagent-coordinator:before_delegation",
    "subagent-coordinator:after_execution",
    "subagent-coordinator:task_analyzed"
  ],
  "tools": [
    "tool_id_1",
    "tool_id_2"
  ],
  "services": [
    "service_id_1"
  ]
}

```

**Field Requirements:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | Yes | Unique identifier (kebab-case) |
| `name` | string | Yes | Human-readable name |
| `version` | string | Yes | Semver (e.g., "1.0.0") |
| `runtime` | string | Yes | Must be `"openclaw"` |
| `entry` | string | Yes | Path to entry file |
| `description` | string | Yes | Brief description |
| `events` | string[] | No | Events this plugin listens to |
| `tools` | string[] | No | Tool IDs this plugin provides |
| `services` | string[] | No | Service IDs this plugin provides |

---

## 2. Plugin Entry Pattern

### 2.1 Basic Structure

```typescript
import { SUBAGENT_COORDINATOR_EVENTS } from "../../events";
import type {
  BeforeDelegationEvent,
  AfterExecutionEvent,
} from "../../events";

export default definePluginEntry({
  id: "my-plugin",
  name: "My Plugin",
  version: "1.0.0",
  description: "Description of what my plugin does",

  register(api) {
    // Register hooks
    // Register tools
    // Register services
  }
});

```

### 2.2 Plugin Entry Interface

```typescript
interface PluginEntry {
  id: string;
  name: string;
  version: string;
  description: string;
  register(api: PluginAPI): void | Promise<void>;
}

interface PluginAPI {
  registerHook<EventName extends SubagentCoordinatorEventName>(
    eventName: EventName,
    handler: HookHandler<EventName>,
    opts?: HookOptions
  ): void;

  registerTool<TInput, TOutput>(
    toolSpec: ToolSpec<TInput, TOutput>,
    opts?: ToolOptions
  ): void;

  registerService(
    serviceSpec: ServiceSpec
  ): void;

  getService(serviceId: string): ServiceInstance | null;
}

```

---

## 3. API Reference

### 3.1 registerHook()

Register a handler for a specific event.

```typescript
api.registerHook<EventName extends SubagentCoordinatorEventName>(
  eventName: EventName,
  handler: (payload: EventPayload) => Promise<HookResult>,
  opts?: HookOptions
): void

```

**Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `eventName` | `SubagentCoordinatorEventName` | Yes | Event constant from `SUBAGENT_COORDINATOR_EVENTS` |
| `handler` | function | Yes | Async function handling the event |
| `opts.priority` | number | No | Handler priority (higher = earlier, default: 0) |
| `opts.blockable` | boolean | No | Whether handler can block event propagation |

**Hook Handler Signature:**

```typescript
type HookHandler<E extends SubagentCoordinatorEventName> = (
  payload: GetEventPayload<E>
) => Promise<HookResult<E>>;

```

**Return Types by Event:**

| Event | Return Type |
|-------|-------------|
| `before_delegation` | `{ block: boolean; checks: Check[]; enhancedRoutingSuggestion?: RoutingSuggestion }` |
| `after_execution` | `{ recorded: boolean; metrics?: Metrics }` |
| `task_analyzed` | `{ enhanced: boolean; enhancedScore?: ComplexityScore }` |
| `decomposition_requested` | `{ strategy: DecompositionStrategy; estimatedSubtasks: number }` |
| `route_decision` | `{ runtime: RuntimeType; agentId: string; reason: string }` |
| `quality_gate` | `{ pass: boolean; checks: Check[] }` |
| `checkpoint_save` | `{ saved: boolean; checkpointId?: string }` |
| `checkpoint_restore` | `{ restored: boolean; checkpoint?: CheckpointData }` |

**Example:**

```typescript
api.registerHook(
  SUBAGENT_COORDINATOR_EVENTS.BEFORE_DELEGATION,
  async (event: BeforeDelegationEvent) => {
    console.log(`Task ${event.task.id} about to be delegated`);

    return {
      block: false,           // Don't block delegation
      checks: [],            // No additional checks
      enhancedRoutingSuggestion: null  // No routing change
    };
  }
);

```

### 3.2 registerTool()

Register a tool that can be called by the main agent or other plugins.

```typescript
api.registerTool<TInput, TOutput>(
  toolSpec: ToolSpec<TInput, TOutput>,
  opts?: ToolOptions
): void

```

**ToolSpec Interface:**

```typescript
interface ToolSpec<TInput, TOutput> {
  id: string;
  name: string;
  description: string;
  inputSchema: JSONSchema | "any";
  outputSchema?: JSONSchema;
  handler: (input: TInput, context: ToolContext) => Promise<TOutput>;
}

```

**ToolContext:**

```typescript
interface ToolContext {
  sessionId: string;
  agentId: string;
  taskId?: string;
  metadata?: Record<string, unknown>;
}

```

**Example:**

```typescript
api.registerTool({
  id: "get_metrics",
  name: "Get Metrics",
  description: "Retrieve performance metrics for a time range",
  inputSchema: {
    type: "object",
    properties: {
      timeRange: {
        type: "object",
        properties: {
          start: { type: "number" },
          end: { type: "number" }
        }
      }
    }
  },
  handler: async (input: { timeRange?: TimeRange }) => {
    const metrics = metricsCollector.getMetrics(input.timeRange);
    return { metrics };
  }
});

```

### 3.3 registerService()

Register a long-running service with state and methods.

```typescript
api.registerService(serviceSpec: ServiceSpec): void

```

**ServiceSpec Interface:**

```typescript
interface ServiceSpec {
  id: string;
  name: string;
  description?: string;
  state?: Record<string, unknown>;
  start?: () => void | Promise<void>;
  stop?: () => void | Promise<void>;
  methods?: Record<string, (...args: unknown[]) => unknown>;
}

```

**Service Lifecycle:**

1. `state` is initialized
2. `start()` is called when plugin loads
3. `methods` are exposed via `api.getService()`
4. `stop()` is called when plugin unloads

**Example:**

```typescript
api.registerService({
  id: "metrics_collector",
  name: "Metrics Collector",
  description: "Collects and aggregates performance metrics",

  state: {
    toolCalls: [],
    tokenUsage: [],
    startTime: Date.now(),
  },

  start: async () => {
    console.log("Metrics collector started");
  },

  stop: async () => {
    console.log("Metrics collector stopping");
    // Flush pending data
  },

  methods: {
    recordToolCall(call) {
      this.state.toolCalls.push(call);
    },

    getMetrics(timeRange) {
      return this.state.toolCalls.filter(
        c => !timeRange || (c.timestamp >= timeRange.start && c.timestamp <= timeRange.end)
      );
    },
  }
});

```

### 3.4 getService()

Retrieve a registered service instance.

```typescript
api.getService(serviceId: string): ServiceInstance | null

```

**Usage:**

```typescript
const service = api.getService("metrics_collector");
if (service) {
  const result = await service.methods.recordToolCall({ ... });
}

```

---

## 4. Event Constants

All events are prefixed with `subagent-coordinator:` and imported from `events.ts`.

```typescript
export const SUBAGENT_COORDINATOR_EVENTS = {
  /** Before delegating task — Plugin can provide routing suggestions */
  BEFORE_DELEGATION: "subagent-coordinator:before_delegation",

  /** After task execution completes — Plugin can record metrics */
  AFTER_EXECUTION: "subagent-coordinator:after_execution",

  /** After task analysis — Plugin can enhance complexity score */
  TASK_ANALYZED: "subagent-coordinator:task_analyzed",

  /** When decomposition is needed — Plugin can provide strategy */
  DECOMPOSITION_REQUESTED: "subagent-coordinator:decomposition_requested",

  /** When making routing decision — Plugin can suggest runtime */
  ROUTE_DECISION: "subagent-coordinator:route_decision",

  /** During quality gate check — Plugin can add validation */
  QUALITY_GATE: "subagent-coordinator:quality_gate",

  /** When saving checkpoint — Plugin can persist data */
  CHECKPOINT_SAVE: "subagent-coordinator:checkpoint_save",

  /** When restoring checkpoint — Plugin can provide data */
  CHECKPOINT_RESTORE: "subagent-coordinator:checkpoint_restore",
} as const;

```

---

## 5. Event Payloads

### 5.1 BEFORE_DELEGATION

```typescript
interface BeforeDelegationEvent {
  task: {
    id: string;
    description: string;
    steps?: number;
    files?: string[];
    estimatedDuration?: number;
    priority?: "low" | "normal" | "high" | "urgent";
  };
  complexity: {
    total: number;
    breakdown: { steps: number; files: number; dependency: number; determinism: number };
    keywords: string[];
  };
  operatorLevel: "L1" | "L2" | "L3" | "L4" | "L5";
  routingDecision: {
    runtime: "subagent" | "acp";
    agentId: string;
    reason: string;
  };
  timestamp: number;
}

// Return type
interface BeforeDelegationResult {
  block: boolean;
  checks: { name: string; pass: boolean; message?: string }[];
  enhancedRoutingSuggestion?: {
    runtime: "subagent" | "acp";
    agentId: string;
    reason: string;
  };
}

```

### 5.2 AFTER_EXECUTION

```typescript
interface AfterExecutionEvent {
  task: Task;
  result: {
    taskId: string;
    success: boolean;
    output?: unknown;
    error?: string;
    duration: number;
    tokensUsed?: number;
  };
  complexity: ComplexityScore;
  operatorLevel: OperatorLevel;
  timestamp: number;
}

// Return type
interface AfterExecutionResult {
  recorded: boolean;
  metrics?: {
    durationMs: number;
    tokensUsed?: number;
    success: boolean;
  };
}

```

### 5.3 TASK_ANALYZED

```typescript
interface TaskAnalyzedEvent {
  task: Task;
  complexity: ComplexityScore;
  operatorLevel: OperatorLevel;
  decompositionTriggered: boolean;
  timestamp: number;
}

// Return type
interface TaskAnalyzedResult {
  enhanced: boolean;
  originalScore?: ComplexityScore;
  enhancedScore?: ComplexityScore;
}

```

### 5.4 DECOMPOSITION_REQUESTED

```typescript
interface DecompositionRequestedEvent {
  task: Task;
  complexity: ComplexityScore;
  suggestedStrategy?: "by_file" | "by_step" | "by_domain";
  timestamp: number;
}

// Return type
interface DecompositionRequestedResult {
  strategy: "by_file" | "by_step" | "by_domain";
  estimatedSubtasks: number;
}

```

### 5.5 ROUTE_DECISION

```typescript
interface RouteDecisionEvent {
  task: Task;
  complexity: ComplexityScore;
  proposedRuntime: "subagent" | "acp";
  proposedAgentId: string;
  timestamp: number;
}

// Return type
interface RouteDecisionResult {
  runtime: "subagent" | "acp";
  agentId: string;
  reason: string;
}

```

### 5.6 QUALITY_GATE

```typescript
interface QualityGateEvent {
  task: Task;
  preExecution: boolean;
  result: {
    pass: boolean;
    checks: { name: string; pass: boolean; message?: string }[];
  };
  timestamp: number;
}

// Return type
interface QualityGateResult {
  pass: boolean;
  checks: { name: string; pass: boolean; message?: string }[];
}

```

### 5.7 CHECKPOINT_SAVE

```typescript
interface CheckpointSaveEvent {
  checkpoint: {
    taskId: string;
    subtasks: Subtask[];
    completedSubtasks: string[];
    results: Record<string, ExecutionResult>;
    timestamp: number;
  };
  timestamp: number;
}

// Return type
interface CheckpointSaveResult {
  saved: boolean;
  checkpointId?: string;
}

```

### 5.8 CHECKPOINT_RESTORE

```typescript
interface CheckpointRestoreEvent {
  checkpointId: string;
  taskId: string;
  timestamp: number;
}

// Return type
interface CheckpointRestoreResult {
  restored: boolean;
  checkpoint?: CheckpointData;
}

```

---

## 6. Service Pattern

Services provide shared state and methods accessible to hooks and tools within a plugin.

### 6.1 Service Definition

```typescript
// In your plugin's src/services/my_service.ts

export interface MyServiceState {
  counter: number;
  records: Record<string, unknown>[];
}

export function createMyServiceState(): MyServiceState {
  return {
    counter: 0,
    records: [],
  };
}

export function createMyService(state: MyServiceState) {
  return {
    increment() {
      state.counter++;
      return state.counter;
    },

    addRecord(record: Record<string, unknown>) {
      state.records.push(record);
      return state.records.length;
    },

    getRecords() {
      return state.records;
    },
  };
}

```

### 6.2 Service Registration

```typescript
// In your plugin's src/index.ts

import { createMyServiceState, createMyService } from "./services/my_service";

export default definePluginEntry({
  id: "my-plugin",
  name: "My Plugin",

  register(api) {
    // Initialize state
    const state = createMyServiceState();

    // Create service
    const service = createMyService(state);

    // Register service
    api.registerService({
      id: "my_service",
      name: "My Service",
      state,  // State will be serialized if persistence is enabled
      methods: service,  // Methods exposed via api.getService()
    });
  }
});

```

### 6.3 Service Access

```typescript
// In a hook within the same plugin

api.registerHook(
  SUBAGENT_COORDINATOR_EVENTS.AFTER_EXECUTION,
  async (event) => {
    const service = api.getService("my_service");
    if (service) {
      service.methods.addRecord({
        taskId: event.task.id,
        timestamp: event.timestamp,
      });
    }

    return { recorded: true };
  }
);

```

---

## 7. Hook Handler Signature

All hook handlers follow a consistent signature:

```typescript
async function handler(
  payload: GetEventPayload<EventName>
): Promise<HookResult<EventName>>

```

### 7.1 Handler Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `payload` | Event-specific | Event data payload |

### 7.2 Handler Return Value

All handlers must return a result object. The structure depends on the event type (see Section 5).

### 7.3 Error Handling

```typescript
api.registerHook(
  SUBAGENT_COORDINATOR_EVENTS.AFTER_EXECUTION,
  async (event) => {
    try {
      await saveMetrics(event);
      return { recorded: true };
    } catch (error) {
      console.error("Failed to record metrics:", error);
      return { recorded: false };  // Non-blocking failure
    }
  }
);

```

### 7.4 Async Operations

Handlers can perform async operations:

```typescript
api.registerHook(
  SUBAGENT_COORDINATOR_EVENTS.BEFORE_DELEGATION,
  async (event) => {
    // Await database queries
    const routingSuggestion = await analyzeHistoricalData(event);

    return {
      block: false,
      checks: [],
      enhancedRoutingSuggestion: routingSuggestion,
    };
  }
);

```

---

## 8. Complete Example Plugin

### 8.1 File Structure

```text

plugins/
└── my-example-plugin/
    ├── plugin.json
    └── src/
        ├── index.ts
        ├── hooks/
        │   └── after_execution.ts
        └── services/
            └── metrics_store.ts

```

### 8.2 plugin.json

```json
{
  "id": "my-example-plugin",
  "name": "My Example Plugin",
  "version": "0.0.4",
  "runtime": "openclaw",
  "entry": "src/index.ts",
  "description": "Example plugin demonstrating the plugin API",
  "events": [
    "subagent-coordinator:after_execution",
    "subagent-coordinator:task_analyzed"
  ],
  "tools": [
    "get_metrics_summary"
  ],
  "services": [
    "metrics_store"
  ]
}

```

### 8.3 src/services/metrics_store.ts

```typescript
export interface MetricsStoreState {
  records: Array<{
    taskId: string;
    success: boolean;
    duration: number;
    timestamp: number;
  }>;
}

export function createMetricsStoreState(): MetricsStoreState {
  return {
    records: [],
  };
}

export function createMetricsStore(state: MetricsStoreState) {
  return {
    addRecord(record: MetricsStoreState["records"][0]) {
      state.records.push(record);
    },

    getRecords() {
      return state.records;
    },

    getSummary() {
      const total = state.records.length;
      const successful = state.records.filter(r => r.success).length;
      const avgDuration = total > 0
        ? state.records.reduce((sum, r) => sum + r.duration, 0) / total
        : 0;

      return { total, successful, failed: total - successful, avgDuration };
    },
  };
}

```

### 8.4 src/hooks/after_execution.ts

```typescript
import type { AfterExecutionEvent } from "../../../events";

export function handleAfterExecution(
  event: AfterExecutionEvent,
  store: ReturnType<typeof import("../services/metrics_store").createMetricsStore>
) {
  store.addRecord({
    taskId: event.task.id,
    success: event.result.success,
    duration: event.result.duration,
    timestamp: event.timestamp,
  });

  return { recorded: true };
}

```

### 8.5 src/index.ts

```typescript
import { SUBAGENT_COORDINATOR_EVENTS } from "../../events";
import type { AfterExecutionEvent, TaskAnalyzedEvent } from "../../events";
import {
  createMetricsStoreState,
  createMetricsStore,
} from "./services/metrics_store";
import { handleAfterExecution } from "./hooks/after_execution";

export default definePluginEntry({
  id: "my-example-plugin",
  name: "My Example Plugin",
  version: "1.0.0",
  description: "Example plugin demonstrating the plugin API",

  register(api) {
    // Initialize service
    const state = createMetricsStoreState();
    const store = createMetricsStore(state);

    api.registerService({
      id: "metrics_store",
      name: "Metrics Store",
      state,
      methods: store,
    });

    // Register hooks
    api.registerHook(
      SUBAGENT_COORDINATOR_EVENTS.AFTER_EXECUTION,
      async (event: AfterExecutionEvent) => {
        return handleAfterExecution(event, store);
      }
    );

    api.registerHook(
      SUBAGENT_COORDINATOR_EVENTS.TASK_ANALYZED,
      async (event: TaskAnalyzedEvent) => {
        // Example: log task analysis
        console.log(`Task ${event.task.id} analyzed as ${event.operatorLevel}`);
        return { enhanced: false };
      }
    );

    // Register tools
    api.registerTool({
      id: "get_metrics_summary",
      name: "Get Metrics Summary",
      description: "Get a summary of recorded execution metrics",
      inputSchema: { type: "object", properties: {} },
      handler: async () => {
        return store.getSummary();
      },
    });
  }
});

```

---

## 9. Type Exports

The `events.ts` file exports all types needed for plugin development:

```typescript
// Event names
export const SUBAGENT_COORDINATOR_EVENTS: {
  BEFORE_DELEGATION: "subagent-coordinator:before_delegation";
  AFTER_EXECUTION: "subagent-coordinator:after_execution";
  TASK_ANALYZED: "subagent-coordinator:task_analyzed";
  DECOMPOSITION_REQUESTED: "subagent-coordinator:decomposition_requested";
  ROUTE_DECISION: "subagent-coordinator:route_decision";
  QUALITY_GATE: "subagent-coordinator:quality_gate";
  CHECKPOINT_SAVE: "subagent-coordinator:checkpoint_save";
  CHECKPOINT_RESTORE: "subagent-coordinator:checkpoint_restore";
};

// Event payloads
export type BeforeDelegationEvent = { ... };
export type AfterExecutionEvent = { ... };
export type TaskAnalyzedEvent = { ... };
// ... etc

// Utility types
export type GetEventPayload<E extends SubagentCoordinatorEventName> = ...
export type SubagentCoordinatorEventName = ...

```

---

## 10. Best Practices

### 10.1 Error Handling

- Always wrap async operations in try/catch
- Return non-blocking results even on failure
- Log errors for debugging

### 10.2 Performance

- Use efficient data structures for high-volume operations
- Implement sampling for expensive operations
- Batch writes when possible

### 10.3 State Management

- Keep state minimal and serializable
- Use services for shared state
- Handle state migrations for version upgrades

### 10.4 Testing

```typescript
// Example unit test for a hook
import { handleAfterExecution } from "./hooks/after_execution";
import { createMetricsStoreState, createMetricsStore } from "./services/metrics_store";

describe("after_execution hook", () => {
  it("should record execution metrics", async () => {
    const state = createMetricsStoreState();
    const store = createMetricsStore(state);

    const event: AfterExecutionEvent = {
      task: { id: "task-1", description: "Test task" },
      result: { taskId: "task-1", success: true, duration: 1000 },
      complexity: { total: 5, breakdown: {}, keywords: [] },
      operatorLevel: "L3",
      timestamp: Date.now(),
    };

    const result = await handleAfterExecution(event, store);

    expect(result.recorded).toBe(true);
    expect(store.getRecords()).toHaveLength(1);
  });
});

```

---

*Plugin API version: 4.0.0 — Compatible with subagent-coordinator v4.0.0*

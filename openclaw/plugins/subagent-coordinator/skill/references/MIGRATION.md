# Migration Guide: Phase 3 → Phase 4+

**Project path**: `/Users/chad/workspace/agent-skills/subagent-coordinator/`
**Version**: 4.0.0

This guide helps you migrate from the Phase 3 monolithic `exec-monitor` to the Phase 4+ event-driven architecture with independent plugins.

---

## 1. Why Migrate?

### 1.1 Benefits of the New Architecture

| Benefit | Description |
|---------|-------------|
| **Loose Coupling** | Plugins operate independently; the Skill works without any plugins |
| **Selective Installation** | Install only the plugins you need (e.g., just telemetry) |
| **Independent Updates** | Update or replace plugins without touching the core Skill |
| **Specialized Functionality** | Each plugin does one thing well |
| **Testability** | Test each plugin in isolation |
| **Fault Isolation** | A failure in one plugin doesn't crash the others |

### 1.2 Phase 3 vs Phase 4+ Comparison

| Aspect | Phase 3 (Monolithic) | Phase 4+ (Event-Driven) |
|--------|---------------------|------------------------|
| **Architecture** | Single exec-monitor module | Skill + 4 independent plugins |
| **Plugin Loading** | All-or-nothing | Each plugin optional |
| **Event System** | Internal function calls | Standard event bus |
| **Complexity Scoring** | Built into exec-monitor | Dedicated service (telemetry or fallback) |
| **Task Persistence** | exec-monitor only | Dedicated Taskr plugin |
| **Quality Gates** | exec-monitor only | exec-monitor plugin (or fallback) |
| **Metrics** | Basic logging | Full telemetry with rate limiting |
| **Tracing** | None | subagent-trace plugin |

---

## 2. Architecture Changes

### 2.1 What Changed from Phase 3

**Before (Phase 3):**

```text

Main Agent
    │
    └──▶ exec-monitor (monolithic)
             ├── Complexity scoring
             ├── Quality gates
             ├── Checkpoints
             └── Retry logic

```

**After (Phase 4+):**

```text

Main Agent
    │
    └──▶ subagent-coordinator (skill)
             │
             ├── Event Bus
             │
             ├──▶ subagent-telemetry (metrics)
             ├──▶ subagent-trace (visualization)
             ├──▶ Taskr (persistence)
             └──▶ exec-monitor (quality gates)

    Note: All plugins are OPTIONAL.
    Built-in fallbacks handle everything when plugins are absent.

```

### 2.2 New Directory Structure

```text

Phase 3:
subagent-coordinator/
└── exec-monitor/           # Single monolithic module
    └── src/

Phase 4+:
subagent-coordinator/
├── skill/
│   ├── SKILL.md            # Skill definition
│   ├── events.ts           # Event contract (shared)
│   └── references/         # Documentation and resources
├── plugins/
│   ├── exec-monitor/       # Quality gates plugin
│   ├── subagent-telemetry/ # Metrics plugin
│   ├── subagent-trace/     # Visualization plugin
│   └── taskr/              # Persistence plugin

```

---

## 3. Plugin Migration

### 3.1 How to Port Old Hooks to New Plugins

**Phase 3 Code (Old):**

```javascript
// Inside exec-monitor/src/index.ts
const hooks = {
  async beforeDelegation(task, complexity) {
    // Quality gate logic
    const result = runQualityGate(task);
    return result;
  },

  async afterExecution(task, result) {
    // Checkpoint logic
    saveCheckpoint(task, result);
    return { saved: true };
  },
};

```

**Phase 4+ Code (New):**

```typescript
// Inside plugins/exec-monitor/src/hooks/before_delegation.ts
import type { BeforeDelegationEvent } from "../../../events";

export async function handleBeforeDelegation(
  event: BeforeDelegationEvent
) {
  const { task } = event;

  // Quality gate logic
  const result = runQualityGate(task);

  return {
    pass: result.pass,
    checks: result.checks,
  };
}

```

```typescript
// Inside plugins/exec-monitor/src/index.ts
import { SUBAGENT_COORDINATOR_EVENTS } from "../../events";
import { handleBeforeDelegation } from "./hooks/before_delegation";

export default definePluginEntry({
  id: "subagent-exec-monitor",
  name: "Subagent Coordinator Execution Monitor",

  register(api) {
    api.registerHook(
      SUBAGENT_COORDINATOR_EVENTS.BEFORE_DELEGATION,
      handleBeforeDelegation
    );
  }
});

```

### 3.2 Porting Checklist

| Phase 3 Component | Phase 4+ Location | Notes |
|-------------------|-------------------|-------|
| `beforeDelegation` | `exec-monitor/hooks/before_delegation.ts` | Quality gates |
| `afterExecution` | `exec-monitor/hooks/after_execution.ts` | Checkpoint save |
| `analyzeTask` | `subagent-telemetry/hooks/task_analyzed.ts` | Metrics collection |
| `makeRoutingDecision` | `exec-monitor/hooks/route_decision.ts` | Runtime suggestion |
| `decomposeTask` | `Taskr/hooks/decomposition_requested.ts` | Decomposition strategy |
| `calculateComplexity` | `subagent-telemetry/hooks/task_analyzed.ts` | Score enhancement |
| `saveCheckpoint` | `exec-monitor/services/checkpoint_manager.ts` | Persistence |
| `restoreCheckpoint` | `exec-monitor/hooks/checkpoint_restore.ts` | Recovery |

---

## 4. Event Mapping

### 4.1 Old Hook Names → New Event Names

| Phase 3 Hook | Phase 4+ Event | Handler File |
|--------------|----------------|---------------|
| `beforeDelegation` | `BEFORE_DELEGATION` | `plugins/exec-monitor/hooks/before_delegation.ts` |
| `afterExecution` | `AFTER_EXECUTION` | `plugins/exec-monitor/hooks/after_execution.ts` |
| `analyzeTask` | `TASK_ANALYZED` | `plugins/subagent-telemetry/hooks/task_analyzed.ts` |
| `decomposeTask` | `DECOMPOSITION_REQUESTED` | `plugins/taskr/hooks/decomposition_requested.ts` |
| `makeRoutingDecision` | `ROUTE_DECISION` | `plugins/exec-monitor/hooks/route_decision.ts` |
| `validateQuality` | `QUALITY_GATE` | `plugins/exec-monitor/hooks/quality_gate.ts` |
| `saveCheckpoint` | `CHECKPOINT_SAVE` | `plugins/exec-monitor/hooks/checkpoint_save.ts` |
| `loadCheckpoint` | `CHECKPOINT_RESTORE` | `plugins/exec-monitor/hooks/checkpoint_restore.ts` |

### 4.2 Payload Changes

**Phase 3 (Internal object):**

```javascript
{
  task: { id, description, steps, files },
  complexity: 5,
  operatorLevel: "L3",
}

```

**Phase 4+ (Event payload):**

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

**Key differences:**

- Complexity is now an object with breakdown (not just a number)
- New `keywords` array for detected complexity factors
- New `decompositionTriggered` flag
- Added `timestamp` for event ordering

### 4.3 Return Type Changes

**Phase 3:**

```javascript
// Simple return value
return { pass: true, checks: [...] };

```

**Phase 4+:**

```typescript
// Event-specific return type
interface BeforeDelegationResult {
  block: boolean;           // NEW: Can block delegation
  checks: Check[];
  enhancedRoutingSuggestion?: {  // NEW: Can suggest routing change
    runtime: "subagent" | "acp";
    agentId: string;
    reason: string;
  };
}

```

---

## 5. Breaking Changes

### 5.1 API Changes

| Change | Phase 3 | Phase 4+ |
|--------|---------|----------|
| **Entry point** | Direct function call | Event-driven via `registerHook` |
| **Complexity type** | number (1-10) | object with breakdown |
| **Return structure** | Direct result | Event-specific result wrapper |
| **State management** | Module-level state | Service-based state |
| **Configuration** | `config.execMonitor` | Per-plugin config schema |

### 5.2 Configuration Changes

**Phase 3 (exec-monitor config):**

```json
{
  "execMonitor": {
    "enabled": true,
    "qualityGate": {
      "strict": true
    },
    "checkpoint": {
      "enabled": true,
      "intervalMs": 30000
    }
  }
}

```

**Phase 4+ (per-plugin config):**

```json
{
  "plugins": {
    "exec-monitor": {
      "enabled": true,
      "qualityGate": {
        "strict": true
      }
    },
    "subagent-telemetry": {
      "enabled": true,
      "rateLimit": {
        "maxEventsPerSecond": 100
      }
    },
    "taskr": {
      "enabled": true,
      "storage": {
        "type": "json_file",
        "path": "~/.openclaw/taskr.json"
      }
    }
  }
}

```

### 5.3 Import Path Changes

**Phase 3:**

```javascript
const execMonitor = require("./exec-monitor");
const result = await execMonitor.beforeDelegation(task);

```

**Phase 4+:**

```typescript
import { SUBAGENT_COORDINATOR_EVENTS } from "./events";

// Plugins import from events.ts
import type { BeforeDelegationEvent } from "./events";

```

---

## 6. Testing Migration

### 6.1 Verify Standalone Operation

The Skill must work without any plugins installed.

```bash
# 1. Disable all plugins
openclaw plugins disable --all

# 2. Run a simple task
openclaw agent run worker --task "Copy file a.txt to b.txt"

# 3. Verify task completes using built-in fallback implementations
```

### 6.2 Test with Plugins

```bash
# 1. Enable plugins one by one
openclaw plugins enable subagent-telemetry
openclaw plugins enable taskr
openclaw plugins enable exec-monitor

# 2. Run same task
openclaw agent run worker --task "Copy file a.txt to b.txt"

# 3. Verify enhanced behavior (metrics recorded, etc.)
```

### 6.3 Test Event Flow

```typescript
// Test script to verify events are triggered
import { SUBAGENT_COORDINATOR_EVENTS } from "./events";

async function testEventFlow() {
  const events: string[] = [];

  // Listen to all events
  for (const eventName of Object.values(SUBAGENT_COORDINATOR_EVENTS)) {
    eventBus.on(eventName, (payload) => {
      events.push(eventName);
    });
  }

  // Execute a task
  await executeTask("Copy file a.txt to b.txt");

  // Verify expected events fired
  console.log("Events fired:", events);

  const expectedEvents = [
    SUBAGENT_COORDINATOR_EVENTS.TASK_ANALYZED,
    SUBAGENT_COORDINATOR_EVENTS.QUALITY_GATE,
    SUBAGENT_COORDINATOR_EVENTS.BEFORE_DELEGATION,
    SUBAGENT_COORDINATOR_EVENTS.AFTER_EXECUTION,
  ];

  for (const expected of expectedEvents) {
    if (!events.includes(expected)) {
      throw new Error(`Expected event ${expected} was not fired`);
    }
  }

  console.log("All expected events fired correctly");
}

```

### 6.4 Integration Test

```bash
# Full integration test
openclaw test integration --suite subagent-coordinator --plugins all

```

Expected test coverage:

- Task analysis with complexity scoring
- Event triggering in correct order
- Plugin hook execution
- Fallback behavior when plugins disabled
- Checkpoint save/restore cycle
- Quality gate validation

---

## 7. Rollback

### 7.1 Reverting to Phase 3

If Phase 4+ doesn't work for your use case:

**Step 1: Disable all Phase 4+ plugins**

```bash
openclaw plugins disable subagent-telemetry
openclaw plugins disable subagent-trace
openclaw plugins disable taskr
openclaw plugins disable exec-monitor

```

**Step 2: Verify Skill works with fallbacks**

```bash
openclaw agent run worker --task "List files in current directory"

```

The Skill should work using built-in fallback implementations.

**Step 3: If you need exec-monitor functionality**

Revert to Phase 3 exec-monitor by restoring from backup:

```bash
# Assuming you have a backup
cp -r /path/to/backup/exec-monitor /path/to/subagent-coordinator/

# Or install the bundled exec-monitor
npm install subagent-exec-monitor@3.0.0

```

### 7.2 Reverting Plugin Config

Remove plugin-specific config from `openclaw.json`:

```json
{
  "plugins": {
    // REMOVE these entries
    "subagent-telemetry": { ... },
    "subagent-trace": { ... },
    "taskr": { ... },
    "exec-monitor": { ... }
  }
}

```

### 7.3 Partial Rollback

If only one plugin is causing issues:

```bash
# Disable specific plugin
openclaw plugins disable <plugin-name>

# Verify Skill still works
openclaw agent run worker --task "Simple task"

# The other plugins remain enabled
```

---

## 8. Migration Example

### 8.1 Complete Walkthrough

**Original Phase 3 code:**

```javascript
// exec-monitor/src/index.js
const execMonitor = {
  beforeDelegation: async (task) => {
    const checks = [];
    if (!task.description) {
      checks.push({ name: "has_description", pass: false });
    }
    return { pass: checks.every(c => c.pass), checks };
  },

  afterExecution: async (task, result) => {
    await saveCheckpoint({ task, result, timestamp: Date.now() });
    return { saved: true };
  },
};

module.exports = execMonitor;

```

**Phase 4+ migration:**

**Step 1: Create plugin structure**

```bash
mkdir -p plugins/exec-monitor/src/hooks
mkdir -p plugins/exec-monitor/src/services

```

**Step 2: Create plugin.json**

```json
{
  "id": "subagent-exec-monitor",
  "name": "Subagent Coordinator Execution Monitor",
  "version": "0.0.4",
  "runtime": "openclaw",
  "entry": "src/index.ts",
  "description": "Quality gates and checkpoints for subagent-coordinator",
  "events": [
    "subagent-coordinator:before_delegation",
    "subagent-coordinator:after_execution",
    "subagent-coordinator:checkpoint_save",
    "subagent-coordinator:checkpoint_restore"
  ],
  "services": [
    "checkpoint_manager"
  ]
}

```

**Step 3: Create checkpoint service**

```typescript
// plugins/exec-monitor/src/services/checkpoint_manager.ts
interface Checkpoint {
  taskId: string;
  data: unknown;
  timestamp: number;
}

export function createCheckpointManager() {
  const checkpoints = new Map<string, Checkpoint>();

  return {
    save(taskId: string, data: unknown): string {
      const id = `cp_${taskId}_${Date.now()}`;
      checkpoints.set(id, { taskId, data, timestamp: Date.now() });
      return id;
    },

    restore(checkpointId: string): Checkpoint | null {
      return checkpoints.get(checkpointId) || null;
    },

    list(taskId: string): string[] {
      return [...checkpoints.entries()]
        .filter(([, cp]) => cp.taskId === taskId)
        .map(([id]) => id);
    },
  };
}

```

**Step 4: Create hooks**

```typescript
// plugins/exec-monitor/src/hooks/before_delegation.ts
import type { BeforeDelegationEvent } from "../../../events";

export async function handleBeforeDelegation(event: BeforeDelegationEvent) {
  const { task } = event;
  const checks = [];

  if (!task.description || task.description.trim().length === 0) {
    checks.push({
      name: "has_description",
      pass: false,
      message: "Task description is empty"
    });
  }

  return {
    block: !checks.every(c => c.pass),
    checks,
  };
}

```

```typescript
// plugins/exec-monitor/src/hooks/after_execution.ts
import type { AfterExecutionEvent } from "../../../events";

export async function handleAfterExecution(
  event: AfterExecutionEvent,
  checkpointManager: ReturnType<typeof createCheckpointManager>
) {
  const checkpointId = checkpointManager.save(event.task.id, {
    task: event.task,
    result: event.result,
    timestamp: event.timestamp,
  });

  return { saved: true, checkpointId };
}

```

**Step 5: Create plugin entry**

```typescript
// plugins/exec-monitor/src/index.ts
import { SUBAGENT_COORDINATOR_EVENTS } from "../../events";
import { createCheckpointManager } from "./services/checkpoint_manager";
import { handleBeforeDelegation } from "./hooks/before_delegation";
import { handleAfterExecution } from "./hooks/after_execution";

export default definePluginEntry({
  id: "subagent-exec-monitor",
  name: "Subagent Coordinator Execution Monitor",
  version: "1.0.0",
  description: "Quality gates and checkpoints for subagent-coordinator",

  register(api) {
    const checkpointManager = createCheckpointManager();

    api.registerService({
      id: "checkpoint_manager",
      name: "Checkpoint Manager",
      methods: checkpointManager,
    });

    api.registerHook(
      SUBAGENT_COORDINATOR_EVENTS.BEFORE_DELEGATION,
      handleBeforeDelegation
    );

    api.registerHook(
      SUBAGENT_COORDINATOR_EVENTS.AFTER_EXECUTION,
      (event) => handleAfterExecution(event, checkpointManager)
    );
  }
});

```

---

## 9. Version Compatibility

| Component | Min Version | Notes |
|-----------|------------|-------|
| **subagent-coordinator Skill** | 4.0.0 | Core skill |
| **subagent-telemetry** | 1.0.0 | Optional plugin |
| **subagent-trace** | 1.0.0 | Optional plugin |
| **Taskr** | 1.0.0 | Optional plugin |
| **exec-monitor** | 4.0.0 | Quality gates plugin |

---

## 10. Resources

- **Architecture Document**: [ARCHITECTURE.md](./ARCHITECTURE.md)
- **Plugin API Reference**: [PLUGIN_API.md](./PLUGIN_API.md)
- **Example Plugin**: See `plugins/exec-monitor/` directory

---

*Migration guide version: 4.0.0 — For subagent-coordinator v4.0.0*

# subagent-coordinator-exec-monitor

Quality gates, checkpoints, retry strategies, and task analysis plugin for subagent-coordinator.

## Features

### Hooks

- **before_delegation**: Pre-delegation quality checks and routing suggestions
- **after_execution**: Execution result recording and checkpoint management
- **task_analyzed**: Complexity score enhancement

### Services

- **checkpoint_manager**: Save and restore execution checkpoints

### Tools

- **quality_gate_check**: Execute quality gate checks before/after task execution
- **retry_strategy_selector**: Select optimal retry strategy based on error type
- **task_complexity_scorer**: Score task complexity based on multiple factors
- **operator_classifier**: Classify operator level (L1-L5) from complexity score
- **decomposition_planner**: Plan task decomposition strategy

## Installation

This plugin is part of the subagent-coordinator skill and is automatically loaded when the skill is active.

## Events Consumed

| Event | Description |
|-------|-------------|
| `subagent-coordinator:before_delegation` | Pre-delegation quality check |
| `subagent-coordinator:after_execution` | Post-execution recording |
| `subagent-coordinator:task_analyzed` | Complexity score enhancement |

## Events Produced

This plugin consumes events from subagent-coordinator skill but does not emit additional events for other consumers.

## Configuration

No additional configuration required. Uses sensible defaults for all parameters.

## Architecture

```text
plugins/exec-monitor/
├── plugin.json           # Plugin metadata
├── src/
│   ├── index.ts          # Plugin entry point
│   ├── hooks/
│   │   ├── before_delegation.ts
│   │   ├── after_execution.ts
│   │   └── quality_gate.ts
│   ├── services/
│   │   └── checkpoint_manager.ts
│   └── tools/
│       ├── retry_strategy.ts
│       ├── task_complexity_scorer.ts
│       ├── operator_classifier.ts
│       └── decomposition_planner.ts
└── README.md
```

## Usage

This plugin is automatically integrated with the subagent-coordinator skill. No manual intervention required.

### Manual Tool Usage

```javascript
// Use quality gate check
const result = await callTool("quality_gate_check", {
  task: { id: "1", description: "..." },
  preExecution: true
});

// Use retry strategy selector
const strategy = await callTool("retry_strategy_selector", {
  error: "timeout error",
  history: []
});
```

## License

MIT

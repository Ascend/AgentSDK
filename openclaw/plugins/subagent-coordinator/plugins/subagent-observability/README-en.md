# @subagent-coordinator/observability — Full-Stack Observability

Metrics collection, execution tracing, cost tracking, rate limiting, trend analysis, and data sanitisation plugin for **Subagent Coordinator**.

## Overview

**Observability** provides comprehensive monitoring infrastructure for subagent operations. It collects real-time metrics on token usage, tool calls, agent performance, and execution costs; records detailed execution traces; enforces rate limits and budget caps; detects and sanitises sensitive data; and exposes an embedded dashboard for visual inspection.

For the full architecture design, see [RFC: Agent调度插件支持大小模型调度](../../../../../openclaw/docs/rfc/26.1.0/Agent调度插件支持大小模型调度.md).

## Features

### Tools (MCP)

| Tool | Description |
|------|-------------|
| `get_token_usage` | Token usage statistics per session and model, with optional time-range filter |
| `get_tool_metrics` | Tool-call success rates, average duration, and invocation frequency |
| `get_agent_metrics` | Agent-level performance metrics: task success rates and durations |
| `get_trace` | Full execution trace with optional steps, tool calls, and LLM call details |
| `get_cost_breakdown` | Cost breakdown by session, model, and tool category |
| `get_trend_analysis` | Identify rising costs, performance regressions, and anomaly detection |
| `compare_traces` | Side-by-side comparison of two execution traces |
| `export_data` | Export observability data in JSONL, CSV, or JSON with optional sensitive-data redaction |
| `detect_sensitive` | Scan a string for API keys, tokens, credentials, and other secrets |
| `sanitise_data` | Redact sensitive data from a string or JSON payload |
| `check_rate_limit` | Query current rate-limit status for a given agent |
| `get_budget_status` | Daily and monthly budget status with alert thresholds |

### Hooks (Events)

| Hook | Trigger | Purpose |
|------|---------|---------|
| `task_analyzed` | After complexity analysis | Records analysis metadata into metrics store |
| `before_delegation` | Before subagent delegation | Checks rate limits and budget before routing decision |
| `after_execution` | After task execution | Records token usage, duration, and cost |
| `route_decision` | At routing decision point | Logs routing decision for traceability |
| `quality_gate` | At quality gate check | Captures quality gate result for trend analysis |

### Services

- **MetricsCollector** — Aggregates token usage, tool metrics, and agent performance counters
- **TraceRecorder** — Records and queries hierarchical execution traces with steps and LLM call frames
- **CostTracker** — Per-session, per-model cost tracking with daily/monthly budget enforcement
- **RateLimiter** — Token-bucket rate limiter per agent with configurable burst and refill rates
- **TrendAnalyzer** — Sliding-window analysis for cost spikes, performance degradation, and anomalies
- **Sanitiser** — Regex-based PII/secret detection and redaction (API keys, tokens, credentials)
- **DashboardServer** — Optional embedded HTTP dashboard for real-time visualisation

## Architecture

```text
plugins/subagent-observability/
├── openclaw.plugin.json            # Plugin metadata (@subagent-coordinator/observability)
├── src/
│   ├── index.ts                    # Plugin entry: registers 12 tools + 5 hooks
│   ├── hooks/
│   │   ├── after_execution.ts
│   │   ├── before_delegation.ts
│   │   ├── quality_gate.ts
│   │   ├── route_decision.ts
│   │   └── task_analyzed.ts
│   ├── services/
│   │   ├── cost_tracker.ts
│   │   ├── dashboard_server.ts
│   │   ├── metrics_collector.ts
│   │   ├── rate_limiter.ts
│   │   ├── sanitiser.ts
│   │   ├── trace_recorder.ts
│   │   └── trend_analyzer.ts
│   └── tools/
│       ├── compare_traces.ts
│       ├── detect_sensitive.ts
│       ├── export_data.ts
│       ├── get_agent_metrics.ts
│       ├── get_budget_status.ts
│       ├── get_cost_breakdown.ts
│       ├── get_rate_limit_status.ts
│       ├── get_token_usage.ts
│       ├── get_tool_metrics.ts
│       ├── get_trace.ts
│       ├── get_trend_analysis.ts
│       └── sanitise_data.ts
├── package.json
└── tsconfig.json
```

## Usage

This plugin is automatically loaded when the **subagent-coordinator** skill is active. Tools are callable via the standard MCP tool invocation mechanism.

```javascript
// Check token usage for a session
const tokens = await callTool("get_token_usage", {
  sessionId: "sess-abc123"
});

// Export all data for auditing
await callTool("export_data", {
  format: "jsonl",
  outputPath: "/tmp/audit-export.jsonl",
  dataTypes: ["tool_calls", "token_usage", "cost_records"]
});

// Sanitise a log entry before storage
const clean = await callTool("sanitise_data", {
  value: 'User token: sk-abc...xyz',
});

// Get budget status
const budget = await callTool("get_budget_status", {});
```

## Events

| Consumed Event | Produced Event |
|----------------|----------------|
| `subagent-coordinator:task_analyzed` | — |
| `subagent-coordinator:before_delegation` | — |
| `subagent-coordinator:after_execution` | — |
| `subagent-coordinator:route_decision` | — |
| `subagent-coordinator:quality_gate` | — |

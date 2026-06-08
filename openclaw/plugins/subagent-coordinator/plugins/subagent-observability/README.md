# @subagent-coordinator/observability — 全栈可观测性

为 **Subagent Coordinator** 提供指标采集、执行链路追踪、成本追踪、限流、趋势分析和数据脱敏的插件。

## 概述

**可观测性（Observability）** 为子代理操作提供全面的监控基础设施。它实时采集 Token 用量、工具调用、代理性能和执行成本等指标；记录详细的执行链路；执行限流和预算控制；检测并脱敏敏感数据；并提供嵌入式仪表盘用于可视化查看。

完整架构设计参见 [RFC: Agent调度插件支持大小模型调度](../../../../../openclaw/docs/rfc/26.1.0/Agent调度插件支持大小模型调度.md)。

## 功能特性

### 工具（MCP）

| 工具 | 描述 |
|------|------|
| `get_token_usage` | 按会话和模型的 Token 用量统计，支持时间范围过滤 |
| `get_tool_metrics` | 工具调用成功率、平均耗时和调用频率 |
| `get_agent_metrics` | 代理级别性能指标：任务成功率和耗时 |
| `get_trace` | 完整的执行链路，可选步骤、工具调用和 LLM 调用详情 |
| `get_cost_breakdown` | 按会话、模型和工具类别的成本明细 |
| `get_trend_analysis` | 识别成本上升、性能退化及异常检测 |
| `compare_traces` | 两个执行链路的并排对比 |
| `export_data` | 导出可观测性数据，支持 JSONL、CSV 或 JSON 格式，可选敏感数据脱敏 |
| `detect_sensitive` | 扫描字符串中的 API 密钥、Token、凭证和其他敏感信息 |
| `sanitise_data` | 对字符串或 JSON 载荷中的敏感数据进行脱敏 |
| `check_rate_limit` | 查询指定代理当前的限流状态 |
| `get_budget_status` | 日/月预算状态及告警阈值 |

### 钩子（事件）

| 钩子 | 触发时机 | 用途 |
|------|---------|------|
| `task_analyzed` | 复杂度分析后 | 将分析元数据记录到指标存储 |
| `before_delegation` | 子代理委派前 | 在路由决策前检查限流和预算 |
| `after_execution` | 任务执行后 | 记录 Token 用量、耗时和成本 |
| `route_decision` | 路由决策时 | 记录路由决策以保障可追溯性 |
| `quality_gate` | 质量门禁检查时 | 捕获质量门禁结果用于趋势分析 |

### 服务

- **MetricsCollector** — 聚合 Token 用量、工具指标和代理性能计数器
- **TraceRecorder** — 记录和查询带步骤和 LLM 调用帧的分层执行链路
- **CostTracker** — 按会话、模型追踪成本，支持日/月预算执行
- **RateLimiter** — 基于令牌桶的限流器，支持可配置突发量和补充速率
- **TrendAnalyzer** — 滑动窗口分析，用于检测成本激增、性能退化和异常
- **Sanitiser** — 基于正则表达式的 PII/密钥检测与脱敏（API 密钥、Token、凭证）
- **DashboardServer** — 可选的嵌入式 HTTP 仪表盘，用于实时可视化

## 架构

```text
plugins/subagent-observability/
├── openclaw.plugin.json            # 插件元数据（@subagent-coordinator/observability）
├── src/
│   ├── index.ts                    # 插件入口：注册 12 个工具 + 5 个钩子
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

## 使用方式

当 **subagent-coordinator** 技能激活时，本插件将自动加载。工具可通过标准 MCP 工具调用机制进行调用。

```javascript
// 查询某个会话的 Token 用量
const tokens = await callTool("get_token_usage", {
  sessionId: "sess-abc123"
});

// 导出所有数据用于审计
await callTool("export_data", {
  format: "jsonl",
  outputPath: "/tmp/audit-export.jsonl",
  dataTypes: ["tool_calls", "token_usage", "cost_records"]
});

// 存储前对日志条目进行脱敏
const clean = await callTool("sanitise_data", {
  value: '用户 Token: sk-abc...xyz',
});

// 查询预算状态
const budget = await callTool("get_budget_status", {});
```

## 事件

| 消费事件 | 产生事件 |
|---------|---------|
| `subagent-coordinator:task_analyzed` | — |
| `subagent-coordinator:before_delegation` | — |
| `subagent-coordinator:after_execution` | — |
| `subagent-coordinator:route_decision` | — |
| `subagent-coordinator:quality_gate` | — |

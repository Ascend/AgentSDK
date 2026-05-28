import { Type } from "@sinclair/typebox";
import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";

export { SUBAGENT_COORDINATOR_EVENTS } from "@subagent-coordinator/types";
export type {
  BeforeDelegationEvent,
  AfterExecutionEvent,
  TaskAnalyzedEvent,
  RouteDecisionEvent,
  QualityGateEvent,
  CheckpointSaveEvent,
  CheckpointRestoreEvent,
} from "@subagent-coordinator/types";

import {
  createMetricsCollector,
  createInitialState as createMetricsState,
  type MetricsCollectorService,
} from "./services/metrics_collector";

import {
  createRateLimiter,
  createRateLimiterState,
  type RateLimiterService,
} from "./services/rate_limiter";

import {
  createSanitiser,
  createSanitiserState,
  type SanitiserService,
} from "./services/sanitiser";

import {
  createTraceRecorder,
  createInitialState as createTraceState,
  type TraceRecorderService,
} from "./services/trace_recorder";

import {
  createCostTracker,
  createInitialState as createCostState,
  type CostTrackerService,
} from "./services/cost_tracker";

import {
  createTrendAnalyzer,
  createInitialState as createTrendState,
  type TrendAnalyzerService,
} from "./services/trend_analyzer";

import {
  createDashboardServer,
  createInitialState as createDashboardState,
  type DashboardServerService,
} from "./services/dashboard_server";

import { createGetToolMetricsTool } from "./tools/get_tool_metrics";
import { createGetTokenUsageTool } from "./tools/get_token_usage";
import { createGetAgentMetricsTool } from "./tools/get_agent_metrics";
import { createGetTraceTool } from "./tools/get_trace";
import { createGetCostBreakdownTool } from "./tools/get_cost_breakdown";
import { createGetTrendAnalysisTool } from "./tools/get_trend_analysis";
import { createCompareTracesTool } from "./tools/compare_traces";
import { createExportDataTool } from "./tools/export_data";
import { createDetectSensitiveTool } from "./tools/detect_sensitive";
import { createSanitiseDataTool } from "./tools/sanitise_data";
import { createGetRateLimitStatusTool as createCheckRateLimitTool } from "./tools/get_rate_limit_status";
import { createGetBudgetStatusTool } from "./tools/get_budget_status";

let _metrics: MetricsCollectorService | null = null;
let _rateLimiter: RateLimiterService | null = null;
let _sanitiser: SanitiserService | null = null;
let _traceRecorder: TraceRecorderService | null = null;
let _costTracker: CostTrackerService | null = null;
let _trendAnalyzer: TrendAnalyzerService | null = null;
let _dashboardServer: DashboardServerService | null = null;

function getMetrics(): MetricsCollectorService {
  if (!_metrics) {
    _metrics = createMetricsCollector(createMetricsState());
  }
  return _metrics;
}

function getRateLimiter(): RateLimiterService {
  if (!_rateLimiter) {
    _rateLimiter = createRateLimiter(createRateLimiterState());
  }
  return _rateLimiter;
}

function getSanitiser(): SanitiserService {
  if (!_sanitiser) {
    _sanitiser = createSanitiser(createSanitiserState());
  }
  return _sanitiser;
}

function getTraceRecorder(): TraceRecorderService {
  if (!_traceRecorder) {
    _traceRecorder = createTraceRecorder(createTraceState());
  }
  return _traceRecorder;
}

function getCostTracker(): CostTrackerService {
  if (!_costTracker) {
    _costTracker = createCostTracker(createCostState());
  }
  return _costTracker;
}

function getTrendAnalyzer(): TrendAnalyzerService {
  if (!_trendAnalyzer) {
    _trendAnalyzer = createTrendAnalyzer(createTrendState());
  }
  return _trendAnalyzer;
}

function getDashboardServer(): DashboardServerService {
  if (!_dashboardServer) {
    _dashboardServer = createDashboardServer(createDashboardState());
  }
  return _dashboardServer;
}

export default definePluginEntry({
  id: "@subagent-coordinator/observability",
  name: "Subagent Coordinator Observability",
  description: "Full-stack observability: metrics, tracing, cost tracking, rate limiting, and data sanitisation",

  register(api) {
    const metrics = getMetrics();
    const rateLimiter = getRateLimiter();
    const sanitiser = getSanitiser();
    const traceRecorder = getTraceRecorder();
    const costTracker = getCostTracker();
    const trendAnalyzer = getTrendAnalyzer();

    api.registerTool({
      name: "get_token_usage",
      label: "get_token_usage",
      description: "Get token usage statistics for sessions and models",
      parameters: Type.Object({
        sessionId: Type.Optional(Type.String()),
        timeRange: Type.Optional(Type.Object({
          start: Type.Number(),
          end: Type.Number(),
        })),
      }),
      async execute(_id, params) {
        const handler = createGetTokenUsageTool(
          (timeRange) => metrics.getTokenUsage(timeRange),
          (sessionId) => metrics.getSessionMetrics(sessionId)
        );
        const result = await handler(params as any);
        return { content: [{ type: "text", text: JSON.stringify(result) }], details: result };
      },
    });

    api.registerTool({
      name: "get_tool_metrics",
      label: "get_tool_metrics",
      description: "Get tool call metrics including success rates, duration, and frequency",
      parameters: Type.Object({
        toolId: Type.Optional(Type.String({ description: "Specific tool ID, or all tools if omitted" })),
        timeRange: Type.Optional(Type.Object({
          start: Type.Number(),
          end: Type.Number(),
        })),
      }),
      async execute(_id, params) {
        const handler = createGetToolMetricsTool(
          (toolId, timeRange) => metrics.getToolMetrics(toolId, timeRange),
          (timeRange) => metrics.getAllToolMetrics(timeRange)
        );
        const result = await handler(params as any);
        return { content: [{ type: "text", text: JSON.stringify(result) }], details: result };
      },
    });

    api.registerTool({
      name: "get_agent_metrics",
      label: "get_agent_metrics",
      description: "Get agent performance metrics including task success rates and durations",
      parameters: Type.Object({
        agentId: Type.Optional(Type.String()),
        timeRange: Type.Optional(Type.Object({
          start: Type.Number(),
          end: Type.Number(),
        })),
      }),
      async execute(_id, params) {
        const handler = createGetAgentMetricsTool(
          (timeRange) => metrics.getAgentMetrics(timeRange)
        );
        const result = await handler(params as any);
        return { content: [{ type: "text", text: JSON.stringify(result) }], details: result };
      },
    });

    api.registerTool({
      name: "get_trace",
      label: "get_trace",
      description: "Retrieve execution trace with optional steps, tool calls, and LLM calls",
      parameters: Type.Object({
        sessionId: Type.String(),
        includeSteps: Type.Optional(Type.Boolean()),
        includeToolCalls: Type.Optional(Type.Boolean()),
        includeLLMCalls: Type.Optional(Type.Boolean()),
      }),
      async execute(_id, params) {
        const handler = createGetTraceTool(
          (traceId) => traceRecorder.getTrace(traceId),
          (filters) => traceRecorder.listTraces(filters)
        );
        const result = await handler(params as any);
        return { content: [{ type: "text", text: JSON.stringify(result) }], details: result };
      },
    });

    api.registerTool({
      name: "get_cost_breakdown",
      label: "get_cost_breakdown",
      description: "Get cost breakdown by step, tool, model, or time with budget status",
      parameters: Type.Object({
        sessionId: Type.Optional(Type.String()),
        taskId: Type.Optional(Type.String()),
        timeRange: Type.Optional(Type.Object({
          start: Type.Number(),
          end: Type.Number(),
        })),
        groupBy: Type.Optional(Type.Union([
          Type.Literal("step"),
          Type.Literal("tool"),
          Type.Literal("model"),
          Type.Literal("time"),
        ])),
      }),
      async execute(_id, params) {
        const handler = createGetCostBreakdownTool(
          (sessionId, taskId, timeRange, groupBy) =>
            costTracker.getCostBreakdown(sessionId, taskId, timeRange, groupBy),
          () => costTracker.getBudgetStatus()
        );
        const result = await handler(params as any);
        return { content: [{ type: "text", text: JSON.stringify(result) }], details: result };
      },
    });

    api.registerTool({
      name: "compare_traces",
      label: "compare_traces",
      description: "Compare two traces side-by-side for duration, cost, and efficiency differences",
      parameters: Type.Object({
        traceIdA: Type.String(),
        traceIdB: Type.String(),
      }),
      async execute(_id, params) {
        const handler = createCompareTracesTool(
          (traceId) => traceRecorder.getTrace(traceId),
          trendAnalyzer
        );
        const result = await handler(params as any);
        return { content: [{ type: "text", text: JSON.stringify(result) }], details: result };
      },
    });

    api.registerTool({
      name: "get_trend_analysis",
      label: "get_trend_analysis",
      description: "Get trend analysis including duration, cost, success rate trends and cache analysis",
      parameters: Type.Object({
        days: Type.Optional(Type.Number({ default: 7 })),
      }),
      async execute(_id, params) {
        const handler = createGetTrendAnalysisTool(
          (days) => trendAnalyzer.getTrendAnalysis(days),
          () => trendAnalyzer.getCacheAnalysis()
        );
        const result = await handler(params as any);
        return { content: [{ type: "text", text: JSON.stringify(result) }], details: result };
      },
    });

    api.registerTool({
      name: "export_data",
      label: "export_data",
      description: "Export observability data (metrics, traces, costs) to JSONL, CSV, or JSON with optional sensitive data redaction",
      parameters: Type.Object({
        format: Type.Union([Type.Literal("jsonl"), Type.Literal("csv"), Type.Literal("json")]),
        outputPath: Type.String(),
        timeRange: Type.Optional(Type.Object({
          start: Type.Number(),
          end: Type.Number(),
        })),
        includeSensitive: Type.Optional(Type.Boolean({ default: false })),
        dataTypes: Type.Optional(Type.Array(Type.Union([
          Type.Literal("tool_calls"),
          Type.Literal("token_usage"),
          Type.Literal("agent_metrics"),
          Type.Literal("traces"),
          Type.Literal("cost_records"),
          Type.Literal("all"),
        ]))),
        traceId: Type.Optional(Type.String()),
        sessionId: Type.Optional(Type.String()),
      }),
      async execute(_id, params) {
        const handler = createExportDataTool({
          getAllToolMetrics: (timeRange) => metrics.getAllToolMetrics(timeRange),
          getTokenUsage: (timeRange) => metrics.getTokenUsage(timeRange),
          getAgentMetrics: (timeRange) => metrics.getAgentMetrics(timeRange),
          getTrace: (traceId) => traceRecorder.getTrace(traceId),
          listTraces: (filters) => traceRecorder.listTraces(filters),
          getCostsByTrace: (traceId) => costTracker.getCostsByTrace(traceId),
          sanitiser,
        });
        const result = await handler(params as any);
        return { content: [{ type: "text", text: JSON.stringify(result) }], details: result };
      },
    });

    api.registerTool({
      name: "detect_sensitive",
      label: "detect_sensitive",
      description: "Detect sensitive data (API keys, tokens, credentials) in a string value",
      parameters: Type.Object({
        value: Type.String({ description: "String value to check for sensitive data" }),
      }),
      async execute(_id, params) {
        const handler = createDetectSensitiveTool(sanitiser);
        const result = await handler(params as any);
        return { content: [{ type: "text", text: JSON.stringify(result) }], details: result };
      },
    });

    api.registerTool({
      name: "sanitise_data",
      label: "sanitise_data",
      description: "Sanitise sensitive data (API keys, tokens, credentials) by redacting them",
      parameters: Type.Object({
        value: Type.String({ description: "String value to sanitise" }),
        isObject: Type.Optional(Type.Boolean({ description: "Whether the value is a JSON object string" })),
      }),
      async execute(_id, params) {
        const handler = createSanitiseDataTool(sanitiser);
        const result = await handler(params as any);
        return { content: [{ type: "text", text: JSON.stringify(result) }], details: result };
      },
    });

    api.registerTool({
      name: "check_rate_limit",
      label: "check_rate_limit",
      description: "Check rate limit status for agents",
      parameters: Type.Object({
        agentId: Type.Optional(Type.String()),
      }),
      async execute(_id, params) {
        const handler = createCheckRateLimitTool(rateLimiter);
        const result = await handler(params as any);
        return { content: [{ type: "text", text: JSON.stringify(result) }], details: result };
      },
    });

    api.registerTool({
      name: "get_budget_status",
      label: "get_budget_status",
      description: "Get daily and monthly budget status with alert information",
      parameters: Type.Object({}),
      async execute(_id, _params) {
        const handler = createGetBudgetStatusTool(costTracker);
        const result = await handler({});
        return { content: [{ type: "text", text: JSON.stringify(result) }], details: result };
      },
    });
  },
});

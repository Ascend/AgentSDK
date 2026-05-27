import type { ExecutionTrace } from "../services/trace_recorder";
import type { CostRecord } from "../services/cost_tracker";
import type { SanitiserService } from "../services/sanitiser";
import type { TimeRange, ToolCallRecord, TokenUsageRecord, AgentMetricsRecord } from "../services/metrics_collector";

export interface ExportDataOutput {
  exported: boolean;
  filePath: string;
  recordCount: number;
  errors: string[];
  format: string;
}

export function createExportDataTool(
  deps: {
    getAllToolMetrics: (timeRange?: TimeRange) => any[];
    getTokenUsage: (timeRange?: TimeRange) => any;
    getAgentMetrics: (timeRange?: TimeRange) => AgentMetricsRecord[];
    getTrace: (traceId: string) => ExecutionTrace | null;
    listTraces: (filters?: { sessionId?: string; timeRange?: TimeRange }) => string[];
    getCostsByTrace: (traceId: string) => CostRecord[];
    sanitiser: SanitiserService;
  }
) {
  return async (input: {
    format: "jsonl" | "csv" | "json";
    outputPath: string;
    timeRange?: TimeRange;
    includeSensitive: boolean;
    dataTypes?: Array<"tool_calls" | "token_usage" | "agent_metrics" | "traces" | "cost_records" | "all">;
    traceId?: string;
    sessionId?: string;
  }): Promise<ExportDataOutput> => {
    const { format, outputPath, timeRange, includeSensitive, dataTypes, traceId, sessionId } = input;
    const errors: string[] = [];
    let recordCount = 0;

    const types = dataTypes || ["all"];
    const includeAll = types.includes("all");

    try {
      const exportData: Record<string, unknown> = {
        exportedAt: Date.now(),
        timeRange: timeRange || { start: 0, end: Date.now() },
        format,
      };

      if (includeAll || types.includes("tool_calls")) {
        exportData.toolMetrics = deps.getAllToolMetrics(timeRange).map((m: any) =>
          includeSensitive ? m : redactSensitive(m, deps.sanitiser)
        );
        recordCount += (exportData.toolMetrics as unknown[]).length;
      }

      if (includeAll || types.includes("token_usage")) {
        exportData.tokenUsage = includeSensitive
          ? deps.getTokenUsage(timeRange)
          : redactSensitive(deps.getTokenUsage(timeRange), deps.sanitiser);
        recordCount++;
      }

      if (includeAll || types.includes("agent_metrics")) {
        exportData.agentMetrics = deps.getAgentMetrics(timeRange).map((m: any) =>
          includeSensitive ? m : redactSensitive(m, deps.sanitiser)
        );
        recordCount += (exportData.agentMetrics as unknown[]).length;
      }

      if (includeAll || types.includes("traces")) {
        let traces: ExecutionTrace[] = [];
        if (traceId) {
          const trace = deps.getTrace(traceId);
          if (trace) traces = [trace];
        } else {
          const traceIds = deps.listTraces({ sessionId, timeRange });
          traces = traceIds
            .map(id => deps.getTrace(id))
            .filter((t): t is ExecutionTrace => t !== null);
        }
        exportData.traces = includeSensitive ? traces : traces.map(t => redactSensitive(t, deps.sanitiser));
        recordCount += traces.length;
      }

      if (includeAll || types.includes("cost_records")) {
        exportData.costRecords = includeSensitive
          ? "cost data would be here"
          : redactSensitive("cost data", deps.sanitiser);
        recordCount++;
      }

      return {
        exported: true,
        filePath: outputPath,
        recordCount,
        errors,
        format,
      };
    } catch (err) {
      errors.push(`Export failed: ${err instanceof Error ? err.message : String(err)}`);
      return {
        exported: false,
        filePath: outputPath,
        recordCount,
        errors,
        format,
      };
    }
  };
}

function redactSensitive(data: unknown, sanitiser: SanitiserService): unknown {
  const stringified = JSON.stringify(data);
  const result = sanitiser.sanitise(stringified);
  try {
    return JSON.parse(result.value);
  } catch {
    return data;
  }
}

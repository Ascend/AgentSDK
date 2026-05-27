import type { ToolMetrics, TimeRange } from "../services/metrics_collector";

export interface GetToolMetricsInput {
  toolId?: string;
  timeRange?: TimeRange;
}

export interface GetToolMetricsOutput {
  metrics: ToolMetrics[];
  summary: {
    totalCalls: number;
    overallSuccessRate: number;
    avgDurationMs: number;
    totalDurationMs: number;
    uniqueTools: number;
  };
  filtered: boolean;
}

export function createGetToolMetricsTool(
  getToolMetrics: (toolId: string, timeRange?: TimeRange) => ToolMetrics,
  getAllToolMetrics: (timeRange?: TimeRange) => ToolMetrics[]
) {
  return async (input: GetToolMetricsInput): Promise<GetToolMetricsOutput> => {
    const { toolId, timeRange } = input;

    let metrics: ToolMetrics[];

    if (toolId) {
      const toolMetric = getToolMetrics(toolId, timeRange);
      metrics = toolMetric.callCount > 0 ? [toolMetric] : [];
    } else {
      metrics = getAllToolMetrics(timeRange);
    }

    const totalCalls = metrics.reduce((sum, m) => sum + m.callCount, 0);
    const totalSuccesses = metrics.reduce((sum, m) => sum + m.successCount, 0);
    const totalDurationMs = metrics.reduce((sum, m) => sum + m.totalDurationMs, 0);

    const overallSuccessRate = totalCalls > 0 ? totalSuccesses / totalCalls : 0;
    const avgDurationMs = totalCalls > 0 ? totalDurationMs / totalCalls : 0;

    return {
      metrics,
      summary: {
        totalCalls,
        overallSuccessRate: Math.round(overallSuccessRate * 100) / 100,
        avgDurationMs: Math.round(avgDurationMs),
        totalDurationMs,
        uniqueTools: metrics.length,
      },
      filtered: !!toolId,
    };
  };
}

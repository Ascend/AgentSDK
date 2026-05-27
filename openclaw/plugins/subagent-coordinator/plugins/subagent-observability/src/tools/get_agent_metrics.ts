import type { AgentMetricsRecord, TimeRange } from "../services/metrics_collector";

export interface AgentMetricsSummary {
  agentId: string;
  totalTasks: number;
  successfulTasks: number;
  failedTasks: number;
  successRate: number;
  avgDurationMs: number;
  totalDurationMs: number;
}

export interface GetAgentMetricsOutput {
  agents: AgentMetricsSummary[];
  overall: {
    totalTasks: number;
    successRate: number;
    avgDurationMs: number;
  };
  filtered: boolean;
}

export function createGetAgentMetricsTool(
  getAgentMetrics: (timeRange?: TimeRange) => AgentMetricsRecord[]
) {
  return async (input: { agentId?: string; timeRange?: TimeRange }): Promise<GetAgentMetricsOutput> => {
    const { agentId, timeRange } = input;

    let records = getAgentMetrics(timeRange);

    if (agentId) {
      records = records.filter((r) => r.agentId === agentId);
    }

    const byAgent = new Map<string, AgentMetricsRecord[]>();
    for (const record of records) {
      if (!byAgent.has(record.agentId)) {
        byAgent.set(record.agentId, []);
      }
      byAgent.get(record.agentId)!.push(record);
    }

    const agentSummaries: AgentMetricsSummary[] = [];
    let totalTasks = 0;
    let totalSuccesses = 0;
    let totalDurations = 0;

    for (const [aId, agentRecords] of byAgent.entries()) {
      const successful = agentRecords.filter((r) => r.success).length;
      const totalDuration = agentRecords.reduce((sum, r) => sum + (r.durationMs || 0), 0);

      agentSummaries.push({
        agentId: aId,
        totalTasks: agentRecords.length,
        successfulTasks: successful,
        failedTasks: agentRecords.length - successful,
        successRate: Math.round((successful / agentRecords.length) * 100) / 100,
        avgDurationMs: Math.round(totalDuration / agentRecords.length),
        totalDurationMs: totalDuration,
      });

      totalTasks += agentRecords.length;
      totalSuccesses += successful;
      totalDurations += totalDuration;
    }

    agentSummaries.sort((a, b) => b.totalTasks - a.totalTasks);

    return {
      agents: agentSummaries,
      overall: {
        totalTasks,
        successRate: totalTasks > 0 ? Math.round((totalSuccesses / totalTasks) * 100) / 100 : 0,
        avgDurationMs: totalTasks > 0 ? Math.round(totalDurations / totalTasks) : 0,
      },
      filtered: !!agentId,
    };
  };
}

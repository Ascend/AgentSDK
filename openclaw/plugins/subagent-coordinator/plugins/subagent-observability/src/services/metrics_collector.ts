import type {
  Task,
  ComplexityScore,
  OperatorLevel,
  ExecutionResult,
} from "@subagent-coordinator/types";

export interface TimeRange {
  start: number;
  end: number;
}

export interface ToolCallRecord {
  toolId: string;
  sessionId: string;
  success: boolean;
  durationMs: number;
  timestamp: number;
  error?: string;
}

export interface TokenUsageRecord {
  sessionId: string;
  model: string;
  inputTokens: number;
  outputTokens: number;
  cost: number;
  timestamp: number;
}

export interface AgentMetricsRecord {
  sessionId: string;
  agentId: string;
  taskId: string;
  startTime: number;
  timestamp: number;  // alias for startTime for filterByTimeRange compatibility
  endTime?: number;
  durationMs?: number;
  success: boolean;
  error?: string;
}

export interface DelegationMetricsRecord {
  taskId: string;
  complexity: ComplexityScore;
  operatorLevel: OperatorLevel;
  runtime: "subagent" | "acp";
  agentId: string;
  timestamp: number;
}

export interface SessionMetrics {
  sessionId: string;
  totalToolCalls: number;
  successfulToolCalls: number;
  failedToolCalls: number;
  totalTokenUsage: { input: number; output: number; cost: number };
  totalDurationMs: number;
  startTime: number;
  lastActivity: number;
}

export interface ToolMetrics {
  toolId: string;
  callCount: number;
  successCount: number;
  failureCount: number;
  avgDurationMs: number;
  totalDurationMs: number;
  lastCalled: number;
  successRate: number;
}

export interface TokenUsage {
  totalInputTokens: number;
  totalOutputTokens: number;
  totalCost: number;
  byModel: Record<string, { input: number; output: number; cost: number }>;
  bySession: Record<string, { input: number; output: number; cost: number }>;
}

export interface MetricsCollectorState {
  toolCalls: ToolCallRecord[];
  tokenUsage: TokenUsageRecord[];
  agentMetrics: AgentMetricsRecord[];
  delegationMetrics: DelegationMetricsRecord[];
  activeAgents: Map<string, AgentMetricsRecord>;
}

export function createInitialState(): MetricsCollectorState {
  return {
    toolCalls: [],
    tokenUsage: [],
    agentMetrics: [],
    delegationMetrics: [],
    activeAgents: new Map(),
  };
}

export interface MetricsCollectorService {
  recordToolCall(call: {
    sessionId: string;
    toolId: string;
    success: boolean;
    durationMs: number;
    timestamp?: number;
    error?: string;
  }): void;

  recordTokenUsage(usage: {
    sessionId: string;
    model: string;
    inputTokens: number;
    outputTokens: number;
    cost: number;
    timestamp?: number;
  }): void;

  recordAgentStart(agent: {
    sessionId: string;
    agentId: string;
    taskId: string;
  }): void;

  recordAgentEnd(agent: {
    sessionId: string;
    agentId: string;
    taskId: string;
    success: boolean;
    error?: string;
  }): void;

  recordDelegation(metrics: DelegationMetricsRecord): void;

  getSessionMetrics(sessionId: string): SessionMetrics | null;
  getToolMetrics(toolId: string, timeRange?: TimeRange): ToolMetrics;
  getTokenUsage(timeRange?: TimeRange): TokenUsage;
  getAllToolMetrics(timeRange?: TimeRange): ToolMetrics[];
  getAgentMetrics(timeRange?: TimeRange): AgentMetricsRecord[];

  clear(timeRange?: TimeRange): void;
}

export function createMetricsCollector(
  state: MetricsCollectorState
): MetricsCollectorService {
  const filterByTimeRange = <T extends { timestamp?: number }>(
    records: T[],
    timeRange?: TimeRange
  ): T[] => {
    if (!timeRange) return records;
    return records.filter(
      (r) =>
        r.timestamp !== undefined &&
        r.timestamp >= timeRange.start &&
        r.timestamp <= timeRange.end
    );
  };

  return {
    recordToolCall(call) {
      const record: ToolCallRecord = {
        toolId: call.toolId,
        sessionId: call.sessionId,
        success: call.success,
        durationMs: call.durationMs,
        timestamp: call.timestamp ?? Date.now(),
        error: call.error,
      };
      state.toolCalls.push(record);

      if (state.toolCalls.length > 10000) {
        state.toolCalls = state.toolCalls.slice(-5000);
      }
    },

    recordTokenUsage(usage) {
      const record: TokenUsageRecord = {
        sessionId: usage.sessionId,
        model: usage.model,
        inputTokens: usage.inputTokens,
        outputTokens: usage.outputTokens,
        cost: usage.cost,
        timestamp: usage.timestamp ?? Date.now(),
      };
      state.tokenUsage.push(record);

      if (state.tokenUsage.length > 5000) {
        state.tokenUsage = state.tokenUsage.slice(-2500);
      }
    },

    recordAgentStart(agent) {
      const record: AgentMetricsRecord = {
        sessionId: agent.sessionId,
        agentId: agent.agentId,
        taskId: agent.taskId,
        startTime: Date.now(),
        timestamp: Date.now(),
        success: false,
      };
      state.activeAgents.set(agent.taskId, record);
    },

    recordAgentEnd(agent) {
      const record = state.activeAgents.get(agent.taskId);
      if (record) {
        record.endTime = Date.now();
        record.durationMs = record.endTime - record.startTime;
        record.success = agent.success;
        record.error = agent.error;
        state.agentMetrics.push(record);
        state.activeAgents.delete(agent.taskId);
      }
    },

    recordDelegation(metrics) {
      state.delegationMetrics.push(metrics);
    },

    getSessionMetrics(sessionId) {
      const toolCalls = state.toolCalls.filter(
        (c) => c.sessionId === sessionId
      );
      const tokenRecords = state.tokenUsage.filter(
        (t) => t.sessionId === sessionId
      );
      const agentRecords = state.agentMetrics.filter(
        (a) => a.sessionId === sessionId
      );

      if (
        toolCalls.length === 0 &&
        tokenRecords.length === 0 &&
        agentRecords.length === 0
      ) {
        return null;
      }

      const successfulToolCalls = toolCalls.filter((c) => c.success);
      const failedToolCalls = toolCalls.filter((c) => !c.success);

      const totalInputTokens = tokenRecords.reduce(
        (sum, t) => sum + t.inputTokens,
        0
      );
      const totalOutputTokens = tokenRecords.reduce(
        (sum, t) => sum + t.outputTokens,
        0
      );
      const totalCost = tokenRecords.reduce((sum, t) => sum + t.cost, 0);

      const totalDurationMs = toolCalls.reduce(
        (sum, c) => sum + c.durationMs,
        0
      );

      const startTime = Math.min(
        ...[
          toolCalls[0]?.timestamp ?? Infinity,
          tokenRecords[0]?.timestamp ?? Infinity,
          agentRecords[0]?.startTime ?? Infinity,
        ].filter((t) => t !== Infinity)
      );

      const lastActivity = Math.max(
        ...[
          toolCalls[toolCalls.length - 1]?.timestamp ?? 0,
          tokenRecords[tokenRecords.length - 1]?.timestamp ?? 0,
          agentRecords[agentRecords.length - 1]?.endTime ?? 0,
        ].filter((t) => t !== 0)
      );

      return {
        sessionId,
        totalToolCalls: toolCalls.length,
        successfulToolCalls: successfulToolCalls.length,
        failedToolCalls: failedToolCalls.length,
        totalTokenUsage: {
          input: totalInputTokens,
          output: totalOutputTokens,
          cost: totalCost,
        },
        totalDurationMs,
        startTime: startTime === Infinity ? Date.now() : startTime,
        lastActivity,
      };
    },

    getToolMetrics(toolId, timeRange) {
      const records = filterByTimeRange(
        state.toolCalls.filter((c) => c.toolId === toolId),
        timeRange
      );

      if (records.length === 0) {
        return {
          toolId,
          callCount: 0,
          successCount: 0,
          failureCount: 0,
          avgDurationMs: 0,
          totalDurationMs: 0,
          lastCalled: 0,
          successRate: 0,
        };
      }

      const successCount = records.filter((c) => c.success).length;
      const totalDurationMs = records.reduce((sum, c) => sum + c.durationMs, 0);

      return {
        toolId,
        callCount: records.length,
        successCount,
        failureCount: records.length - successCount,
        avgDurationMs: Math.round(totalDurationMs / records.length),
        totalDurationMs,
        lastCalled: records[records.length - 1].timestamp,
        successRate: Math.round((successCount / records.length) * 100) / 100,
      };
    },

    getAllToolMetrics(timeRange) {
      const records = filterByTimeRange(state.toolCalls, timeRange);
      const toolIds = [...new Set(records.map((c) => c.toolId))];

      return toolIds.map((toolId) =>
        this.getToolMetrics(toolId, timeRange)
      );
    },

    getTokenUsage(timeRange) {
      const records = filterByTimeRange(state.tokenUsage, timeRange);

      const totalInputTokens = records.reduce(
        (sum, r) => sum + r.inputTokens,
        0
      );
      const totalOutputTokens = records.reduce(
        (sum, r) => sum + r.outputTokens,
        0
      );
      const totalCost = records.reduce((sum, r) => sum + r.cost, 0);

      const byModel: Record<string, { input: number; output: number; cost: number }> = {};
      for (const record of records) {
        if (!byModel[record.model]) {
          byModel[record.model] = { input: 0, output: 0, cost: 0 };
        }
        byModel[record.model].input += record.inputTokens;
        byModel[record.model].output += record.outputTokens;
        byModel[record.model].cost += record.cost;
      }

      const bySession: Record<string, { input: number; output: number; cost: number }> = {};
      for (const record of records) {
        if (!bySession[record.sessionId]) {
          bySession[record.sessionId] = { input: 0, output: 0, cost: 0 };
        }
        bySession[record.sessionId].input += record.inputTokens;
        bySession[record.sessionId].output += record.outputTokens;
        bySession[record.sessionId].cost += record.cost;
      }

      return {
        totalInputTokens,
        totalOutputTokens,
        totalCost,
        byModel,
        bySession,
      };
    },

    getAgentMetrics(timeRange) {
      return filterByTimeRange(state.agentMetrics, timeRange);
    },

    clear(timeRange) {
      if (!timeRange) {
        state.toolCalls = [];
        state.tokenUsage = [];
        state.agentMetrics = [];
        state.delegationMetrics = [];
        state.activeAgents.clear();
        return;
      }

      state.toolCalls = state.toolCalls.filter(
        (c) => c.timestamp !== undefined && c.timestamp < timeRange.start
      );
      state.tokenUsage = state.tokenUsage.filter(
        (t) => t.timestamp !== undefined && t.timestamp < timeRange.start
      );
      state.agentMetrics = state.agentMetrics.filter(
        (a) => a.startTime < timeRange.start
      );
      state.delegationMetrics = state.delegationMetrics.filter(
        (d) => d.timestamp < timeRange.start
      );
    },
  };
}

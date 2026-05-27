import type { ExecutionTrace, ToolCall, LLMCall, TraceStep } from "../services/trace_recorder";

export interface GetTraceOutput {
  trace: {
    sessionId: string;
    taskId: string;
    startTime: number;
    endTime: number;
    totalDurationMs: number;
    success: boolean;
    cost: number;
    complexity?: {
      total: number;
      breakdown: Record<string, number>;
    };
    operatorLevel?: string;
    steps?: TraceStep[];
    toolCalls?: ToolCall[];
    llmCalls?: LLMCall[];
  } | null;
  found: boolean;
}

export function createGetTraceTool(
  getTrace: (traceId: string) => ExecutionTrace | null,
  listTraces: (filters?: { sessionId?: string }) => string[]
) {
  return async (input: {
    sessionId: string;
    includeSteps?: boolean;
    includeToolCalls?: boolean;
    includeLLMCalls?: boolean;
  }): Promise<GetTraceOutput> => {
    const { sessionId, includeSteps = true, includeToolCalls = true, includeLLMCalls = true } = input;

    const traceIds = listTraces({ sessionId });

    if (traceIds.length === 0) {
      return { trace: null, found: false };
    }

    const mostRecentTraceId = traceIds[traceIds.length - 1];
    const trace = getTrace(mostRecentTraceId);

    if (!trace) {
      return { trace: null, found: false };
    }

    const response: GetTraceOutput["trace"] = {
      sessionId: trace.sessionId,
      taskId: trace.taskId,
      startTime: trace.startTime,
      endTime: trace.endTime,
      totalDurationMs: trace.totalDurationMs,
      success: trace.success,
      cost: trace.cost,
      complexity: trace.complexity ? {
        total: trace.complexity.total,
        breakdown: trace.complexity.breakdown,
      } : undefined,
      operatorLevel: trace.operatorLevel,
    };

    if (includeSteps) {
      response.steps = trace.steps;
    }

    if (includeToolCalls) {
      response.toolCalls = trace.toolCalls;
    }

    if (includeLLMCalls) {
      response.llmCalls = trace.llmCalls;
    }

    return { trace: response, found: true };
  };
}

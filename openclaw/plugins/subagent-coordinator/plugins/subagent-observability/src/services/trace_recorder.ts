import type {
  AfterExecutionEvent,
  TaskAnalyzedEvent,
  QualityGateEvent,
  ComplexityScore,
} from "@subagent-coordinator/types";

export interface TimeRange {
  start: number;
  end: number;
}

export interface TraceStep {
  stepId: string;
  timestamp: number;
  type: "analysis" | "execution" | "delegation" | "quality_check";
  durationMs: number;
  description?: string;
  result?: unknown;
}

export interface ToolCall {
  toolId: string;
  toolName: string;
  args: Record<string, unknown>;
  result?: unknown;
  durationMs: number;
  success: boolean;
  timestamp: number;
}

export interface LLMCall {
  model: string;
  promptTokens: number;
  completionTokens: number;
  cost: number;
  durationMs: number;
  timestamp: number;
}

export interface ExecutionTrace {
  traceId: string;
  sessionId: string;
  taskId: string;
  startTime: number;
  endTime: number;
  totalDurationMs: number;
  steps: TraceStep[];
  toolCalls: ToolCall[];
  llmCalls: LLMCall[];
  cost: number;
  success: boolean;
  complexity?: ComplexityScore;
  operatorLevel?: string;
}

export interface TraceRecorderState {
  traces: Map<string, ExecutionTrace>;
  currentTrace: ExecutionTrace | null;
  activeTraces: Map<string, ExecutionTrace>;
}

export function createInitialState(): TraceRecorderState {
  return {
    traces: new Map(),
    currentTrace: null,
    activeTraces: new Map(),
  };
}

export interface TraceRecorderService {
  startTrace(sessionId: string, taskId: string): string;
  endTrace(traceId: string, success: boolean): ExecutionTrace;

  recordStep(traceId: string, step: Omit<TraceStep, "stepId" | "timestamp">): void;
  recordToolCall(traceId: string, call: Omit<ToolCall, "timestamp">): void;
  recordLLMCall(traceId: string, call: Omit<LLMCall, "timestamp">): void;

  getTrace(traceId: string): ExecutionTrace | null;
  listTraces(filters?: { sessionId?: string; timeRange?: TimeRange; taskId?: string }): string[];
  getCurrentTrace(): ExecutionTrace | null;

  getTraceCost(traceId: string): number;

  clear(filters?: { olderThan?: number }): number;
}

export function createTraceRecorder(
  state: TraceRecorderState
): TraceRecorderService {
  const generateTraceId = (): string => {
    return `trace_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
  };

  const generateStepId = (): string => {
    return `step_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
  };

  return {
    startTrace(sessionId: string, taskId: string): string {
      const traceId = generateTraceId();
      const now = Date.now();

      const trace: ExecutionTrace = {
        traceId,
        sessionId,
        taskId,
        startTime: now,
        endTime: 0,
        totalDurationMs: 0,
        steps: [],
        toolCalls: [],
        llmCalls: [],
        cost: 0,
        success: false,
      };

      state.activeTraces.set(traceId, trace);
      state.currentTrace = trace;

      return traceId;
    },

    endTrace(traceId: string, success: boolean): ExecutionTrace {
      const trace = state.activeTraces.get(traceId);
      if (!trace) {
        throw new Error(`Trace ${traceId} not found`);
      }

      const now = Date.now();
      trace.endTime = now;
      trace.totalDurationMs = now - trace.startTime;
      trace.success = success;

      trace.cost = trace.llmCalls.reduce((sum, call) => sum + call.cost, 0);

      state.activeTraces.delete(traceId);
      state.traces.set(traceId, trace);
      state.currentTrace = null;

      return trace;
    },

    recordStep(traceId: string, step: Omit<TraceStep, "stepId" | "timestamp">): void {
      const trace = state.activeTraces.get(traceId);
      if (!trace) {
        console.warn(`[trace_recorder] Cannot record step: trace ${traceId} not found`);
        return;
      }

      const fullStep: TraceStep = {
        ...step,
        stepId: generateStepId(),
        timestamp: Date.now(),
      };

      trace.steps.push(fullStep);
    },

    recordToolCall(traceId: string, call: Omit<ToolCall, "timestamp">): void {
      const trace = state.activeTraces.get(traceId);
      if (!trace) {
        console.warn(`[trace_recorder] Cannot record tool call: trace ${traceId} not found`);
        return;
      }

      const fullCall: ToolCall = {
        ...call,
        timestamp: Date.now(),
      };

      trace.toolCalls.push(fullCall);
    },

    recordLLMCall(traceId: string, call: Omit<LLMCall, "timestamp">): void {
      const trace = state.activeTraces.get(traceId);
      if (!trace) {
        console.warn(`[trace_recorder] Cannot record LLM call: trace ${traceId} not found`);
        return;
      }

      const fullCall: LLMCall = {
        ...call,
        timestamp: Date.now(),
      };

      trace.llmCalls.push(fullCall);
    },

    getTrace(traceId: string): ExecutionTrace | null {
      const active = state.activeTraces.get(traceId);
      if (active) return active;

      return state.traces.get(traceId) || null;
    },

    listTraces(filters?: { sessionId?: string; timeRange?: TimeRange; taskId?: string }): string[] {
      const allTraces = [
        ...Array.from(state.activeTraces.entries()),
        ...Array.from(state.traces.entries()),
      ];

      return allTraces
        .filter(([, trace]) => {
          if (filters?.sessionId && trace.sessionId !== filters.sessionId) return false;
          if (filters?.taskId && trace.taskId !== filters.taskId) return false;
          if (filters?.timeRange) {
            if (trace.startTime < filters.timeRange.start) return false;
            if (trace.endTime > 0 && trace.endTime > filters.timeRange.end) return false;
          }
          return true;
        })
        .map(([id]) => id);
    },

    getCurrentTrace(): ExecutionTrace | null {
      return state.currentTrace;
    },

    getTraceCost(traceId: string): number {
      const trace = this.getTrace(traceId);
      if (!trace) return 0;
      return trace.llmCalls.reduce((sum, call) => sum + call.cost, 0);
    },

    clear(filters?: { olderThan?: number }): number {
      let count = 0;
      const cutoff = filters?.olderThan ?? Date.now() - 7 * 24 * 60 * 60 * 1000;

      for (const [id, trace] of state.traces.entries()) {
        if (trace.endTime > 0 && trace.endTime < cutoff) {
          state.traces.delete(id);
          count++;
        }
      }

      for (const [id, trace] of state.activeTraces.entries()) {
        if (trace.startTime < cutoff) {
          state.activeTraces.delete(id);
          count++;
        }
      }

      return count;
    },
  };
}

export function createTraceFromAfterExecution(
  event: AfterExecutionEvent,
  traceRecorder: TraceRecorderService
): string {
  const { task, result, complexity, operatorLevel } = event;

  const traceId = traceRecorder.startTrace(task.id, task.id);

  traceRecorder.recordStep(traceId, {
    type: "execution",
    durationMs: result.duration,
    description: `Task execution: ${task.description.substring(0, 100)}`,
    result: result.output,
  });

  const trace = traceRecorder.getTrace(traceId);
  if (trace) {
    trace.complexity = complexity;
    trace.operatorLevel = operatorLevel;
  }

  return traceId;
}

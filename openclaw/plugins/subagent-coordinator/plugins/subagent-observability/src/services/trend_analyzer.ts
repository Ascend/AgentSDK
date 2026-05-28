import type { ExecutionTrace } from "./trace_recorder";
import type { CostBreakdown } from "./cost_tracker";

export interface TimeRange {
  start: number;
  end: number;
}

export interface TrendDataPoint {
  timestamp: number;
  value: number;
  label?: string;
}

export interface TrendAnalysis {
  period: TimeRange;
  durationTrend: TrendDataPoint[];
  costTrend: TrendDataPoint[];
  successRateTrend: TrendDataPoint[];
  stepCountTrend: TrendDataPoint[];
  tokenUsageTrend: TrendDataPoint[];

  summary: {
    avgDurationMs: number;
    avgCost: number;
    avgSuccessRate: number;
    avgStepCount: number;
    totalTraces: number;
    totalCost: number;
  };

  insights: string[];
  recommendations: string[];
}

export interface CacheAnalysis {
  cacheHitRate: number;
  cacheMissCount: number;
  cacheHitCount: number;
  estimatedSavings: number;
  byTool: Record<string, { hits: number; misses: number; hitRate: number }>;
}

export interface EfficiencyScore {
  score: number;
  durationScore: number;
  costScore: number;
  successScore: number;
  stepEfficiency: number;
  breakdown: {
    durationMs: number;
    cost: number;
    steps: number;
    success: boolean;
  };
}

export interface TraceComparison {
  traceA: {
    traceId: string;
    durationMs: number;
    cost: number;
    steps: number;
    success: boolean;
    efficiency: number;
  };
  traceB: {
    traceId: string;
    durationMs: number;
    cost: number;
    steps: number;
    success: boolean;
    efficiency: number;
  };
  durationDiff: number;
  costDiff: number;
  stepCountDiff: number;
  efficiencyScore: {
    a: number;
    b: number;
    winner: "a" | "b" | "tie";
  };
  recommendations: string[];
}

export interface TrendAnalyzerState {
  historicalTraces: ExecutionTrace[];
  cacheHits: Map<string, number>;
  cacheMisses: Map<string, number>;
}

export function createInitialState(): TrendAnalyzerState {
  return {
    historicalTraces: [],
    cacheHits: new Map(),
    cacheMisses: new Map(),
  };
}

export interface TrendAnalyzerService {
  recordTrace(trace: ExecutionTrace): void;
  getTrendAnalysis(days?: number): TrendAnalysis;
  get7DayTrend(): TrendAnalysis;
  recordCacheHit(toolId: string): void;
  recordCacheMiss(toolId: string): void;
  getCacheAnalysis(): CacheAnalysis;
  calculateEfficiency(trace: ExecutionTrace): EfficiencyScore;
  compareTraces(traceIdA: string, traceIdB: string): TraceComparison;
  clear(olderThan?: number): number;
}

export function createTrendAnalyzer(
  state: TrendAnalyzerState
): TrendAnalyzerService {
  const get7DaysAgo = (): number => {
    return Date.now() - 7 * 24 * 60 * 60 * 1000;
  };

  const calculateEfficiency = (trace: ExecutionTrace): number => {
    if (trace.steps.length === 0) return 50;

    const durationFactor = Math.max(0, 1 - trace.totalDurationMs / 60000) * 30;
    const costFactor = Math.max(0, 1 - trace.cost / 0.5) * 30;
    const stepFactor = Math.max(0, 1 - trace.steps.length / 10) * 20;
    const successBonus = trace.success ? 20 : 0;

    return Math.min(100, Math.round(durationFactor + costFactor + stepFactor + successBonus));
  };

  return {
    recordTrace(trace: ExecutionTrace): void {
      state.historicalTraces.push(trace);

      if (state.historicalTraces.length > 1000) {
        state.historicalTraces = state.historicalTraces.slice(-500);
      }
    },

    getTrendAnalysis(days: number = 7): TrendAnalysis {
      const now = Date.now();
      const startTime = now - days * 24 * 60 * 60 * 1000;
      const timeRange: TimeRange = { start: startTime, end: now };

      const traces = state.historicalTraces.filter(
        (t) => t.startTime >= startTime && t.startTime <= now
      );

      const byDay = new Map<string, ExecutionTrace[]>();
      for (const trace of traces) {
        const dayKey = new Date(trace.startTime).toISOString().substring(0, 10);
        if (!byDay.has(dayKey)) {
          byDay.set(dayKey, []);
        }
        byDay.get(dayKey)!.push(trace);
      }

      const durationTrend: TrendDataPoint[] = [];
      const costTrend: TrendDataPoint[] = [];
      const successRateTrend: TrendDataPoint[] = [];
      const stepCountTrend: TrendDataPoint[] = [];
      const tokenUsageTrend: TrendDataPoint[] = [];

      const sortedDays = Array.from(byDay.keys()).sort();
      for (const day of sortedDays) {
        const dayTraces = byDay.get(day)!;
        const timestamp = new Date(day).getTime();

        const avgDuration = dayTraces.reduce((sum, t) => sum + t.totalDurationMs, 0) / dayTraces.length;
        durationTrend.push({ timestamp, value: Math.round(avgDuration), label: day });

        const totalCost = dayTraces.reduce((sum, t) => sum + t.cost, 0);
        costTrend.push({ timestamp, value: Math.round(totalCost * 100) / 100, label: day });

        const successCount = dayTraces.filter((t) => t.success).length;
        successRateTrend.push({
          timestamp,
          value: Math.round((successCount / dayTraces.length) * 100),
          label: day,
        });

        const avgSteps = dayTraces.reduce((sum, t) => sum + t.steps.length, 0) / dayTraces.length;
        stepCountTrend.push({ timestamp, value: Math.round(avgSteps * 10) / 10, label: day });

        const totalTokens = dayTraces.reduce((sum, t) => {
          return sum + t.cost * 100000;
        }, 0);
        tokenUsageTrend.push({ timestamp, value: Math.round(totalTokens), label: day });
      }

      const totalDuration = traces.reduce((sum, t) => sum + t.totalDurationMs, 0);
      const totalCost = traces.reduce((sum, t) => sum + t.cost, 0);
      const successCount = traces.filter((t) => t.success).length;
      const totalSteps = traces.reduce((sum, t) => sum + t.steps.length, 0);

      const summary = {
        avgDurationMs: traces.length > 0 ? Math.round(totalDuration / traces.length) : 0,
        avgCost: traces.length > 0 ? Math.round((totalCost / traces.length) * 100) / 100 : 0,
        avgSuccessRate: traces.length > 0 ? Math.round((successCount / traces.length) * 100) : 0,
        avgStepCount: traces.length > 0 ? Math.round((totalSteps / traces.length) * 10) / 10 : 0,
        totalTraces: traces.length,
        totalCost: Math.round(totalCost * 100) / 100,
      };

      const insights: string[] = [];
      const recommendations: string[] = [];

      if (traces.length > 0) {
        if (summary.avgDurationMs > 60000) {
          insights.push("Average execution time is over 1 minute");
          recommendations.push("Consider breaking down complex tasks into smaller subtasks");
        }

        if (summary.avgCost > 1) {
          insights.push("Average cost per trace is over $1");
          recommendations.push("Review high-cost LLM calls and optimize prompt lengths");
        }

        if (summary.avgSuccessRate < 80) {
          insights.push("Success rate is below 80%");
          recommendations.push("Review failed traces to identify common failure patterns");
        }

        if (summary.avgStepCount > 10) {
          insights.push("Average step count is high (>10 steps)");
          recommendations.push("Consider task decomposition to reduce complexity");
        }
      }

      return {
        period: timeRange,
        durationTrend,
        costTrend,
        successRateTrend,
        stepCountTrend,
        tokenUsageTrend,
        summary,
        insights,
        recommendations,
      };
    },

    get7DayTrend(): TrendAnalysis {
      return this.getTrendAnalysis(7);
    },

    recordCacheHit(toolId: string): void {
      const current = state.cacheHits.get(toolId) || 0;
      state.cacheHits.set(toolId, current + 1);
    },

    recordCacheMiss(toolId: string): void {
      const current = state.cacheMisses.get(toolId) || 0;
      state.cacheMisses.set(toolId, current + 1);
    },

    getCacheAnalysis(): CacheAnalysis {
      const allTools = new Set([
        ...Array.from(state.cacheHits.keys()),
        ...Array.from(state.cacheMisses.keys()),
      ]);

      let totalHits = 0;
      let totalMisses = 0;
      const byTool: CacheAnalysis["byTool"] = {};

      for (const toolId of allTools) {
        const hits = state.cacheHits.get(toolId) || 0;
        const misses = state.cacheMisses.get(toolId) || 0;
        const total = hits + misses;

        totalHits += hits;
        totalMisses += misses;

        byTool[toolId] = {
          hits,
          misses,
          hitRate: total > 0 ? Math.round((hits / total) * 100) / 100 : 0,
        };
      }

      const totalRequests = totalHits + totalMisses;
      const cacheHitRate = totalRequests > 0 ? totalHits / totalRequests : 0;

      const estimatedSavings = totalHits * 0.001;

      return {
        cacheHitRate: Math.round(cacheHitRate * 100) / 100,
        cacheMissCount: totalMisses,
        cacheHitCount: totalHits,
        estimatedSavings: Math.round(estimatedSavings * 100) / 100,
        byTool,
      };
    },

    calculateEfficiency(trace: ExecutionTrace): EfficiencyScore {
      const durationScore = Math.max(0, Math.min(100, 100 - trace.totalDurationMs / 1000));
      const costScore = Math.max(0, Math.min(100, 100 - trace.cost * 100));
      const successScore = trace.success ? 100 : 0;
      const stepEfficiency = Math.max(0, Math.min(100, 100 - trace.steps.length * 5));

      const score = Math.round(
        (durationScore * 0.3 + costScore * 0.3 + successScore * 0.25 + stepEfficiency * 0.15)
      );

      return {
        score,
        durationScore: Math.round(durationScore),
        costScore: Math.round(costScore),
        successScore,
        stepEfficiency: Math.round(stepEfficiency),
        breakdown: {
          durationMs: trace.totalDurationMs,
          cost: trace.cost,
          steps: trace.steps.length,
          success: trace.success,
        },
      };
    },

    compareTraces(traceIdA: string, traceIdB: string): TraceComparison {
      const nullTrace = {
        traceId: "unknown",
        durationMs: 0,
        cost: 0,
        steps: 0,
        success: false,
        efficiency: 0,
      };

      const traceA = nullTrace;
      const traceB = nullTrace;

      const durationDiff = traceA.durationMs > 0
        ? Math.round(((traceA.durationMs - traceB.durationMs) / traceA.durationMs) * 100)
        : 0;
      const costDiff = traceA.cost > 0
        ? Math.round(((traceA.cost - traceB.cost) / traceA.cost) * 100)
        : 0;
      const stepCountDiff = traceA.steps - traceB.steps;

      const winner = traceA.efficiency > traceB.efficiency ? "a" :
                     traceB.efficiency > traceA.efficiency ? "b" : "tie";

      const recommendations: string[] = [];
      if (durationDiff > 20) {
        recommendations.push("Trace A is significantly slower. Consider optimizing the execution flow.");
      }
      if (costDiff > 20) {
        recommendations.push("Trace A is more expensive. Review token usage and prompt efficiency.");
      }

      return {
        traceA,
        traceB,
        durationDiff,
        costDiff,
        stepCountDiff,
        efficiencyScore: {
          a: traceA.efficiency,
          b: traceB.efficiency,
          winner,
        },
        recommendations,
      };
    },

    clear(olderThan?: number): number {
      const cutoff = olderThan ?? get7DaysAgo();
      const initialCount = state.historicalTraces.length;

      state.historicalTraces = state.historicalTraces.filter(
        (t) => t.startTime >= cutoff
      );

      return initialCount - state.historicalTraces.length;
    },
  };
}

import type { ExecutionTrace } from "../services/trace_recorder";
import type { TrendAnalyzerService } from "../services/trend_analyzer";

export interface CompareTracesOutput {
  comparison: {
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
  };
  foundA: boolean;
  foundB: boolean;
}

export function createCompareTracesTool(
  getTrace: (traceId: string) => ExecutionTrace | null,
  trendAnalyzer: TrendAnalyzerService
) {
  return async (input: { traceIdA: string; traceIdB: string }): Promise<CompareTracesOutput> => {
    const { traceIdA, traceIdB } = input;

    const traceA = getTrace(traceIdA);
    const traceB = getTrace(traceIdB);

    const foundA = traceA !== null;
    const foundB = traceB !== null;

    const buildTraceInfo = (trace: ExecutionTrace | null, traceId: string) => {
      if (!trace) {
        return {
          traceId,
          durationMs: 0,
          cost: 0,
          steps: 0,
          success: false,
          efficiency: 0,
        };
      }
      return {
        traceId: trace.traceId,
        durationMs: trace.totalDurationMs,
        cost: trace.cost,
        steps: trace.steps.length,
        success: trace.success,
        efficiency: trendAnalyzer.calculateEfficiency(trace).score,
      };
    };

    const infoA = buildTraceInfo(traceA, traceIdA);
    const infoB = buildTraceInfo(traceB, traceIdB);

    const durationDiff = infoA.durationMs > 0 && infoB.durationMs > 0
      ? Math.round(((infoA.durationMs - infoB.durationMs) / infoB.durationMs) * 100)
      : 0;

    const costDiff = infoA.cost > 0 && infoB.cost > 0
      ? Math.round(((infoA.cost - infoB.cost) / infoB.cost) * 100)
      : 0;

    const stepCountDiff = infoA.steps - infoB.steps;

    const winner = infoA.efficiency > infoB.efficiency ? "a" :
                   infoB.efficiency > infoA.efficiency ? "b" : "tie";

    const recommendations: string[] = [];

    if (foundA && foundB) {
      if (durationDiff > 30) {
        recommendations.push(`Trace A is ${durationDiff}% slower than Trace B. Consider optimizing the execution flow.`);
      } else if (durationDiff < -30) {
        recommendations.push(`Trace B is ${Math.abs(durationDiff)}% slower than Trace A. Consider adopting Trace A's approach.`);
      }

      if (costDiff > 30) {
        recommendations.push(`Trace A costs ${costDiff}% more than Trace B. Review token usage in Trace A.`);
      } else if (costDiff < -30) {
        recommendations.push(`Trace B costs ${Math.abs(costDiff)}% more than Trace A. Consider Trace A's approach.`);
      }

      if (stepCountDiff > 5) {
        recommendations.push(`Trace A has ${stepCountDiff} more steps than Trace B. Consider more granular decomposition.`);
      } else if (stepCountDiff < -5) {
        recommendations.push(`Trace B has ${Math.abs(stepCountDiff)} more steps than Trace A. Consider Trace A's approach.`);
      }

      if (recommendations.length === 0) {
        recommendations.push("Both traces are relatively similar in performance.");
      }
    } else if (!foundA || !foundB) {
      recommendations.push(`One or both traces not found. A: ${foundA}, B: ${foundB}`);
    }

    return {
      comparison: {
        traceA: infoA,
        traceB: infoB,
        durationDiff,
        costDiff,
        stepCountDiff,
        efficiencyScore: {
          a: infoA.efficiency,
          b: infoB.efficiency,
          winner,
        },
        recommendations,
      },
      foundA,
      foundB,
    };
  };
}

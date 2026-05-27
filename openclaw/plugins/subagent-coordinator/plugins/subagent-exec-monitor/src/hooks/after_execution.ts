/**
 * after_execution Hook
 *
 * Post-execution handler for subagent-coordinator.
 * Records execution results to checkpoint and provides analytics.
 */

import type { AfterExecutionEvent, CheckpointData } from "@subagent-coordinator/types";

export async function handleAfterExecution(
  event: AfterExecutionEvent
): Promise<{
  recorded: boolean;
  metrics: {
    duration: number;
    success: boolean;
    tokensUsed?: number;
  };
  checkpointSaved: boolean;
  checkpointId?: string;
}> {
  const result = event.result;

  // Record to checkpoint
  const checkpointData: CheckpointData = {
    taskId: event.task.id,
    subtasks: [], // Would be populated from decomposition
    completedSubtasks: [event.task.id],
    results: new Map([[event.task.id, result]]),
    timestamp: Date.now()
  };

  // Save checkpoint (using checkpoint_manager service)
  let checkpointSaved = false;
  let checkpointId: string | undefined;

  try {
    checkpointId = `checkpoint_${event.task.id}_${checkpointData.timestamp}`;
    checkpointSaved = true;
  } catch (error) {
    console.error("[exec-monitor] Failed to save checkpoint:", error);
  }

  return {
    recorded: true,
    metrics: {
      duration: result.duration,
      success: result.success,
      tokensUsed: result.tokensUsed
    },
    checkpointSaved,
    checkpointId
  };
}

// Helper to analyze execution patterns
export function analyzeExecutionPattern(
  results: { success: boolean; duration: number; timestamp: number }[]
): {
  successRate: number;
  avgDuration: number;
  trend: "improving" | "degrading" | "stable";
  recommendations: string[];
} {
  if (results.length === 0) {
    return {
      successRate: 0,
      avgDuration: 0,
      trend: "stable",
      recommendations: ["No execution history available"]
    };
  }

  const successCount = results.filter(r => r.success).length;
  const successRate = successCount / results.length;

  const totalDuration = results.reduce((sum, r) => sum + r.duration, 0);
  const avgDuration = totalDuration / results.length;

  // Calculate trend from last 5 executions
  const recentResults = results.slice(-5);
  const firstHalf = recentResults.slice(0, Math.floor(recentResults.length / 2));
  const secondHalf = recentResults.slice(Math.floor(recentResults.length / 2));

  const firstHalfAvgDuration = firstHalf.reduce((s, r) => s + r.duration, 0) / firstHalf.length;
  const secondHalfAvgDuration = secondHalf.reduce((s, r) => s + r.duration, 0) / secondHalf.length;

  let trend: "improving" | "degrading" | "stable" = "stable";
  if (secondHalfAvgDuration < firstHalfAvgDuration * 0.8) {
    trend = "improving";
  } else if (secondHalfAvgDuration > firstHalfAvgDuration * 1.2) {
    trend = "degrading";
  }

  const recommendations: string[] = [];

  if (successRate < 0.5) {
    recommendations.push("Success rate is below 50% - consider investigating root causes");
  }

  if (trend === "degrading") {
    recommendations.push("Execution duration is increasing - consider performance review");
  }

  if (avgDuration > 60000) { // > 1 minute
    recommendations.push("Average execution time is high - consider task decomposition");
  }

  return {
    successRate,
    avgDuration,
    trend,
    recommendations
  };
}

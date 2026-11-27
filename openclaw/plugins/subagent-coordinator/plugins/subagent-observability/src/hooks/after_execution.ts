/*!
 * This file is part of the AgentSDK project.
 * Copyright (c) 2026 Huawei Technologies Co.,Ltd.
 *
 * AgentSDK is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *          http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

import type {
  AfterExecutionEvent,
  HookResult,
} from "@subagent-coordinator/types";

import type { MetricsCollectorService } from "../services/metrics_collector";
import type { TraceRecorderService } from "../services/trace_recorder";
import type { CostTrackerService } from "../services/cost_tracker";
import type { TrendAnalyzerService } from "../services/trend_analyzer";

export function handleAfterExecution(
  event: AfterExecutionEvent,
  metrics: MetricsCollectorService,
  traceRecorder: TraceRecorderService,
  costTracker: CostTrackerService,
  trendAnalyzer: TrendAnalyzerService
): HookResult {
  const { task, result, complexity, operatorLevel } = event;

  metrics.recordAgentEnd({
    sessionId: task.id,
    agentId: operatorLevel,
    taskId: task.id,
    success: result.success,
    error: result.error,
  });

  const INPUT_COST_PER_M = 2.0;
  const OUTPUT_COST_PER_M = 8.0;
  const estimatedTokens = result.tokensUsed || 0;
  const inputTokens = Math.round(estimatedTokens * 0.7);
  const outputTokens = estimatedTokens - inputTokens;
  const cost = (inputTokens / 1000000) * INPUT_COST_PER_M + (outputTokens / 1000000) * OUTPUT_COST_PER_M;

  metrics.recordTokenUsage({
    sessionId: task.id,
    model: "default",
    inputTokens,
    outputTokens,
    cost,
  });

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

  const endedTrace = traceRecorder.endTrace(traceId, result.success);

  trendAnalyzer.recordTrace(endedTrace);

  const budgetAlert = costTracker.checkBudgetAlert();

  return {
    recorded: true,
    metrics: {
      cost,
      tokensUsed: estimatedTokens,
      budgetAlert: budgetAlert.shouldAlert ? budgetAlert.message : undefined,
    },
  };
}

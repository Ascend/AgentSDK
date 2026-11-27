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
  TaskAnalyzedEvent,
  ComplexityScore,
  HookResult,
} from "@subagent-coordinator/types";

import type { MetricsCollectorService } from "../services/metrics_collector";

export function handleTaskAnalyzed(
  event: TaskAnalyzedEvent,
  metrics: MetricsCollectorService
): HookResult {
  const { task, complexity } = event;

  const recentMetrics = metrics.getAgentMetrics();
  const similarTasks = recentMetrics.filter(
    (r) => r.taskId.includes(task.id.substring(0, 8))
  );

  let enhancedScore: ComplexityScore = { ...complexity };

  if (similarTasks.length > 5) {
    const failedTasks = similarTasks.filter((r) => !r.success);
    if (failedTasks.length / similarTasks.length > 0.3) {
      enhancedScore = {
        ...enhancedScore,
        total: Math.min(10, enhancedScore.total + 1),
      };
    }
  }

  const desc = task.description.toLowerCase();
  const crossSystemKeywords = ["integrate", "migrate", "sync", "connect", "bridge"];
  const securityKeywords = ["auth", "security", "encrypt", "credential", "permission"];
  const dataTransformKeywords = ["transform", "convert", "parse", "serialize", "map"];

  if (crossSystemKeywords.some(kw => desc.includes(kw))) {
    enhancedScore = { ...enhancedScore, total: Math.min(10, enhancedScore.total + 1) };
  }
  if (securityKeywords.some(kw => desc.includes(kw))) {
    enhancedScore = { ...enhancedScore, total: Math.min(10, enhancedScore.total + 1) };
  }
  if (dataTransformKeywords.some(kw => desc.includes(kw))) {
    enhancedScore = { ...enhancedScore, total: Math.min(10, enhancedScore.total + 1) };
  }

  return {
    recorded: true,
    enhanced: enhancedScore.total !== complexity.total,
    originalScore: complexity,
    enhancedScore,
  };
}

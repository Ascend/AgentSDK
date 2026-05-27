import type {
  BeforeDelegationEvent,
  HookResult,
  RoutingSuggestion,
} from "@subagent-coordinator/types";

import type { MetricsCollectorService } from "../services/metrics_collector";
import type { RateLimiterService } from "../services/rate_limiter";

export function handleBeforeDelegation(
  event: BeforeDelegationEvent,
  metrics: MetricsCollectorService,
  rateLimiter: RateLimiterService
): HookResult {
  const { task, complexity, operatorLevel, routingDecision } = event;

  metrics.recordDelegation({
    taskId: task.id,
    complexity,
    operatorLevel,
    runtime: routingDecision.runtime,
    agentId: routingDecision.agentId,
    timestamp: Date.now(),
  });

  const rateLimitResult = rateLimiter.check(routingDecision.agentId);

  let enhancedRoutingSuggestion: RoutingSuggestion | undefined;

  if (!rateLimitResult.allowed) {
    enhancedRoutingSuggestion = {
      runtime: routingDecision.runtime === "subagent" ? "acp" : "subagent",
      agentId: "alternative-agent",
      reason: `Rate limit exceeded for ${routingDecision.agentId}. Suggesting alternative runtime.`,
      priority: "high",
    };
  }

  const recentDelegations = metrics.getAgentMetrics();
  const similarTasks = recentDelegations.filter(
    (r) => r.agentId === routingDecision.agentId
  );
  const failedSimilarTasks = similarTasks.filter((r) => !r.success);

  if (similarTasks.length > 3 && failedSimilarTasks.length / similarTasks.length > 0.3) {
    enhancedRoutingSuggestion = enhancedRoutingSuggestion || {
      runtime: routingDecision.runtime,
      agentId: routingDecision.agentId,
      reason: `High failure rate (${Math.round((failedSimilarTasks.length / similarTasks.length) * 100)}%) for similar tasks with this agent`,
      priority: "normal",
    };
  }

  return {
    recorded: true,
    block: !rateLimitResult.allowed,
    enhancedRoutingSuggestion,
    metrics: {
      rateLimit: rateLimitResult,
    },
  };
}

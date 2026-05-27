import type {
  RouteDecisionEvent,
  HookResult,
  RuntimeType,
} from "@subagent-coordinator/types";

import type { RateLimiterService } from "../services/rate_limiter";
import type { MetricsCollectorService } from "../services/metrics_collector";

export function handleRouteDecision(
  event: RouteDecisionEvent,
  rateLimiter: RateLimiterService,
  metrics: MetricsCollectorService
): HookResult {
  const { task, proposedRuntime, proposedAgentId } = event;

  const rateLimitResult = rateLimiter.check(proposedAgentId);

  const agentMetrics = metrics.getAgentMetrics();
  const tasksForAgent = agentMetrics.filter((r) => r.agentId === proposedAgentId);
  const successfulTasks = tasksForAgent.filter((r) => r.success);
  const successRate = tasksForAgent.length > 0 ? successfulTasks.length / tasksForAgent.length : 1;

  let suggestedRuntime: RuntimeType | undefined;
  let suggestedAgentId: string | undefined;
  let reason: string | undefined;

  if (successRate < 0.6 && tasksForAgent.length >= 3) {
    suggestedRuntime = proposedRuntime === "subagent" ? "acp" : "subagent";
    suggestedAgentId = "alternative-agent";
    reason = `Low success rate (${Math.round(successRate * 100)}%) for agent ${proposedAgentId}. Suggesting alternative.`;
  }

  if (!rateLimitResult.allowed) {
    suggestedRuntime = proposedRuntime === "subagent" ? "acp" : "subagent";
    suggestedAgentId = "alternative-agent";
    reason = `Rate limit exceeded for ${proposedAgentId}. Suggesting alternative.`;
  }

  return {
    recorded: true,
    runtime: suggestedRuntime || proposedRuntime,
    agentId: suggestedAgentId || proposedAgentId,
    reason,
    metrics: {
      rateLimit: rateLimitResult,
      successRate,
    },
  };
}

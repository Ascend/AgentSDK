import type { RateLimiterService, RateLimiterState } from "../services/rate_limiter";

export function createGetRateLimitStatusTool(rateLimiter: RateLimiterService) {
  return async (input: { agentId?: string }): Promise<{
    rateLimitedAgents: string[];
    agentStatus?: {
      allowed: boolean;
      currentRate: number;
      maxRate: number;
      retryAfterMs?: number;
    };
  }> => {
    const { agentId } = input;

    const rateLimitedAgents = rateLimiter.getRateLimitedAgents();

    let agentStatus;
    if (agentId) {
      const result = rateLimiter.check(agentId);
      agentStatus = {
        allowed: result.allowed,
        currentRate: result.currentRate,
        maxRate: result.maxRate,
        retryAfterMs: result.retryAfterMs,
      };
    }

    return {
      rateLimitedAgents,
      agentStatus,
    };
  };
}

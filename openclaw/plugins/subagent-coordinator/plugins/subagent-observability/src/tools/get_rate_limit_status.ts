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

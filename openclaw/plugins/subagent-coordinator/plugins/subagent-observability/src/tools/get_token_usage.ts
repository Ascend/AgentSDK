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

import type { TokenUsage } from "../services/metrics_collector";

export interface GetTokenUsageInput {
  sessionId?: string;
  timeRange?: {
    start: number;
    end: number;
  };
}

export interface TokenUsageOutput {
  usage: TokenUsage;
  period: { start: number; end: number };
  sessionFiltered: boolean;
}

export function createGetTokenUsageTool(
  getTokenUsage: (timeRange?: { start: number; end: number }) => TokenUsage,
  getSessionMetrics: (sessionId: string) => { totalTokenUsage: { input: number; output: number; cost: number } } | null
) {
  return async (input: GetTokenUsageInput): Promise<TokenUsageOutput> => {
    const { sessionId, timeRange } = input;

    if (sessionId) {
      const sessionMetrics = getSessionMetrics(sessionId);

      const period = timeRange || {
        start: Date.now() - 24 * 60 * 60 * 1000,
        end: Date.now(),
      };

      if (sessionMetrics) {
        return {
          usage: {
            totalInputTokens: sessionMetrics.totalTokenUsage.input,
            totalOutputTokens: sessionMetrics.totalTokenUsage.output,
            totalCost: sessionMetrics.totalTokenUsage.cost,
            byModel: {},
            bySession: {
              [sessionId]: {
                input: sessionMetrics.totalTokenUsage.input,
                output: sessionMetrics.totalTokenUsage.output,
                cost: sessionMetrics.totalTokenUsage.cost,
              },
            },
          },
          period,
          sessionFiltered: true,
        };
      }

      return {
        usage: {
          totalInputTokens: 0,
          totalOutputTokens: 0,
          totalCost: 0,
          byModel: {},
          bySession: {},
        },
        period,
        sessionFiltered: true,
      };
    }

    const period = timeRange || {
      start: Date.now() - 24 * 60 * 60 * 1000,
      end: Date.now(),
    };

    const usage = getTokenUsage(timeRange);

    return {
      usage,
      period,
      sessionFiltered: false,
    };
  };
}

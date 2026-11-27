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

import type { CostBreakdown, BudgetStatus } from "../services/cost_tracker";

export interface GetCostBreakdownOutput {
  breakdown: CostBreakdown;
  budget?: {
    daily: {
      limit: number;
      used: number;
      remaining: number;
      remainingPercent: number;
      isAlerted: boolean;
    };
    monthly: {
      limit: number;
      used: number;
      remaining: number;
      remainingPercent: number;
      isAlerted: boolean;
    };
  };
  filtered: boolean;
  period?: {
    start: number;
    end: number;
  };
}

export function createGetCostBreakdownTool(
  getCostBreakdown: (
    sessionId?: string,
    taskId?: string,
    timeRange?: { start: number; end: number },
    groupBy?: "step" | "tool" | "model" | "time"
  ) => CostBreakdown,
  getBudgetStatus: () => BudgetStatus
) {
  return async (input: {
    sessionId?: string;
    taskId?: string;
    timeRange?: { start: number; end: number };
    groupBy?: "step" | "tool" | "model" | "time";
  }): Promise<GetCostBreakdownOutput> => {
    const { sessionId, taskId, timeRange, groupBy = "model" } = input;

    const breakdown = getCostBreakdown(sessionId, taskId, timeRange, groupBy);
    const budgetStatus = getBudgetStatus();

    return {
      breakdown,
      budget: {
        daily: {
          limit: budgetStatus.daily.limit,
          used: budgetStatus.daily.used,
          remaining: budgetStatus.daily.remaining,
          remainingPercent: budgetStatus.daily.remainingPercent,
          isAlerted: budgetStatus.daily.isAlerted,
        },
        monthly: {
          limit: budgetStatus.monthly.limit,
          used: budgetStatus.monthly.used,
          remaining: budgetStatus.monthly.remaining,
          remainingPercent: budgetStatus.monthly.remainingPercent,
          isAlerted: budgetStatus.monthly.isAlerted,
        },
      },
      filtered: !!(sessionId || taskId || timeRange),
      period: timeRange,
    };
  };
}

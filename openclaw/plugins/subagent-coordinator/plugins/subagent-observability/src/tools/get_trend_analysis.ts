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

import type { TrendAnalysis } from "../services/trend_analyzer";

export interface GetTrendAnalysisOutput {
  trend: {
    period: { start: number; end: number };
    durationTrend: Array<{ timestamp: number; value: number; label?: string }>;
    costTrend: Array<{ timestamp: number; value: number; label?: string }>;
    successRateTrend: Array<{ timestamp: number; value: number; label?: string }>;
    stepCountTrend: Array<{ timestamp: number; value: number; label?: string }>;
    tokenUsageTrend: Array<{ timestamp: number; value: number; label?: string }>;
    summary: {
      avgDurationMs: number;
      avgCost: number;
      avgSuccessRate: number;
      avgStepCount: number;
      totalTraces: number;
      totalCost: number;
    };
    insights: string[];
    recommendations: string[];
  };
  cacheAnalysis?: {
    cacheHitRate: number;
    cacheMissCount: number;
    cacheHitCount: number;
    estimatedSavings: number;
    byTool: Record<string, { hits: number; misses: number; hitRate: number }>;
  };
}

export function createGetTrendAnalysisTool(
  getTrendAnalysis: (days?: number) => TrendAnalysis,
  getCacheAnalysis: () => {
    cacheHitRate: number;
    cacheMissCount: number;
    cacheHitCount: number;
    estimatedSavings: number;
    byTool: Record<string, { hits: number; misses: number; hitRate: number }>;
  }
) {
  return async (input?: { days?: number }): Promise<GetTrendAnalysisOutput> => {
    const days = input?.days ?? 7;
    const trend = getTrendAnalysis(days);
    const cacheAnalysis = getCacheAnalysis();

    return {
      trend: {
        period: trend.period,
        durationTrend: trend.durationTrend,
        costTrend: trend.costTrend,
        successRateTrend: trend.successRateTrend,
        stepCountTrend: trend.stepCountTrend,
        tokenUsageTrend: trend.tokenUsageTrend,
        summary: trend.summary,
        insights: trend.insights,
        recommendations: trend.recommendations,
      },
      cacheAnalysis,
    };
  };
}

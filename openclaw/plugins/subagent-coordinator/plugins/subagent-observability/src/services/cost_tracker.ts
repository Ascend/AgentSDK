import type { LLMCall } from "./trace_recorder";

export interface TimeRange {
  start: number;
  end: number;
}

export interface CostRecord {
  id: string;
  traceId?: string;
  sessionId: string;
  taskId: string;
  model: string;
  cost: number;
  inputTokens: number;
  outputTokens: number;
  timestamp: number;
}

export interface BudgetConfig {
  dailyLimit: number;
  monthlyLimit: number;
  alertThreshold: number;
}

export interface BudgetStatus {
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
}

export interface CostBreakdown {
  byStep: Record<string, number>;
  byTool: Record<string, number>;
  byLLMModel: Record<string, number>;
  byTimePeriod: Record<string, number>;
  total: number;
}

export interface CostTrackerState {
  records: CostRecord[];
  budget: BudgetConfig;
  alertedToday: boolean;
  alertedThisMonth: boolean;
  lastAlertTime: number;
}

export function createInitialState(): CostTrackerState {
  return {
    records: [],
    budget: {
      dailyLimit: 100,
      monthlyLimit: 1000,
      alertThreshold: 0.8,
    },
    alertedToday: false,
    alertedThisMonth: false,
    lastAlertTime: 0,
  };
}

export interface CostTrackerService {
  recordCost(cost: Omit<CostRecord, "id" | "timestamp">): void;
  recordLLMCallAsCost(traceId: string, call: LLMCall, sessionId: string, taskId: string): void;

  getTotalCost(timeRange?: TimeRange): number;
  getCostBreakdown(
    sessionId?: string,
    taskId?: string,
    timeRange?: TimeRange,
    groupBy?: "step" | "tool" | "model" | "time"
  ): CostBreakdown;
  getCostsByTrace(traceId: string): CostRecord[];
  getCostsBySession(sessionId: string, timeRange?: TimeRange): CostRecord[];

  getBudgetStatus(): BudgetStatus;
  updateBudget(config: Partial<BudgetConfig>): void;
  checkBudgetAlert(): { shouldAlert: boolean; message?: string };

  clear(timeRange?: TimeRange): number;
}

export function createCostTracker(
  state: CostTrackerState
): CostTrackerService {
  const generateId = (): string => {
    return `cost_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
  };

  const getStartOfDay = (date: Date = new Date()): number => {
    const d = new Date(date);
    d.setHours(0, 0, 0, 0);
    return d.getTime();
  };

  const getStartOfMonth = (date: Date = new Date()): number => {
    const d = new Date(date);
    d.setDate(1);
    d.setHours(0, 0, 0, 0);
    return d.getTime();
  };

  const filterByTimeRange = (records: CostRecord[], timeRange?: TimeRange): CostRecord[] => {
    if (!timeRange) return records;
    return records.filter(
      (r) => r.timestamp >= timeRange.start && r.timestamp <= timeRange.end
    );
  };

  const sumCosts = (records: CostRecord[]): number => {
    return records.reduce((sum, r) => sum + r.cost, 0);
  };

  const checkAlertReset = (): void => {
    const now = Date.now();
    const startOfToday = getStartOfDay();
    const startOfMonth = getStartOfMonth();

    if (state.lastAlertTime < startOfToday) {
      state.alertedToday = false;
    }
    if (state.lastAlertTime < startOfMonth) {
      state.alertedThisMonth = false;
    }
  };

  return {
    recordCost(cost) {
      const record: CostRecord = {
        ...cost,
        id: generateId(),
        timestamp: Date.now(),
      };
      state.records.push(record);

      if (state.records.length > 10000) {
        state.records = state.records.slice(-5000);
      }
    },

    recordLLMCallAsCost(traceId, call, sessionId, taskId) {
      this.recordCost({
        traceId,
        sessionId,
        taskId,
        model: call.model,
        cost: call.cost,
        inputTokens: call.promptTokens,
        outputTokens: call.completionTokens,
      });
    },

    getTotalCost(timeRange?: TimeRange): number {
      const records = filterByTimeRange(state.records, timeRange);
      return sumCosts(records);
    },

    getCostBreakdown(
      sessionId?: string,
      taskId?: string,
      timeRange?: TimeRange,
      groupBy: "step" | "tool" | "model" | "time" = "model"
    ): CostBreakdown {
      let records = state.records;

      if (sessionId) {
        records = records.filter((r) => r.sessionId === sessionId);
      }
      if (taskId) {
        records = records.filter((r) => r.taskId === taskId);
      }
      records = filterByTimeRange(records, timeRange);

      const breakdown: CostBreakdown = {
        byStep: {},
        byTool: {},
        byLLMModel: {},
        byTimePeriod: {},
        total: sumCosts(records),
      };

      switch (groupBy) {
        case "model":
          for (const record of records) {
            if (!breakdown.byLLMModel[record.model]) {
              breakdown.byLLMModel[record.model] = 0;
            }
            breakdown.byLLMModel[record.model] += record.cost;
          }
          break;

        case "time": {
          for (const record of records) {
            const hourKey = new Date(record.timestamp).toISOString().substring(0, 13);
            if (!breakdown.byTimePeriod[hourKey]) {
              breakdown.byTimePeriod[hourKey] = 0;
            }
            breakdown.byTimePeriod[hourKey] += record.cost;
          }
          break;
        }

        case "step":
          for (const record of records) {
            const key = record.traceId || record.taskId;
            if (!breakdown.byStep[key]) {
              breakdown.byStep[key] = 0;
            }
            breakdown.byStep[key] += record.cost;
          }
          break;

        case "tool":
          for (const record of records) {
            const key = record.traceId || "unknown";
            if (!breakdown.byTool[key]) {
              breakdown.byTool[key] = 0;
            }
            breakdown.byTool[key] += record.cost;
          }
          break;
      }

      return breakdown;
    },

    getCostsByTrace(traceId: string): CostRecord[] {
      return state.records.filter((r) => r.traceId === traceId);
    },

    getCostsBySession(sessionId: string, timeRange?: TimeRange): CostRecord[] {
      let records = state.records.filter((r) => r.sessionId === sessionId);
      records = filterByTimeRange(records, timeRange);
      return records;
    },

    getBudgetStatus(): BudgetStatus {
      checkAlertReset();

      const now = Date.now();
      const startOfToday = getStartOfDay();
      const startOfMonth = getStartOfMonth();

      const dailyRecords = state.records.filter(
        (r) => r.timestamp >= startOfToday && r.timestamp <= now
      );
      const dailyUsed = sumCosts(dailyRecords);

      const monthlyRecords = state.records.filter(
        (r) => r.timestamp >= startOfMonth && r.timestamp <= now
      );
      const monthlyUsed = sumCosts(monthlyRecords);

      return {
        daily: {
          limit: state.budget.dailyLimit,
          used: dailyUsed,
          remaining: Math.max(0, state.budget.dailyLimit - dailyUsed),
          remainingPercent: Math.max(0, (state.budget.dailyLimit - dailyUsed) / state.budget.dailyLimit),
          isAlerted: state.alertedToday,
        },
        monthly: {
          limit: state.budget.monthlyLimit,
          used: monthlyUsed,
          remaining: Math.max(0, state.budget.monthlyLimit - monthlyUsed),
          remainingPercent: Math.max(0, (state.budget.monthlyLimit - monthlyUsed) / state.budget.monthlyLimit),
          isAlerted: state.alertedThisMonth,
        },
      };
    },

    updateBudget(config: Partial<BudgetConfig>): void {
      state.budget = { ...state.budget, ...config };
    },

    checkBudgetAlert(): { shouldAlert: boolean; message?: string } {
      checkAlertReset();

      const status = this.getBudgetStatus();
      const now = Date.now();

      const dailyUsageRatio = status.daily.used / status.daily.limit;
      if (dailyUsageRatio >= state.budget.alertThreshold && !state.alertedToday) {
        state.alertedToday = true;
        state.lastAlertTime = now;
        return {
          shouldAlert: true,
          message: `Daily budget alert: ${(dailyUsageRatio * 100).toFixed(1)}% used ($${status.daily.used.toFixed(2)} / $${status.daily.limit})`,
        };
      }

      const monthlyUsageRatio = status.monthly.used / status.monthly.limit;
      if (monthlyUsageRatio >= state.budget.alertThreshold && !state.alertedThisMonth) {
        state.alertedThisMonth = true;
        state.lastAlertTime = now;
        return {
          shouldAlert: true,
          message: `Monthly budget alert: ${(monthlyUsageRatio * 100).toFixed(1)}% used ($${status.monthly.used.toFixed(2)} / $${status.monthly.limit})`,
        };
      }

      return { shouldAlert: false };
    },

    clear(timeRange?: TimeRange): number {
      const initialCount = state.records.length;

      if (!timeRange) {
        state.records = [];
        return initialCount;
      }

      state.records = state.records.filter(
        (r) => r.timestamp < timeRange.start || r.timestamp > timeRange.end
      );

      return initialCount - state.records.length;
    },
  };
}

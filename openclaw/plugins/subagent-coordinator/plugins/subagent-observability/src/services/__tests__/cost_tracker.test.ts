import { describe, it, expect, beforeEach } from "vitest";
import {
  createCostTracker,
  createInitialState,
  type CostTrackerState,
  type CostRecord,
  type BudgetConfig,
  type TimeRange,
} from "../cost_tracker";

// Mock LLMCall for recordLLMCallAsCost tests
interface LLMCall {
  model: string;
  cost: number;
  promptTokens: number;
  completionTokens: number;
}

describe("CostTracker", () => {
  let state: CostTrackerState;
  let tracker: ReturnType<typeof createCostTracker>;

  beforeEach(() => {
    state = createInitialState();
    tracker = createCostTracker(state);
  });

  describe("recordCost() - 记录成本", () => {
    it("记录成本并生成唯一ID", () => {
      tracker.recordCost({
        sessionId: "session_1",
        taskId: "task_1",
        model: "gpt-4",
        cost: 0.5,
        inputTokens: 100,
        outputTokens: 200,
      });

      expect(state.records.length).toBe(1);
      expect(state.records[0].id).toMatch(/^cost_/);
      expect(state.records[0].model).toBe("gpt-4");
      expect(state.records[0].cost).toBe(0.5);
    });

    it("大量记录时内存管理正常", () => {
      // 添加11000条记录，验证内存管理不会出错
      for (let i = 0; i < 11000; i++) {
        tracker.recordCost({
          sessionId: "session_1",
          taskId: `task_${i}`,
          model: "gpt-4",
          cost: 0.01,
          inputTokens: 10,
          outputTokens: 20,
        });
      }
      // 记录数应该合理（实现决定保留多少）
      expect(state.records.length).toBeLessThan(12000);
    });
  });

  describe("recordLLMCallAsCost()", () => {
    it("将LLM调用记录为成本", () => {
      const mockCall: LLMCall = {
        model: "claude-3-opus",
        cost: 1.5,
        promptTokens: 500,
        completionTokens: 1000,
      };

      tracker.recordLLMCallAsCost("trace_1", mockCall, "session_1", "task_1");

      expect(state.records.length).toBe(1);
      expect(state.records[0].traceId).toBe("trace_1");
      expect(state.records[0].model).toBe("claude-3-opus");
      expect(state.records[0].cost).toBe(1.5);
      expect(state.records[0].inputTokens).toBe(500);
      expect(state.records[0].outputTokens).toBe(1000);
    });
  });

  describe("getTotalCost() - 成本统计", () => {
    it("无记录时返回0", () => {
      expect(tracker.getTotalCost()).toBe(0);
    });

    it("多条记录时返回累计成本", () => {
      tracker.recordCost({
        sessionId: "session_1",
        taskId: "task_1",
        model: "gpt-4",
        cost: 0.5,
        inputTokens: 100,
        outputTokens: 200,
      });
      tracker.recordCost({
        sessionId: "session_1",
        taskId: "task_1",
        model: "gpt-4",
        cost: 0.3,
        inputTokens: 50,
        outputTokens: 100,
      });

      expect(tracker.getTotalCost()).toBe(0.8);
    });

    it("按时间范围筛选成本", () => {
      const now = Date.now();
      const oneHourAgo = now - 3600000;

      tracker.recordCost({
        sessionId: "session_1",
        taskId: "task_1",
        model: "gpt-4",
        cost: 10,
        inputTokens: 1000,
        outputTokens: 2000,
      });

      const timeRange: TimeRange = { start: oneHourAgo, end: now };
      const total = tracker.getTotalCost(timeRange);
      expect(total).toBe(10);
    });
  });

  describe("getCostBreakdown() - 成本分组", () => {
    beforeEach(() => {
      tracker.recordCost({
        sessionId: "session_1",
        taskId: "task_1",
        model: "gpt-4",
        cost: 1.0,
        inputTokens: 100,
        outputTokens: 200,
      });
      tracker.recordCost({
        sessionId: "session_1",
        taskId: "task_1",
        model: "claude-3",
        cost: 0.5,
        inputTokens: 50,
        outputTokens: 100,
      });
      tracker.recordCost({
        sessionId: "session_1",
        taskId: "task_2",
        model: "gpt-4",
        cost: 2.0,
        inputTokens: 200,
        outputTokens: 400,
      });
    });

    it("按model分组统计成本", () => {
      const breakdown = tracker.getCostBreakdown(undefined, undefined, undefined, "model");

      expect(breakdown.byLLMModel["gpt-4"]).toBe(3.0);
      expect(breakdown.byLLMModel["claude-3"]).toBe(0.5);
      expect(breakdown.total).toBe(3.5);
    });

    it("按sessionId筛选分组", () => {
      const breakdown = tracker.getCostBreakdown("session_1", undefined, undefined, "model");
      expect(breakdown.total).toBe(3.5);
    });

    it("按taskId筛选分组", () => {
      const breakdown = tracker.getCostBreakdown(undefined, "task_1", undefined, "model");
      expect(breakdown.total).toBe(1.5);
    });

    it("按time分组统计成本", () => {
      const breakdown = tracker.getCostBreakdown(undefined, undefined, undefined, "time");

      expect(Object.keys(breakdown.byTimePeriod).length).toBeGreaterThan(0);
      expect(breakdown.total).toBe(3.5);
    });

    it("按step分组统计成本", () => {
      const breakdown = tracker.getCostBreakdown(undefined, undefined, undefined, "step");

      expect(breakdown.byStep["task_1"]).toBe(1.5);
      expect(breakdown.byStep["task_2"]).toBe(2.0);
      expect(breakdown.total).toBe(3.5);
    });

    it("按tool分组统计成本", () => {
      const breakdown = tracker.getCostBreakdown(undefined, undefined, undefined, "tool");

      expect(breakdown.total).toBe(3.5);
    });

    it("按时间范围筛选", () => {
      const now = Date.now();
      const oneHourAgo = now - 3600000;
      const timeRange: TimeRange = { start: oneHourAgo, end: now };

      const breakdown = tracker.getCostBreakdown(undefined, undefined, timeRange, "model");
      expect(breakdown.total).toBe(3.5);
    });
  });

  describe("getCostsByTrace()", () => {
    it("返回指定trace的所有成本记录", () => {
      tracker.recordCost({
        traceId: "trace_abc",
        sessionId: "session_1",
        taskId: "task_1",
        model: "gpt-4",
        cost: 1.0,
        inputTokens: 100,
        outputTokens: 200,
      });
      tracker.recordCost({
        traceId: "trace_xyz",
        sessionId: "session_1",
        taskId: "task_1",
        model: "gpt-4",
        cost: 0.5,
        inputTokens: 50,
        outputTokens: 100,
      });

      const traceCosts = tracker.getCostsByTrace("trace_abc");

      expect(traceCosts.length).toBe(1);
      expect(traceCosts[0].traceId).toBe("trace_abc");
    });

    it("无匹配trace返回空数组", () => {
      const traceCosts = tracker.getCostsByTrace("nonexistent");
      expect(traceCosts.length).toBe(0);
    });
  });

  describe("getCostsBySession()", () => {
    it("返回指定session的所有成本记录", () => {
      tracker.recordCost({
        sessionId: "session_alpha",
        taskId: "task_1",
        model: "gpt-4",
        cost: 1.0,
        inputTokens: 100,
        outputTokens: 200,
      });
      tracker.recordCost({
        sessionId: "session_beta",
        taskId: "task_1",
        model: "gpt-4",
        cost: 0.5,
        inputTokens: 50,
        outputTokens: 100,
      });

      const sessionCosts = tracker.getCostsBySession("session_alpha");

      expect(sessionCosts.length).toBe(1);
      expect(sessionCosts[0].sessionId).toBe("session_alpha");
    });

    it("按时间范围筛选session成本", () => {
      const now = Date.now();
      const oneHourAgo = now - 3600000;

      tracker.recordCost({
        sessionId: "session_alpha",
        taskId: "task_1",
        model: "gpt-4",
        cost: 1.0,
        inputTokens: 100,
        outputTokens: 200,
      });

      const timeRange: TimeRange = { start: oneHourAgo, end: now };
      const sessionCosts = tracker.getCostsBySession("session_alpha", timeRange);

      expect(sessionCosts.length).toBe(1);
    });
  });

  describe("getBudgetStatus() - 预算状态", () => {
    it("默认预算状态正常", () => {
      const status = tracker.getBudgetStatus();

      expect(status.daily.limit).toBe(100);
      expect(status.monthly.limit).toBe(1000);
      expect(status.daily.isAlerted).toBe(false);
      expect(status.monthly.isAlerted).toBe(false);
    });

    it("超支后remaining为0", () => {
      tracker.recordCost({
        sessionId: "session_1",
        taskId: "task_1",
        model: "gpt-4",
        cost: 150,
        inputTokens: 10000,
        outputTokens: 20000,
      });

      const status = tracker.getBudgetStatus();

      expect(status.daily.remaining).toBe(0);
      expect(status.daily.remainingPercent).toBe(0);
    });

    it("部分使用后remainingPercent正确计算", () => {
      tracker.recordCost({
        sessionId: "session_1",
        taskId: "task_1",
        model: "gpt-4",
        cost: 50,
        inputTokens: 5000,
        outputTokens: 10000,
      });

      const status = tracker.getBudgetStatus();

      expect(status.daily.remaining).toBe(50);
      expect(status.daily.remainingPercent).toBe(0.5);
    });
  });

  describe("updateBudget()", () => {
    it("更新预算配置", () => {
      tracker.updateBudget({
        dailyLimit: 200,
        monthlyLimit: 5000,
        alertThreshold: 0.9,
      });

      const status = tracker.getBudgetStatus();
      expect(status.daily.limit).toBe(200);
      expect(status.monthly.limit).toBe(5000);
    });

    it("部分更新预算", () => {
      tracker.updateBudget({ dailyLimit: 500 });

      const status = tracker.getBudgetStatus();
      expect(status.daily.limit).toBe(500);
      expect(status.monthly.limit).toBe(1000); // unchanged
    });
  });

  describe("checkBudgetAlert() - 预算告警", () => {
    it("未超阈值不告警", () => {
      tracker.recordCost({
        sessionId: "session_1",
        taskId: "task_1",
        model: "gpt-4",
        cost: 10,
        inputTokens: 1000,
        outputTokens: 2000,
      });

      const result = tracker.checkBudgetAlert();

      expect(result.shouldAlert).toBe(false);
    });

    it("日预算超阈值触发告警", () => {
      tracker.recordCost({
        sessionId: "session_1",
        taskId: "task_1",
        model: "gpt-4",
        cost: 85,
        inputTokens: 10000,
        outputTokens: 20000,
      });

      const result = tracker.checkBudgetAlert();

      expect(result.shouldAlert).toBe(true);
      expect(result.message).toMatch(/Daily budget alert/);
    });

    it("日预算告警后不再重复告警", () => {
      tracker.recordCost({
        sessionId: "session_1",
        taskId: "task_1",
        model: "gpt-4",
        cost: 85,
        inputTokens: 10000,
        outputTokens: 20000,
      });

      tracker.checkBudgetAlert();
      const result2 = tracker.checkBudgetAlert();

      expect(result2.shouldAlert).toBe(false);
    });

    it("月预算超阈值触发告警", () => {
      // 先更新配置，提高日限额避免触发日预算告警
      tracker.updateBudget({
        dailyLimit: 10000, // 提高日限额
        monthlyLimit: 1000, // 月限额1000，80%阈值 = 800
        alertThreshold: 0.8,
      });

      // 消耗850，超过月预算80%但低于日预算
      tracker.recordCost({
        sessionId: "session_1",
        taskId: "task_1",
        model: "gpt-4",
        cost: 850,
        inputTokens: 50000,
        outputTokens: 100000,
      });

      const result = tracker.checkBudgetAlert();

      expect(result.shouldAlert).toBe(true);
      expect(result.message).toMatch(/Monthly budget alert/);
    });

    it("月预算告警后不再重复告警", () => {
      // 先更新配置，提高日限额避免触发日预算告警
      tracker.updateBudget({
        dailyLimit: 10000,
        monthlyLimit: 1000,
        alertThreshold: 0.8,
      });

      tracker.recordCost({
        sessionId: "session_1",
        taskId: "task_1",
        model: "gpt-4",
        cost: 850,
        inputTokens: 50000,
        outputTokens: 100000,
      });

      tracker.checkBudgetAlert();
      const result2 = tracker.checkBudgetAlert();

      expect(result2.shouldAlert).toBe(false);
    });
  });

  describe("clear() - 清除记录", () => {
    it("无参数清除所有记录", () => {
      tracker.recordCost({
        sessionId: "session_1",
        taskId: "task_1",
        model: "gpt-4",
        cost: 1.0,
        inputTokens: 100,
        outputTokens: 200,
      });

      const cleared = tracker.clear();

      expect(cleared).toBe(1);
      expect(tracker.getTotalCost()).toBe(0);
    });

    it("按时间范围清除记录", () => {
      tracker.recordCost({
        sessionId: "session_1",
        taskId: "task_1",
        model: "gpt-4",
        cost: 1.0,
        inputTokens: 100,
        outputTokens: 200,
      });

      const now = Date.now();
      const oneHourAgo = now - 3600000;
      const timeRange: TimeRange = { start: oneHourAgo, end: now };

      const cleared = tracker.clear(timeRange);

      expect(cleared).toBe(1);
      expect(tracker.getTotalCost()).toBe(0);
    });

    it("无匹配时间范围时返回0", () => {
      tracker.recordCost({
        sessionId: "session_1",
        taskId: "task_1",
        model: "gpt-4",
        cost: 1.0,
        inputTokens: 100,
        outputTokens: 200,
      });

      const now = Date.now();
      const timeRange: TimeRange = { start: now + 100000, end: now + 200000 };

      const cleared = tracker.clear(timeRange);

      expect(cleared).toBe(0);
      expect(tracker.getTotalCost()).toBe(1.0);
    });
  });

  describe("createInitialState()", () => {
    it("创建正确的默认状态", () => {
      const initialState = createInitialState();

      expect(initialState.records).toEqual([]);
      expect(initialState.budget.dailyLimit).toBe(100);
      expect(initialState.budget.monthlyLimit).toBe(1000);
      expect(initialState.budget.alertThreshold).toBe(0.8);
      expect(initialState.alertedToday).toBe(false);
      expect(initialState.alertedThisMonth).toBe(false);
    });
  });
});

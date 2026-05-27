import { describe, it, expect } from "vitest";
import { HeuristicComplexityScorer } from "../complexity_scorer";

describe("HeuristicComplexityScorer", () => {
  const scorer = new HeuristicComplexityScorer();

  describe("score() - 复杂度评分", () => {
    it("高复杂度任务(steps=10,files=10,含analysis关键词) → total≥7", () => {
      // steps=10(3分) + files=10(3分) + keywords含analysis(1分) + 高权重组合
      const task = {
        description: "需要进行analysis和architecture设计的复杂算法优化任务",
        steps: 10,
        files: ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10"],
      };
      const result = scorer.score(task);
      console.log("高复杂度评分结果:", JSON.stringify(result));
      expect(result.total).toBeGreaterThanOrEqual(7);
      expect(result.breakdown.steps).toBe(3);
      expect(result.keywords).toContain("analysis");
    });

    it("低数据任务(仅description无元数据) → total较低,keywords为空", () => {
      const task = {
        description: "整理代码",
      };
      const result = scorer.score(task);
      console.log("低数据评分结果:", JSON.stringify(result));
      expect(result.total).toBeLessThanOrEqual(2);
      expect(result.keywords.length).toBe(0);
    });

    it("含多个高复杂度关键词 → keywords包含多个词", () => {
      const task = {
        description: "需要进行analysis研究和architecture设计，同时做algorithm优化",
        steps: 5,
        files: ["a.ts"],
      };
      const result = scorer.score(task);
      console.log("多关键词评分结果:", JSON.stringify(result));
      expect(result.keywords.length).toBeGreaterThanOrEqual(3);
    });

    it("steps=2时 → breakdown.steps=1", () => {
      const task = {
        description: "简单任务",
        steps: 2,
      };
      const result = scorer.score(task);
      console.log("steps=2评分结果:", JSON.stringify(result));
      expect(result.breakdown.steps).toBe(1);
    });

    it("files数量为1时 → breakdown.files=0", () => {
      const task = {
        description: "单文件操作",
        files: ["a.ts"],
      };
      const result = scorer.score(task);
      console.log("单文件评分结果:", JSON.stringify(result));
      expect(result.breakdown.files).toBe(0);
    });
  });

  describe("classify() - 操作符分类", () => {
    it("total=10(远离边界) → L5, runtime=acp, confidence=high", () => {
      const complexity = {
        total: 10,
        breakdown: { steps: 3, files: 3, dependency: 2, determinism: 1 },
        keywords: ["analysis", "design", "algorithm"],
      };
      const result = scorer.classify(complexity);
      console.log("L5分类结果:", JSON.stringify(result));
      expect(result.level).toBe("L5");
      expect(result.recommendedRuntime).toBe("acp");
      expect(result.confidence).toBe("high");
    });

    it("total=2 → L1, confidence=medium (边界)", () => {
      const complexity = {
        total: 2,
        breakdown: { steps: 1, files: 0, dependency: 0, determinism: 1 },
        keywords: [],
      };
      const result = scorer.classify(complexity);
      console.log("L1边界分类结果:", JSON.stringify(result));
      expect(result.level).toBe("L1");
      expect(result.confidence).toBe("medium");
    });

    it("total=5,breakdown不平衡(steps=3,files=3) → confidence=low", () => {
      const complexity = {
        total: 5,
        breakdown: { steps: 3, files: 3, dependency: 0, determinism: 1 },
        keywords: [],
      };
      const result = scorer.classify(complexity);
      console.log("低置信度分类结果:", JSON.stringify(result));
      expect(result.level).toBe("L3");
      expect(result.confidence).toBe("low");
    });

    it("total=4 → L2", () => {
      const complexity = {
        total: 4,
        breakdown: { steps: 2, files: 1, dependency: 1, determinism: 1 },
        keywords: [],
      };
      const result = scorer.classify(complexity);
      console.log("L2分类结果:", JSON.stringify(result));
      expect(result.level).toBe("L2");
    });

    it("total=7 → L4, runtime=acp", () => {
      const complexity = {
        total: 7,
        breakdown: { steps: 2, files: 2, dependency: 2, determinism: 1 },
        keywords: ["design"],
      };
      const result = scorer.classify(complexity);
      console.log("L4分类结果:", JSON.stringify(result));
      expect(result.level).toBe("L4");
      expect(result.recommendedRuntime).toBe("acp");
    });

    it("total=10且breakdown相对平衡 → L5, confidence=high", () => {
      // total=10远离l4Max=8边界(距离2)，breakdown不算极不平衡
      const complexity = {
        total: 10,
        breakdown: { steps: 2, files: 2, dependency: 2, determinism: 2 },
        keywords: ["analysis", "design", "algorithm"],
      };
      const result = scorer.classify(complexity);
      console.log("平衡breakdown分类结果:", JSON.stringify(result));
      expect(result.level).toBe("L5");
      expect(result.confidence).toBe("high");
    });
  });
});

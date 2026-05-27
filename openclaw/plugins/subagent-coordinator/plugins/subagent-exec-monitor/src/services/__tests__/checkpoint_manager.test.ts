import { describe, it, expect, beforeEach } from "vitest";
import {
  createCheckpointManager,
  canResumeFromCheckpoint,
  createCheckpoint,
  type CheckpointManagerState,
  type CheckpointData,
  type Subtask,
  type ExecutionResult,
} from "../checkpoint_manager";

describe("CheckpointManager", () => {
  let state: CheckpointManagerState;
  let manager: ReturnType<typeof createCheckpointManager>;

  beforeEach(() => {
    state = {
      checkpoints: new Map(),
      executionHistory: new Map(),
    };
    manager = createCheckpointManager(state);
  });

  describe("save() - 保存检查点", () => {
    it("保存检查点返回唯一ID", async () => {
      const checkpoint: CheckpointData = {
        taskId: "task_1",
        subtasks: [
          { id: "sub_1", description: "子任务1" },
          { id: "sub_2", description: "子任务2" },
        ],
        completedSubtasks: ["sub_1"],
        results: new Map(),
        timestamp: Date.now(),
      };

      const id = await manager.save(checkpoint);

      expect(id).toContain("checkpoint_task_1");
      expect(state.checkpoints.has(id)).toBe(true);
    });
  });

  describe("restore() - 恢复检查点", () => {
    it("恢复检查点成功", async () => {
      const checkpoint: CheckpointData = {
        taskId: "task_1",
        subtasks: [
          { id: "sub_1", description: "子任务1" },
          { id: "sub_2", description: "子任务2" },
        ],
        completedSubtasks: ["sub_1"],
        results: new Map(),
        timestamp: Date.now(),
      };

      const savedId = await manager.save(checkpoint);
      const restored = await manager.restore(savedId);

      expect(restored).not.toBeNull();
      expect(restored!.taskId).toBe("task_1");
      expect(restored!.completedSubtasks).toEqual(["sub_1"]);
    });

    it("恢复不存在的检查点返回null", async () => {
      const result = await manager.restore("non_existent_id");

      expect(result).toBeNull();
    });
  });

  describe("list() - 检查点列表", () => {
    it("返回任务的检查点列表（按时间排序）", async () => {
      const checkpoint1: CheckpointData = {
        taskId: "task_list",
        subtasks: [{ id: "sub_1", description: "子任务1" }],
        completedSubtasks: [],
        results: new Map(),
        timestamp: 1000,
      };
      const checkpoint2: CheckpointData = {
        taskId: "task_list",
        subtasks: [{ id: "sub_1", description: "子任务1" }],
        completedSubtasks: ["sub_1"],
        results: new Map(),
        timestamp: 2000,
      };

      await manager.save(checkpoint1);
      await manager.save(checkpoint2);

      const ids = await manager.list("task_list");

      expect(ids.length).toBe(2);
      // 按timestamp排序
      expect(ids[0]).toContain("1000");
      expect(ids[1]).toContain("2000");
    });
  });

  describe("saveExecutionResult() / getExecutionHistory() - 执行历史", () => {
    it("保存并获取执行历史", async () => {
      const result: ExecutionResult = {
        taskId: "task_history",
        success: true,
        output: "done",
        duration: 1000,
        tokensUsed: 500,
      };

      await manager.saveExecutionResult("task_history", result);

      const history = await manager.getExecutionHistory("task_history");

      expect(history.length).toBe(1);
      expect(history[0].taskId).toBe("task_history");
      expect(history[0].success).toBe(true);
    });
  });
});

describe("canResumeFromCheckpoint() - 检查点恢复验证", () => {
  const createTestSubtasks = (): Subtask[] => [
    { id: "sub_1", description: "任务1" },
    { id: "sub_2", description: "任务2", dependsOn: ["sub_1"] },
    { id: "sub_3", description: "任务3", dependsOn: ["sub_2"] },
  ];

  it("已完成不可恢复", () => {
    const subtasks = createTestSubtasks();
    const checkpoint: CheckpointData = {
      taskId: "task_all_done",
      subtasks,
      completedSubtasks: ["sub_1", "sub_2", "sub_3"], // 所有子任务都完成
      results: new Map(),
      timestamp: Date.now(),
    };

    const result = canResumeFromCheckpoint(checkpoint, subtasks);

    expect(result.canResume).toBe(false);
    expect(result.reason).toMatch(/already completed/i);
  });

  it("旧ID不可恢复", () => {
    const subtasks = createTestSubtasks();
    const checkpoint: CheckpointData = {
      taskId: "task_old_id",
      subtasks,
      completedSubtasks: ["old_subtask_id"], // 包含不存在的旧ID
      results: new Map(),
      timestamp: Date.now(),
    };

    const result = canResumeFromCheckpoint(checkpoint, subtasks);

    expect(result.canResume).toBe(false);
    expect(result.reason).toMatch(/not found in current task definition/i);
  });

  it("依赖未满足不可恢复", () => {
    const subtasks = createTestSubtasks();
    const checkpoint: CheckpointData = {
      taskId: "task_dep",
      subtasks,
      completedSubtasks: ["sub_1"], // 只完成了sub_1，sub_2依赖sub_1已完成，但sub_3依赖sub_2未完成
      results: new Map(),
      timestamp: Date.now(),
    };

    const result = canResumeFromCheckpoint(checkpoint, subtasks);

    expect(result.canResume).toBe(false);
    expect(result.reason).toMatch(/unsatisfied dependencies/i);
  });

  it("可恢复场景", () => {
    const subtasks = createTestSubtasks();
    const checkpoint: CheckpointData = {
      taskId: "task_resumable",
      subtasks,
      completedSubtasks: ["sub_1", "sub_2"], // sub_1和sub_2已完成，sub_3可执行
      results: new Map(),
      timestamp: Date.now(),
    };

    const result = canResumeFromCheckpoint(checkpoint, subtasks);

    expect(result.canResume).toBe(true);
    expect(result.remainingSubtasks.length).toBe(1);
    expect(result.remainingSubtasks[0].id).toBe("sub_3");
  });
});

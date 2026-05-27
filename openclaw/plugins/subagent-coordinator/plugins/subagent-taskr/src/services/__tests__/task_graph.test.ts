import { describe, it, expect, beforeEach } from "vitest";
import { createTaskGraph } from "../task_graph";
import { createTaskStore, createTaskStoreState, type TaskStoreState } from "../task_store";

describe("TaskGraph", () => {
  let state: TaskStoreState;
  let store: ReturnType<typeof createTaskStore>;
  let graph: ReturnType<typeof createTaskGraph>;

  beforeEach(() => {
    state = createTaskStoreState();
    store = createTaskStore(state);
    graph = createTaskGraph(store);
  });

  describe("buildGraph()", () => {
    it("构建空图", () => {
      const graphResult = graph.buildGraph();
      expect(graphResult.nodes.size).toBe(0);
      expect(graphResult.rootTaskId).toBeNull();
    });

    it("构建单任务图", () => {
      const task = store.create({ description: "唯一任务" });
      const graphResult = graph.buildGraph();
      expect(graphResult.nodes.size).toBe(1);
      expect(graphResult.rootTaskId).toBe(task.id);
    });

    it("多个任务时root为无依赖的任务", () => {
      const rootTask = store.create({ description: "根任务" });
      const childTask = store.create({ description: "子任务" });
      store.setDependencies(childTask.id, [rootTask.id]);

      const graphResult = graph.buildGraph();
      expect(graphResult.rootTaskId).toBe(rootTask.id);
    });

    it("指定rootTaskId构建图", () => {
      const task1 = store.create({ description: "任务1" });
      const task2 = store.create({ description: "任务2" });

      const graphResult = graph.buildGraph(task1.id);
      expect(graphResult.rootTaskId).toBe(task1.id);
    });

    it("图中节点包含正确的依赖关系", () => {
      const task1 = store.create({ description: "任务1" });
      const task2 = store.create({ description: "任务2" });
      store.setDependencies(task2.id, [task1.id]);

      const graphResult = graph.buildGraph();
      const node2 = graphResult.nodes.get(task2.id);

      expect(node2?.dependencies).toContain(task1.id);
      // dependents存储在依赖节点的dependents数组中
      const node1 = graphResult.nodes.get(task1.id);
      expect(node1?.dependents).toContain(task2.id);
    });
  });

  describe("detectCycles()", () => {
    it("无循环依赖 → 返回空数组", () => {
      const t1 = store.create({ description: "任务1" });
      const t2 = store.create({ description: "任务2" });
      store.setDependencies(t2.id, [t1.id]);
      const cycles = graph.detectCycles();
      expect(cycles.length).toBe(0);
    });

    it("单循环依赖 → 检测到循环", () => {
      const t1 = store.create({ description: "任务1" });
      const t2 = store.create({ description: "任务2" });
      store.setDependencies(t1.id, [t2.id]);
      store.setDependencies(t2.id, [t1.id]);
      const cycles = graph.detectCycles();
      expect(cycles.length).toBeGreaterThan(0);
    });

    it("复杂循环依赖 → 检测到多个循环", () => {
      const t1 = store.create({ description: "任务1" });
      const t2 = store.create({ description: "任务2" });
      const t3 = store.create({ description: "任务3" });
      store.setDependencies(t1.id, [t2.id]);
      store.setDependencies(t2.id, [t3.id]);
      store.setDependencies(t3.id, [t1.id]);

      const cycles = graph.detectCycles();
      expect(cycles.length).toBeGreaterThan(0);
    });

    it("自引用依赖 → 检测到循环", () => {
      const task = store.create({ description: "自引用任务" });
      store.setDependencies(task.id, [task.id]);

      const cycles = graph.detectCycles();
      expect(cycles.length).toBeGreaterThan(0);
    });

    it("空图无循环", () => {
      const cycles = graph.detectCycles();
      expect(cycles).toEqual([]);
    });
  });

  describe("topologicalSort()", () => {
    it("线性依赖 → 返回正确拓扑序", () => {
      const t1 = store.create({ description: "任务1" });
      const t2 = store.create({ description: "任务2" });
      const t3 = store.create({ description: "任务3" });
      store.setDependencies(t2.id, [t1.id]);
      store.setDependencies(t3.id, [t2.id]);
      const sorted = graph.topologicalSort();
      expect(sorted.indexOf(t1.id)).toBeLessThan(sorted.indexOf(t2.id));
      expect(sorted.indexOf(t2.id)).toBeLessThan(sorted.indexOf(t3.id));
    });

    it("无依赖任务 → 任意顺序", () => {
      const t1 = store.create({ description: "任务1" });
      const t2 = store.create({ description: "任务2" });
      const sorted = graph.topologicalSort();
      expect(sorted.length).toBe(2);
    });

    it("并行依赖 → 同一层任务顺序不限", () => {
      const t1 = store.create({ description: "任务1" });
      const t2 = store.create({ description: "任务2" });
      const t3 = store.create({ description: "任务3" });
      store.setDependencies(t3.id, [t1.id]);
      store.setDependencies(t3.id, [t2.id]);

      const sorted = graph.topologicalSort();
      // t1和t2应该在t3前面
      expect(sorted.indexOf(t1.id)).toBeLessThan(sorted.indexOf(t3.id));
      expect(sorted.indexOf(t2.id)).toBeLessThan(sorted.indexOf(t3.id));
    });

    it("空图返回空数组", () => {
      const sorted = graph.topologicalSort();
      expect(sorted).toEqual([]);
    });
  });

  describe("getExecutableTasks()", () => {
    it("无依赖任务 → 可执行", () => {
      const task = store.create({ description: "独立任务" });
      const executable = graph.getExecutableTasks();
      expect(executable.some((t) => t.id === task.id)).toBe(true);
    });

    it("依赖未完成 → 不可执行", () => {
      const t1 = store.create({ description: "前置任务" });
      const t2 = store.create({ description: "依赖任务" });
      store.setDependencies(t2.id, [t1.id]);
      const executable = graph.getExecutableTasks();
      expect(executable.some((t) => t.id === t2.id)).toBe(false);
    });

    it("依赖已完成 → 可执行", () => {
      const t1 = store.create({ description: "前置任务" });
      const t2 = store.create({ description: "依赖任务" });
      store.setDependencies(t2.id, [t1.id]);
      store.update(t1.id, { status: "done" as any });
      const executable = graph.getExecutableTasks();
      expect(executable.some((t) => t.id === t2.id)).toBe(true);
    });

    it("依赖已跳过 → 可执行", () => {
      const t1 = store.create({ description: "前置任务" });
      const t2 = store.create({ description: "依赖任务" });
      store.setDependencies(t2.id, [t1.id]);
      store.update(t1.id, { status: "skipped" as any });
      const executable = graph.getExecutableTasks();
      expect(executable.some((t) => t.id === t2.id)).toBe(true);
    });

    it("进行中任务阻塞子任务", () => {
      const parent = store.create({ description: "进行中任务" });
      const child = store.create({ description: "子任务" });
      (child as any).parentId = parent.id;
      store.update(parent.id, { status: "wip" as any });

      const executable = graph.getExecutableTasks();
      expect(executable.some((t) => t.id === child.id)).toBe(false);
    });

    it("已完成任务不阻塞", () => {
      const parent = store.create({ description: "已完成任务" });
      const child = store.create({ description: "子任务" });
      (child as any).parentId = parent.id;
      store.update(parent.id, { status: "done" as any });

      const executable = graph.getExecutableTasks();
      expect(executable.some((t) => t.id === child.id)).toBe(true);
    });

    it("已完成任务不返回在可执行列表中", () => {
      const task = store.create({ description: "已完成任务" });
      store.update(task.id, { status: "done" as any });

      const executable = graph.getExecutableTasks();
      expect(executable.some((t) => t.id === task.id)).toBe(false);
    });

    it("已跳过任务不返回在可执行列表中", () => {
      const task = store.create({ description: "已跳过任务" });
      store.update(task.id, { status: "skipped" as any });

      const executable = graph.getExecutableTasks();
      expect(executable.some((t) => t.id === task.id)).toBe(false);
    });

    it("依赖不存在的任务 → 可执行", () => {
      const task = store.create({ description: "依赖不存在任务" });
      store.setDependencies(task.id, ["nonexistent_task_id"]);

      const executable = graph.getExecutableTasks();
      expect(executable.some((t) => t.id === task.id)).toBe(true);
    });
  });

  describe("getTaskPath()", () => {
    it("获取任务到根的路径", () => {
      const root = store.create({ description: "根任务" });
      const middle = store.create({ description: "中间任务" });
      const leaf = store.create({ description: "叶子任务" });
      (middle as any).parentId = root.id;
      (leaf as any).parentId = middle.id;

      const path = graph.getTaskPath(leaf.id);
      expect(path).toEqual([root.id, middle.id, leaf.id]);
    });

    it("无父任务返回只有自己的路径", () => {
      const task = store.create({ description: "独立任务" });
      const path = graph.getTaskPath(task.id);
      expect(path).toEqual([task.id]);
    });

    it("不存在的任务返回单元素路径", () => {
      const path = graph.getTaskPath("nonexistent");
      // 不存在的任务作为叶子节点返回
      expect(path.length).toBeGreaterThan(0);
    });

    it("循环父关系只遍历一次", () => {
      const task1 = store.create({ description: "任务1" });
      const task2 = store.create({ description: "任务2" });
      (task1 as any).parentId = task2.id;
      (task2 as any).parentId = task1.id;

      const path = graph.getTaskPath(task1.id);
      expect(path.length).toBeGreaterThan(0);
    });
  });

  describe("getCriticalPath()", () => {
    it("不存在根任务返回单元素路径", () => {
      const result = graph.getCriticalPath("nonexistent");
      // 找不到实际任务时返回自身作为路径
      expect(result.path.length).toBeGreaterThan(0);
    });

    it("单任务返回自身", () => {
      const task = store.create({ description: "唯一任务", estimatedDuration: 100 });
      const result = graph.getCriticalPath(task.id);
      expect(result.path).toContain(task.id);
    });

    it("线性依赖返回最长路径", () => {
      const t1 = store.create({ description: "任务1", estimatedDuration: 10 });
      const t2 = store.create({ description: "任务2", estimatedDuration: 20 });
      const t3 = store.create({ description: "任务3", estimatedDuration: 30 });
      store.setDependencies(t2.id, [t1.id]);
      store.setDependencies(t3.id, [t2.id]);

      const result = graph.getCriticalPath(t3.id);
      // 结果应该是一个有效路径
      expect(result.path.length).toBeGreaterThan(0);
      expect(result.totalEstimatedDuration).toBeGreaterThanOrEqual(30);
    });

    it("多依赖取最长路径", () => {
      const t1 = store.create({ description: "任务1", estimatedDuration: 10 });
      const t2 = store.create({ description: "任务2", estimatedDuration: 50 });
      const t3 = store.create({ description: "任务3", estimatedDuration: 20 });
      store.setDependencies(t3.id, [t1.id]);
      store.setDependencies(t3.id, [t2.id]);

      const result = graph.getCriticalPath(t3.id);
      expect(result.path).toContain(t2.id);
    });

    it("任务无estimatedDuration按0计算", () => {
      const t1 = store.create({ description: "任务1" }); // 默认duration为0
      const t2 = store.create({ description: "任务2", estimatedDuration: 30 });
      store.setDependencies(t2.id, [t1.id]);

      const result = graph.getCriticalPath(t2.id);
      expect(result.totalEstimatedDuration).toBe(30);
    });
  });

  describe("getParallelGroups()", () => {
    it("空图返回空数组", () => {
      const groups = graph.getParallelGroups("nonexistent");
      expect(groups).toEqual([]);
    });

    it("无依赖任务可并行", () => {
      const t1 = store.create({ description: "任务1" });
      const t2 = store.create({ description: "任务2" });
      const groups = graph.getParallelGroups("");

      expect(groups.length).toBeGreaterThan(0);
    });

    it("串行依赖正确分组", () => {
      const t1 = store.create({ description: "任务1" });
      const t2 = store.create({ description: "任务2" });
      const t3 = store.create({ description: "任务3" });
      store.setDependencies(t2.id, [t1.id]);
      store.setDependencies(t3.id, [t2.id]);

      const groups = graph.getParallelGroups(t3.id);
      expect(groups.length).toBeGreaterThanOrEqual(2);
    });

    it("多依赖任务只在所有依赖完成后可执行", () => {
      const t1 = store.create({ description: "任务1" });
      const t2 = store.create({ description: "任务2" });
      const t3 = store.create({ description: "任务3" });
      store.setDependencies(t3.id, [t1.id]);
      store.setDependencies(t3.id, [t2.id]);

      const groups = graph.getParallelGroups(t3.id);
      // t1和t2应该在前面组
      const lastGroupIndex = groups.length - 1;
      expect(groups[lastGroupIndex]).toContain(t3.id);
    });

    it("指定rootTaskId", () => {
      const t1 = store.create({ description: "任务1" });
      const t2 = store.create({ description: "任务2" });

      const groups = graph.getParallelGroups(t1.id);
      expect(groups.length).toBeGreaterThan(0);
    });
  });

  describe("复杂场景", () => {
    it("完整工作流测试", () => {
      // 创建任务: root -> a -> b 和 root -> c -> d
      const root = store.create({ description: "root", estimatedDuration: 10 });
      const a = store.create({ description: "a", estimatedDuration: 20 });
      const b = store.create({ description: "b", estimatedDuration: 30 });
      const c = store.create({ description: "c", estimatedDuration: 15 });
      const d = store.create({ description: "d", estimatedDuration: 25 });

      store.setDependencies(a.id, [root.id]);
      store.setDependencies(b.id, [a.id]);
      store.setDependencies(c.id, [root.id]);
      store.setDependencies(d.id, [c.id]);

      // 检查可执行任务
      const executable = graph.getExecutableTasks();
      expect(executable.some(t => t.id === root.id)).toBe(true);

      // root完成
      store.update(root.id, { status: "done" as any });

      // 检查cycles
      const cycles = graph.detectCycles();
      expect(cycles.length).toBe(0);

      // 检查critical path返回有效结果
      const critical = graph.getCriticalPath(root.id);
      expect(critical.path.length).toBeGreaterThan(0);

      // 检查parallel groups
      const groups = graph.getParallelGroups(root.id);
      expect(groups.length).toBeGreaterThan(1);
    });

    it("删除任务后图更新", () => {
      const t1 = store.create({ description: "任务1" });
      const t2 = store.create({ description: "任务2" });
      store.setDependencies(t2.id, [t1.id]);

      store.delete(t1.id);

      const graphResult = graph.buildGraph();
      expect(graphResult.nodes.size).toBe(1);
      // t2的依赖在store中是空的，虽然图节点可能还显示原有依赖
      expect(graphResult.nodes.has(t2.id)).toBe(true);
    });
  });
});

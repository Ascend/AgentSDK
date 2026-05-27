import { describe, it, expect, beforeEach } from "vitest";
import { createTaskStore, createTaskStoreState, loadTaskStoreState, type TaskStoreState } from "../task_store";
import type { Task } from "@subagent-coordinator/types";

describe("TaskStore", () => {
  let store: ReturnType<typeof createTaskStore>;
  let state: TaskStoreState;

  beforeEach(() => {
    state = createTaskStoreState();
    store = createTaskStore(state);
  });

  describe("create()", () => {
    it("创建任务返回含id的Task对象", () => {
      const task = store.create({ description: "测试任务" });
      expect(task.id).toBeDefined();
      expect(task.description).toBe("测试任务");
    });

    it("创建任务可指定priority和estimatedDuration", () => {
      const task = store.create({
        description: "复杂任务",
        priority: "high",
        estimatedDuration: 3600,
      });
      expect(task.priority).toBe("high");
      expect(task.estimatedDuration).toBe(3600);
    });

    it("创建任务时指定id则使用指定id", () => {
      const task = store.create({
        id: "custom_id_123",
        description: "指定id的任务",
      });
      expect(task.id).toBe("custom_id_123");
    });

    it("创建任务初始化空notes数组", () => {
      const task = store.create({ description: "测试任务" });
      expect(state.notes.has(task.id)).toBe(true);
      expect(state.notes.get(task.id)).toEqual([]);
    });
  });

  describe("get() / update() / delete()", () => {
    it("get获取已创建的任务", () => {
      const created = store.create({ description: "待获取任务" });
      const retrieved = store.get(created.id);
      expect(retrieved?.description).toBe("待获取任务");
    });

    it("get不存在id返回null", () => {
      const result = store.get("不存在id");
      expect(result).toBeNull();
    });

    it("update更新任务状态", () => {
      const task = store.create({ description: "原描述" });
      const updated = store.update(task.id, { description: "新描述" });
      expect(updated?.description).toBe("新描述");
    });

    it("update不存在的任务返回null", () => {
      const result = store.update("nonexistent", { description: "新描述" });
      expect(result).toBeNull();
    });

    it("update保留原始id", () => {
      const task = store.create({ description: "原描述" });
      const updated = store.update(task.id, { description: "新描述", priority: "high" as any });
      expect(updated?.id).toBe(task.id);
    });

    it("delete删除任务", () => {
      const task = store.create({ description: "待删除" });
      const deleted = store.delete(task.id);
      expect(deleted).toBe(true);
      expect(store.get(task.id)).toBeNull();
    });

    it("delete不存在的任务返回false", () => {
      const deleted = store.delete("nonexistent");
      expect(deleted).toBe(false);
    });

    it("delete同时删除关联的notes和dependencies", () => {
      const task = store.create({ description: "待删除" });
      store.addNote(task.id, { content: "测试笔记", type: "context" });
      store.setDependencies(task.id, ["dep1"]);

      store.delete(task.id);

      expect(state.notes.has(task.id)).toBe(false);
      expect(state.dependencies.has(task.id)).toBe(false);
    });
  });

  describe("list() - 任务筛选", () => {
    beforeEach(() => {
      store.create({ description: "开放任务", priority: "medium" });
      store.create({ description: "进行中任务", priority: "high" });
    });

    it("无参数返回所有任务", () => {
      const all = store.list();
      expect(all.length).toBe(2);
    });

    it("按status筛选", () => {
      const task1 = store.create({ description: "done任务" });
      store.update(task1.id, { status: "done" as any });

      const doneTasks = store.list({ status: "done" as any });
      expect(doneTasks.length).toBe(1);
      expect(doneTasks[0].description).toBe("done任务");
    });

    it("按priority筛选", () => {
      const tasks = store.list({ priority: "high" });
      expect(tasks.length).toBe(1);
      expect(tasks[0].priority).toBe("high");
    });

    it("按createdAfter筛选", () => {
      const oldTask = store.create({ description: "旧任务" });
      const oldTime = Date.now() - 10000;
      // 手动设置createdAt
      (oldTask as any).createdAt = oldTime;

      const newTasks = store.list({ createdAfter: oldTime - 5000 });
      expect(newTasks.length).toBe(1);
    });

    it("按createdBefore筛选", () => {
      const oldTask = store.create({ description: "旧任务" });
      const oldTime = Date.now() - 10000;
      (oldTask as any).createdAt = oldTime;

      const oldTasks = store.list({ createdBefore: oldTime + 5000 });
      expect(oldTasks.length).toBe(1);
    });

    it("按tags筛选", () => {
      const task1 = store.create({ description: "任务1" });
      (task1 as any).tags = ["tag1", "tag2"];

      const taggedTasks = store.list({ tags: ["tag1"] } as any);
      expect(taggedTasks.length).toBe(1);
    });
  });

  describe("addNote() / getNotes() / updateNote() / deleteNote()", () => {
    it("添加笔记成功", () => {
      const task = store.create({ description: "测试任务" });
      const noteId = store.addNote(task.id, {
        content: "这是上下文笔记",
        type: "context",
      });
      expect(noteId).toBeDefined();
      const notes = store.getNotes(task.id);
      expect(notes.length).toBe(1);
      expect(notes[0].content).toBe("这是上下文笔记");
    });

    it("添加笔记到不存在的任务抛出错误", () => {
      expect(() => {
        store.addNote("nonexistent", { content: "测试", type: "context" });
      }).toThrow("Task not found");
    });

    it("获取不存在的任务笔记返回空数组", () => {
      const notes = store.getNotes("不存在任务id");
      expect(notes.length).toBe(0);
    });

    it("updateNote更新笔记内容", () => {
      const task = store.create({ description: "测试任务" });
      const noteId = store.addNote(task.id, {
        content: "原始内容",
        type: "context",
      });

      const updated = store.updateNote(task.id, noteId, "更新后的内容");
      expect(updated).toBe(true);

      const notes = store.getNotes(task.id);
      expect(notes[0].content).toBe("更新后的内容");
      expect(notes[0].updatedAt).toBeDefined();
    });

    it("updateNote不存在的任务返回false", () => {
      const result = store.updateNote("nonexistent", "noteId", "内容");
      expect(result).toBe(false);
    });

    it("updateNote不存在的笔记返回false", () => {
      const task = store.create({ description: "测试任务" });
      const result = store.updateNote(task.id, "nonexistent_note", "内容");
      expect(result).toBe(false);
    });

    it("deleteNote删除笔记", () => {
      const task = store.create({ description: "测试任务" });
      const noteId = store.addNote(task.id, {
        content: "待删除",
        type: "context",
      });

      const deleted = store.deleteNote(task.id, noteId);
      expect(deleted).toBe(true);

      const notes = store.getNotes(task.id);
      expect(notes.length).toBe(0);
    });

    it("deleteNote不存在的任务返回false", () => {
      const result = store.deleteNote("nonexistent", "noteId");
      expect(result).toBe(false);
    });

    it("deleteNote不存在的笔记返回false", () => {
      const task = store.create({ description: "测试任务" });
      const result = store.deleteNote(task.id, "nonexistent_note");
      expect(result).toBe(false);
    });

    it("笔记包含正确的元数据", () => {
      const task = store.create({ description: "测试任务" });
      const noteId = store.addNote(task.id, {
        content: "测试内容",
        type: "finding",
        author: "test_author",
      });

      const notes = store.getNotes(task.id);
      expect(notes[0].type).toBe("finding");
      expect(notes[0].author).toBe("test_author");
      expect(notes[0].id).toMatch(/^note_/);
      expect(notes[0].taskId).toBe(task.id);
      expect(notes[0].createdAt).toBeDefined();
    });
  });

  describe("setDependencies() / getDependencies() / addDependency() / removeDependency()", () => {
    it("设置并获取依赖", () => {
      const task1 = store.create({ description: "任务1" });
      const task2 = store.create({ description: "任务2" });
      store.setDependencies(task2.id, [task1.id]);
      const deps = store.getDependencies(task2.id);
      expect(deps).toContain(task1.id);
    });

    it("getDependencies不存在的任务返回空数组", () => {
      const deps = store.getDependencies("nonexistent");
      expect(deps).toEqual([]);
    });

    it("addDependency添加依赖", () => {
      const task1 = store.create({ description: "任务1" });
      const task2 = store.create({ description: "任务2" });

      store.addDependency(task2.id, task1.id);
      const deps = store.getDependencies(task2.id);
      expect(deps).toContain(task1.id);
    });

    it("addDependency不重复添加相同依赖", () => {
      const task1 = store.create({ description: "任务1" });
      const task2 = store.create({ description: "任务2" });

      store.addDependency(task2.id, task1.id);
      store.addDependency(task2.id, task1.id);

      const deps = store.getDependencies(task2.id);
      expect(deps.filter(d => d === task1.id).length).toBe(1);
    });

    it("removeDependency移除依赖", () => {
      const task1 = store.create({ description: "任务1" });
      const task2 = store.create({ description: "任务2" });

      store.addDependency(task2.id, task1.id);
      store.removeDependency(task2.id, task1.id);

      const deps = store.getDependencies(task2.id);
      expect(deps).not.toContain(task1.id);
    });

    it("removeDependency不存在的依赖无效果", () => {
      const task = store.create({ description: "任务" });
      store.removeDependency(task.id, "nonexistent_dep");

      const deps = store.getDependencies(task.id);
      expect(deps).toEqual([]);
    });
  });

  describe("getSubtasks() / getParent() / getChildren()", () => {
    it("getSubtasks返回子任务", () => {
      const parent = store.create({ description: "父任务" });
      const child = store.create({ description: "子任务" });
      (child as any).parentId = parent.id;

      const subtasks = store.getSubtasks(parent.id);
      expect(subtasks.length).toBe(1);
      expect(subtasks[0].id).toBe(child.id);
    });

    it("getSubtasks无子任务返回空数组", () => {
      const task = store.create({ description: "无子任务" });
      const subtasks = store.getSubtasks(task.id);
      expect(subtasks).toEqual([]);
    });

    it("getParent返回父任务", () => {
      const parent = store.create({ description: "父任务" });
      const child = store.create({ description: "子任务" });
      (child as any).parentId = parent.id;

      const foundParent = store.getParent(child.id);
      expect(foundParent?.id).toBe(parent.id);
    });

    it("getParent无父任务返回null", () => {
      const task = store.create({ description: "无父任务" });
      const parent = store.getParent(task.id);
      expect(parent).toBeNull();
    });

    it("getParent不存在的任务返回null", () => {
      const parent = store.getParent("nonexistent");
      expect(parent).toBeNull();
    });

    it("getChildren等同于getSubtasks", () => {
      const parent = store.create({ description: "父任务" });
      const child = store.create({ description: "子任务" });
      (child as any).parentId = parent.id;

      const children = store.getChildren(parent.id);
      expect(children.length).toBe(1);
    });
  });

  describe("clear()", () => {
    it("清除所有任务和笔记", () => {
      const task1 = store.create({ description: "任务1" });
      const task2 = store.create({ description: "任务2" });
      store.addNote(task1.id, { content: "笔记", type: "context" });

      store.clear();

      expect(store.list().length).toBe(0);
      expect(store.getNotes(task1.id)).toEqual([]);
    });
  });

  describe("loadTaskStoreState()", () => {
    it("从数据恢复状态", () => {
      const task1 = store.create({ description: "任务1" });
      store.addNote(task1.id, { content: "笔记", type: "context" });
      store.setDependencies(task1.id, ["dep1"]);

      // 序列化并加载
      const serialized = {
        tasks: Array.from(state.tasks.entries()),
        notes: Array.from(state.notes.entries()),
        dependencies: Array.from(state.dependencies.entries()),
      };

      const loadedState = loadTaskStoreState(serialized);
      expect(loadedState.tasks.size).toBe(1);
      expect(loadedState.notes.size).toBe(1);
      expect(loadedState.dependencies.size).toBe(1);
    });
  });

  describe("createTaskStoreState()", () => {
    it("创建空状态", () => {
      const emptyState = createTaskStoreState();
      expect(emptyState.tasks.size).toBe(0);
      expect(emptyState.notes.size).toBe(0);
      expect(emptyState.dependencies.size).toBe(0);
    });
  });
});

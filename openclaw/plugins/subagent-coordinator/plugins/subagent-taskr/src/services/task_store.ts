/**
 * Task Store Service
 *
 * Core service for task CRUD operations, notes management,
 * and dependency tracking.
 */

import type { Task } from "@subagent-coordinator/types";

export interface PersistenceService {
  save(key: string, data: unknown): Promise<void>;
  load(key: string): Promise<unknown | null>;
  delete(key: string): Promise<boolean>;
  exists(key: string): Promise<boolean>;
  list(): Promise<string[]>;
}

export type TaskStatus = "open" | "wip" | "done" | "skipped";

export interface Note {
  id: string;
  taskId: string;
  content: string;
  type: "context" | "finding" | "progress" | "file_list" | "error";
  author?: string;
  createdAt: number;
  updatedAt?: number;
}

export interface TaskFilters {
  status?: TaskStatus;
  parentId?: string | null;
  priority?: Task["priority"];
  tags?: string[];
  createdAfter?: number;
  createdBefore?: number;
}

export interface TaskStoreState {
  tasks: Map<string, Task>;
  notes: Map<string, Note[]>;
  dependencies: Map<string, string[]>;
}

export interface TaskStoreService {
  create(task: Omit<Task, "id"> & { id?: string }): Task;
  get(taskId: string): Task | null;
  update(taskId: string, updates: Partial<Task>): Task | null;
  delete(taskId: string): boolean;
  list(filters?: TaskFilters): Task[];
  addNote(taskId: string, note: Omit<Note, "id" | "taskId" | "createdAt">): string;
  getNotes(taskId: string): Note[];
  updateNote(taskId: string, noteId: string, content: string): boolean;
  deleteNote(taskId: string, noteId: string): boolean;
  setDependencies(taskId: string, dependsOn: string[]): void;
  getDependencies(taskId: string): string[];
  addDependency(taskId: string, dependsOn: string): void;
  removeDependency(taskId: string, dependsOn: string): void;
  getSubtasks(parentId: string): Task[];
  getParent(taskId: string): Task | null;
  getChildren(taskId: string): Task[];
  clear(): void;
}

function generateId(): string {
  return `task_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
}

function generateNoteId(): string {
  return `note_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
}

export function createTaskStore(
  state: TaskStoreState,
  persistence?: PersistenceService
): TaskStoreService {
  async function persist(): Promise<void> {
    if (persistence) {
      await persistence.save("task_store", {
        tasks: Array.from(state.tasks.entries()),
        notes: Array.from(state.notes.entries()),
        dependencies: Array.from(state.dependencies.entries())
      });
    }
  }

  return {
    create(taskInput): Task {
      const task: Task = {
        id: taskInput.id || generateId(),
        description: taskInput.description,
        steps: taskInput.steps,
        files: taskInput.files,
        estimatedDuration: taskInput.estimatedDuration,
        priority: taskInput.priority,
        ...("metadata" in taskInput ? { metadata: (taskInput as any).metadata } : {})
      };

      state.tasks.set(task.id, task);

      // Initialize empty notes array
      if (!state.notes.has(task.id)) {
        state.notes.set(task.id, []);
      }

      persist();
      return task;
    },

    get(taskId): Task | null {
      return state.tasks.get(taskId) || null;
    },

    update(taskId, updates): Task | null {
      const task = state.tasks.get(taskId);
      if (!task) return null;

      const updated = { ...task, ...updates, id: task.id }; // Preserve original id
      state.tasks.set(taskId, updated);
      persist();
      return updated;
    },

    delete(taskId): boolean {
      const deleted = state.tasks.delete(taskId);
      if (deleted) {
        state.notes.delete(taskId);
        state.dependencies.delete(taskId);
        persist();
      }
      return deleted;
    },

    list(filters?: TaskFilters): Task[] {
      let tasks = Array.from(state.tasks.values());

      if (!filters) return tasks;

      if (filters.status !== undefined) {
        const status = filters.status;
        tasks = tasks.filter(t => (t as any).status === status);
      }

      if (filters.parentId !== undefined) {
        tasks = tasks.filter(t => (t as any).parentId === filters.parentId);
      }

      if (filters.priority !== undefined) {
        tasks = tasks.filter(t => t.priority === filters.priority);
      }

      if (filters.tags !== undefined && filters.tags.length > 0) {
        tasks = tasks.filter(t => {
          const taskTags = (t as any).tags || [];
          return filters.tags!.some(tag => taskTags.includes(tag));
        });
      }

      if (filters.createdAfter !== undefined) {
        const createdAt = (t: Task) => (t as any).createdAt || 0;
        tasks = tasks.filter(t => createdAt(t) >= filters.createdAfter!);
      }

      if (filters.createdBefore !== undefined) {
        const createdAt = (t: Task) => (t as any).createdAt || Date.now();
        tasks = tasks.filter(t => createdAt(t) <= filters.createdBefore!);
      }

      return tasks;
    },

    addNote(taskId, noteInput): string {
      if (!state.tasks.has(taskId)) {
        throw new Error(`Task not found: ${taskId}`);
      }

      const note: Note = {
        id: generateNoteId(),
        taskId,
        content: noteInput.content,
        type: noteInput.type,
        author: noteInput.author,
        createdAt: Date.now()
      };

      const notes = state.notes.get(taskId) || [];
      notes.push(note);
      state.notes.set(taskId, notes);
      persist();

      return note.id;
    },

    getNotes(taskId): Note[] {
      return state.notes.get(taskId) || [];
    },

    updateNote(taskId, noteId, content): boolean {
      const notes = state.notes.get(taskId);
      if (!notes) return false;

      const note = notes.find(n => n.id === noteId);
      if (!note) return false;

      note.content = content;
      note.updatedAt = Date.now();
      persist();
      return true;
    },

    deleteNote(taskId, noteId): boolean {
      const notes = state.notes.get(taskId);
      if (!notes) return false;

      const index = notes.findIndex(n => n.id === noteId);
      if (index === -1) return false;

      notes.splice(index, 1);
      persist();
      return true;
    },

    setDependencies(taskId, dependsOn): void {
      state.dependencies.set(taskId, [...dependsOn]);
      persist();
    },

    getDependencies(taskId): string[] {
      return state.dependencies.get(taskId) || [];
    },

    addDependency(taskId, dependsOn): void {
      const deps = state.dependencies.get(taskId) || [];
      if (!deps.includes(dependsOn)) {
        deps.push(dependsOn);
        state.dependencies.set(taskId, deps);
        persist();
      }
    },

    removeDependency(taskId, dependsOn): void {
      const deps = state.dependencies.get(taskId) || [];
      const index = deps.indexOf(dependsOn);
      if (index !== -1) {
        deps.splice(index, 1);
        state.dependencies.set(taskId, deps);
        persist();
      }
    },

    getSubtasks(parentId): Task[] {
      return Array.from(state.tasks.values()).filter(
        t => (t as any).parentId === parentId
      );
    },

    getParent(taskId): Task | null {
      const task = state.tasks.get(taskId);
      if (!task) return null;
      const parentId = (task as any).parentId;
      if (!parentId) return null;
      return state.tasks.get(parentId) || null;
    },

    getChildren(taskId): Task[] {
      return this.getSubtasks(taskId);
    },

    clear(): void {
      state.tasks.clear();
      state.notes.clear();
      state.dependencies.clear();
      persist();
    }
  };
}

export function createTaskStoreState(): TaskStoreState {
  return {
    tasks: new Map(),
    notes: new Map(),
    dependencies: new Map()
  };
}

export function loadTaskStoreState(data: {
  tasks: [string, Task][];
  notes: [string, Note[]][];
  dependencies: [string, string[]][];
}): TaskStoreState {
  return {
    tasks: new Map(data.tasks),
    notes: new Map(data.notes),
    dependencies: new Map(data.dependencies)
  };
}

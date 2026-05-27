/**
 * Memory-Backed Task Store
 *
 * A drop-in TaskStoreService that:
 *   - Keeps hot task data in memory (Map) for fast reads during a session
 *   - Persists each task individually to <workspace>/memory/subagent-tasks/<id>.md
 *   - Loads tasks from those .md files on startup (cross-session persistence)
 *
 * This is NOT a wrapper — it IS a TaskStoreService implementation.
 * Just pass it to tools instead of createTaskStore().
 *
 * Usage in index.ts:
 *   import { createMemoryTaskStore } from "./services/memory_task_store";
 *   const taskStore = createMemoryTaskStore();
 *   // tools call taskStore.create() / taskStore.get() / etc.
 */

import * as fs from "fs";
import * as path from "path";
import {
  type TaskStoreService,
  type TaskFilters,
  type Note,
  type TaskStoreState,
} from "./task_store";
import {
  createMemoryStore,
  createMemoryStoreState,
  type MemoryPersistenceService,
} from "./memory_store";

// Re-export types for consumers
export type { MemoryPersistenceService, MemoryPersistenceConfig } from "./memory_store";

/** Prefix for task files written by this store */
const TASKS_SUBDIR = "memory/subagent-tasks";

function resolveTasksDir(): string {
  const workspace = process.env.OPENCLAW_WORKSPACE
    || path.join(process.env.HOME || "/root", ".openclaw/workspace");
  return path.join(workspace, TASKS_SUBDIR);
}

function generateId(): string {
  return `task_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
}

function generateNoteId(): string {
  return `note_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
}

// ─────────────────────────────────────────────────────────────────────────────
// Serialization helpers
// ─────────────────────────────────────────────────────────────────────────────

type TaskPriority = "low" | "normal" | "high" | "urgent";

interface StoredTask {
  id: string;
  description: string;
  steps?: number;
  files?: string[];
  estimatedDuration?: number;
  priority?: TaskPriority;
  status?: string;
  parentId?: string | null;
  tags?: string[];
  metadata?: Record<string, unknown>;
  createdAt: number;
  updatedAt?: number;
  // Embedded notes and dependencies for cross-session recall
  notes?: Array<{
    id: string;
    content: string;
    type: string;
    author?: string;
    createdAt: number;
    updatedAt?: number;
  }>;
  dependsOn?: string[];
}

/** Serialize a StoredTask into Markdown with JSON code block */
function toMarkdown(task: StoredTask): string {
  const header = `# Task: ${task.id}`;
  const json = JSON.stringify(task, null, 2);
  return `${header}\n\n\`\`\`json\n${json}\n\`\`\`\n`;
}

/** Deserialize a Markdown file to StoredTask. Returns null on parse failure. */
function fromMarkdown(content: string): StoredTask | null {
  const match = content.match(/```json\s*([\s\S]*?)\s*```/);
  if (!match) return null;
  try {
    return JSON.parse(match[1].trim()) as StoredTask;
  } catch {
    return null;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// MemoryTaskStore — implements TaskStoreService
// ─────────────────────────────────────────────────────────────────────────────

export function createMemoryTaskStore(
  opts?: { warnIndexDelay?: boolean }
): TaskStoreService & { getMemoryStore: () => MemoryPersistenceService } {
  const memoryState = createMemoryStoreState({ warnIndexDelay: opts?.warnIndexDelay ?? false });
  const memory = createMemoryStore(memoryState);

  // In-memory state (hot working set)
  const tasks = new Map<string, StoredTask>();
  const notes = new Map<string, StoredTask["notes"]>();
  const dependencies = new Map<string, string[]>();

  // Ensure directory exists
  function ensureDir(): void {
    const dir = resolveTasksDir();
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }
  }

  // Persist a single task to .md file
  function persistTask(task: StoredTask): void {
    ensureDir();
    const filePath = path.join(resolveTasksDir(), `${task.id}.md`);
    fs.writeFileSync(filePath, toMarkdown(task), "utf8");
  }

  // Load all tasks from .md files into memory
  async function bootstrap(): Promise<void> {
    const dir = resolveTasksDir();
    if (!fs.existsSync(dir)) return;

    const files = fs.readdirSync(dir).filter(f => f.endsWith(".md"));
    for (const file of files) {
      const taskId = file.replace(/\.md$/, "");
      const filePath = path.join(dir, file);
      const content = fs.readFileSync(filePath, "utf8");
      const task = fromMarkdown(content);
      if (!task) continue;

      tasks.set(task.id, task);

      // Restore embedded notes
      if (task.notes && task.notes.length > 0) {
        notes.set(task.id, task.notes);
      }

      // Restore embedded dependencies
      if (task.dependsOn && task.dependsOn.length > 0) {
        dependencies.set(task.id, task.dependsOn);
      }
    }
  }

  // Bootstrap on construction (synchronous — file I/O is fast for small task counts)
  try {
    bootstrap();
  } catch (err) {
    console.warn("[taskr:memory] Bootstrap failed (tasks will start empty):", err);
  }

  // ── TaskStoreService implementation ───────────────────────────────────────

  const store: TaskStoreService = {
    create(taskInput): StoredTask {
      const task: StoredTask = {
        id: taskInput.id || generateId(),
        description: taskInput.description,
        steps: taskInput.steps,
        files: taskInput.files,
        estimatedDuration: taskInput.estimatedDuration,
        priority: taskInput.priority,
        status: "open",
        createdAt: Date.now(),
        updatedAt: Date.now(),
        ...(taskInput as any).metadata && { metadata: (taskInput as any).metadata },
      };

      tasks.set(task.id, task);
      notes.set(task.id, []);
      dependencies.set(task.id, []);
      persistTask(task);
      return task;
    },

    get(taskId): StoredTask | null {
      return tasks.get(taskId) || null;
    },

    update(taskId, updates): StoredTask | null {
      const task = tasks.get(taskId);
      if (!task) return null;

      const updated: StoredTask = {
        ...task,
        ...updates,
        id: task.id,       // preserve original id
        createdAt: task.createdAt, // preserve original createdAt
        updatedAt: Date.now(),
        // Merge notes/dependencies from in-memory back into task
        notes: notes.get(taskId) || [],
        dependsOn: dependencies.get(taskId) || [],
      };

      tasks.set(taskId, updated);
      persistTask(updated);
      return updated;
    },

    delete(taskId): boolean {
      if (!tasks.has(taskId)) return false;
      tasks.delete(taskId);
      notes.delete(taskId);
      dependencies.delete(taskId);

      // Delete the .md file
      const filePath = path.join(resolveTasksDir(), `${taskId}.md`);
      if (fs.existsSync(filePath)) {
        fs.unlinkSync(filePath);
      }
      return true;
    },

    list(filters?: TaskFilters): StoredTask[] {
      let result = Array.from(tasks.values());

      if (!filters) return result;

      if (filters.status !== undefined) {
        result = result.filter(t => t.status === filters.status);
      }
      if (filters.parentId !== undefined) {
        result = result.filter(t => t.parentId === filters.parentId);
      }
      if (filters.priority !== undefined) {
        result = result.filter(t => t.priority === filters.priority);
      }
      if (filters.tags !== undefined && filters.tags.length > 0) {
        result = result.filter(t =>
          filters.tags!.some(tag => (t.tags || []).includes(tag))
        );
      }
      if (filters.createdAfter !== undefined) {
        result = result.filter(t => (t.createdAt || 0) >= filters.createdAfter!);
      }
      if (filters.createdBefore !== undefined) {
        result = result.filter(t => (t.createdAt || 0) <= filters.createdBefore!);
      }

      return result;
    },

    addNote(taskId, noteInput): string {
      if (!tasks.has(taskId)) throw new Error(`Task not found: ${taskId}`);

      const note = {
        id: generateNoteId(),
        taskId,
        content: noteInput.content,
        type: noteInput.type,
        author: noteInput.author,
        createdAt: Date.now(),
      };

      const taskNotes = notes.get(taskId) || [];
      taskNotes.push(note);
      notes.set(taskId, taskNotes);

      // Update the stored task file with embedded notes
      const task = tasks.get(taskId)!;
      const updated: StoredTask = { ...task, notes: taskNotes, updatedAt: Date.now() };
      tasks.set(taskId, updated);
      persistTask(updated);

      return note.id;
    },

    getNotes(taskId): Note[] {
      return (notes.get(taskId) || []) as Note[];
    },

    updateNote(taskId, noteId, content): boolean {
      const taskNotes = notes.get(taskId);
      if (!taskNotes) return false;
      const note = taskNotes.find(n => n.id === noteId);
      if (!note) return false;
      note.content = content;
      note.updatedAt = Date.now();

      // Persist with updated notes
      const task = tasks.get(taskId)!;
      const updated: StoredTask = { ...task, notes: taskNotes, updatedAt: Date.now() };
      tasks.set(taskId, updated);
      persistTask(updated);
      return true;
    },

    deleteNote(taskId, noteId): boolean {
      const taskNotes = notes.get(taskId);
      if (!taskNotes) return false;
      const idx = taskNotes.findIndex(n => n.id === noteId);
      if (idx === -1) return false;
      taskNotes.splice(idx, 1);
      notes.set(taskId, taskNotes);

      const task = tasks.get(taskId)!;
      const updated: StoredTask = { ...task, notes: taskNotes, updatedAt: Date.now() };
      tasks.set(taskId, updated);
      persistTask(updated);
      return true;
    },

    setDependencies(taskId, dependsOn): void {
      dependencies.set(taskId, [...dependsOn]);
      const task = tasks.get(taskId);
      if (task) {
        const updated: StoredTask = { ...task, dependsOn, updatedAt: Date.now() };
        tasks.set(taskId, updated);
        persistTask(updated);
      }
    },

    getDependencies(taskId): string[] {
      return dependencies.get(taskId) || [];
    },

    addDependency(taskId, dependsOn): void {
      const deps = dependencies.get(taskId) || [];
      if (!deps.includes(dependsOn)) {
        deps.push(dependsOn);
        this.setDependencies(taskId, deps);
      }
    },

    removeDependency(taskId, dependsOn): void {
      const deps = (dependencies.get(taskId) || []).filter(d => d !== dependsOn);
      this.setDependencies(taskId, deps);
    },

    getSubtasks(parentId): StoredTask[] {
      return Array.from(tasks.values()).filter(t => t.parentId === parentId);
    },

    getParent(taskId): StoredTask | null {
      const task = tasks.get(taskId);
      if (!task || !task.parentId) return null;
      return tasks.get(task.parentId) || null;
    },

    getChildren(taskId): StoredTask[] {
      return this.getSubtasks(taskId);
    },

    clear(): void {
      const dir = resolveTasksDir();
      if (fs.existsSync(dir)) {
        for (const file of fs.readdirSync(dir).filter(f => f.endsWith(".md"))) {
          fs.unlinkSync(path.join(dir, file));
        }
      }
      tasks.clear();
      notes.clear();
      dependencies.clear();
    },
  };

  return {
    ...store,
    getMemoryStore: () => memory,
  };
}

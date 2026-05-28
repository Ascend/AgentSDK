/**
 * create_task Tool
 *
 * Creates a new task in the task store.
 * Supports hierarchical task structure with parent-child relationships.
 */

import type { Task } from "@subagent-coordinator/types";
import type { TaskStoreService } from "../services/task_store";

export interface CreateTaskInput {
  description: string;
  parentId?: string;
  priority?: "low" | "normal" | "high" | "urgent";
  estimatedDuration?: number; // seconds
  metadata?: Record<string, unknown>;
  steps?: number;
  files?: string[];
}

export interface CreateTaskOutput {
  task: Task;
  parent?: Task;
  createdAt: number;
}

export function createCreateTaskTool(
  taskStore: TaskStoreService
) {
  return async (input: CreateTaskInput): Promise<CreateTaskOutput> => {
    // Validate input
    if (!input.description || input.description.trim().length === 0) {
      throw new Error("Task description is required");
    }

    // Check parent exists if specified
    let parent: Task | undefined;
    if (input.parentId) {
      parent = taskStore.get(input.parentId);
      if (!parent) {
        throw new Error(`Parent task not found: ${input.parentId}`);
      }
    }

    // Create task with metadata
    const taskData: Task & { parentId?: string; status?: string; createdAt?: number } = {
      description: input.description.trim(),
      priority: input.priority,
      estimatedDuration: input.estimatedDuration,
      steps: input.steps,
      files: input.files,
      metadata: input.metadata,
      status: "open",
      createdAt: Date.now()
    };

    if (input.parentId) {
      taskData.parentId = input.parentId;
    }

    const task = taskStore.create(taskData);

    // Set parent dependency if parent specified
    if (input.parentId) {
      taskStore.addDependency(task.id, input.parentId);
    }

    return {
      task,
      parent,
      createdAt: taskData.createdAt!
    };
  };
}

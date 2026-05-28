/**
 * get_task Tool
 *
 * Retrieves a task by ID with all its details including notes and dependencies.
 */

import type { Task } from "@subagent-coordinator/types";
import type { TaskStoreService } from "../services/task_store";
import type { TaskGraphService } from "../services/task_graph";

export interface GetTaskInput {
  taskId: string;
  includeNotes?: boolean;
  includeDependencies?: boolean;
  includeChildren?: boolean;
  includePath?: boolean;
}

export interface GetTaskOutput {
  task: Task;
  notes?: Array<{
    id: string;
    content: string;
    type: string;
    author?: string;
    createdAt: number;
    updatedAt?: number;
  }>;
  dependencies?: string[];
  children?: Task[];
  path?: string[];
  parent?: Task | null;
}

export function createGetTaskTool(
  taskStore: TaskStoreService,
  taskGraph?: TaskGraphService
) {
  return async (input: GetTaskInput): Promise<GetTaskOutput> => {
    const { taskId, includeNotes, includeDependencies, includeChildren, includePath } = input;

    const task = taskStore.get(taskId);
    if (!task) {
      throw new Error(`Task not found: ${taskId}`);
    }

    const result: GetTaskOutput = { task };

    if (includeNotes) {
      result.notes = taskStore.getNotes(taskId);
    }

    if (includeDependencies) {
      result.dependencies = taskStore.getDependencies(taskId);
    }

    if (includeChildren) {
      result.children = taskStore.getChildren(taskId);
    }

    if (includePath) {
      result.parent = taskStore.getParent(taskId);
      if (taskGraph) {
        result.path = taskGraph.getTaskPath(taskId);
      }
    }

    return result;
  };
}

/**
 * list_tasks Tool
 *
 * Lists tasks with optional filtering and sorting.
 * Supports complex filter criteria and pagination.
 */

import type { Task } from "@subagent-coordinator/types";
import type { TaskStoreService, TaskFilters } from "../services/task_store";
import type { TaskGraphService } from "../services/task_graph";

export interface ListTasksInput {
  status?: "open" | "wip" | "done" | "skipped";
  parentId?: string | null; // null = root tasks only
  priority?: "low" | "normal" | "high" | "urgent";
  tags?: string[];
  createdAfter?: number;
  createdBefore?: number;
  sortBy?: "createdAt" | "updatedAt" | "priority" | "estimatedDuration";
  sortOrder?: "asc" | "desc";
  limit?: number;
  offset?: number;
}

export interface ListTasksOutput {
  tasks: Task[];
  total: number;
  hasMore: boolean;
}

export function createListTasksTool(
  taskStore: TaskStoreService,
  taskGraph?: TaskGraphService
) {
  return async (input: ListTasksInput): Promise<ListTasksOutput> => {
    const {
      status,
      parentId,
      priority,
      tags,
      createdAfter,
      createdBefore,
      sortBy,
      sortOrder = "desc",
      limit = 50,
      offset = 0
    } = input;

    // Build filters
    const filters: TaskFilters = {};

    if (status !== undefined) {
      filters.status = status;
    }

    if (parentId !== undefined) {
      filters.parentId = parentId;
    }

    if (priority !== undefined) {
      filters.priority = priority;
    }

    if (tags !== undefined && tags.length > 0) {
      filters.tags = tags;
    }

    if (createdAfter !== undefined) {
      filters.createdAfter = createdAfter;
    }

    if (createdBefore !== undefined) {
      filters.createdBefore = createdBefore;
    }

    // Get filtered tasks
    let tasks = taskStore.list(filters);

    // Sort
    if (sortBy) {
      tasks.sort((a, b) => {
        let aVal: number | string | undefined;
        let bVal: number | string | undefined;

        switch (sortBy) {
          case "createdAt":
            aVal = (a as any).createdAt || 0;
            bVal = (b as any).createdAt || 0;
            break;
          case "updatedAt":
            aVal = (a as any).updatedAt || (a as any).createdAt || 0;
            bVal = (b as any).updatedAt || (b as any).createdAt || 0;
            break;
          case "priority":
            const priorityOrder = { urgent: 4, high: 3, normal: 2, low: 1 };
            aVal = priorityOrder[a.priority || "normal"];
            bVal = priorityOrder[b.priority || "normal"];
            break;
          case "estimatedDuration":
            aVal = a.estimatedDuration || 0;
            bVal = b.estimatedDuration || 0;
            break;
        }

        if (aVal === undefined || bVal === undefined) return 0;

        if (sortOrder === "asc") {
          return aVal > bVal ? 1 : -1;
        } else {
          return aVal < bVal ? 1 : -1;
        }
      });
    }

    const total = tasks.length;

    // Apply pagination
    tasks = tasks.slice(offset, offset + limit);

    // Add additional info if task graph available
    if (taskGraph) {
      const cycles = taskGraph.detectCycles();
      const hasCycles = cycles.length > 0;

      // Attach executable status
      const executableTasks = taskGraph.getExecutableTasks();
      const executableIds = new Set(executableTasks.map(t => t.id));

      tasks = tasks.map(task => ({
        ...task,
        ...((hasCycles || executableIds.has(task.id)) ? {
          _taskr_meta: {
            executable: executableIds.has(task.id),
            hasCycles
          }
        } : {})
      })) as Task[];
    }

    return {
      tasks,
      total,
      hasMore: offset + limit < total
    };
  };
}

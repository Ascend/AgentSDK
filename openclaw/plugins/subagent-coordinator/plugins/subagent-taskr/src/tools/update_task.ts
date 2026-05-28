/**
 * update_task Tool
 *
 * Updates an existing task's properties and status.
 * Supports status transitions: open → wip → done/skipped
 */

import type { Task } from "@subagent-coordinator/types";
import type { TaskStoreService } from "../services/task_store";

export type TaskStatus = "open" | "wip" | "done" | "skipped";

export interface UpdateTaskInput {
  taskId: string;
  description?: string;
  status?: TaskStatus;
  priority?: "low" | "normal" | "high" | "urgent";
  estimatedDuration?: number;
  steps?: number;
  files?: string[];
  metadata?: Record<string, unknown>;
}

export interface UpdateTaskOutput {
  task: Task;
  previousStatus?: TaskStatus;
  changedFields: string[];
}

const VALID_STATUS_TRANSITIONS: Record<TaskStatus, TaskStatus[]> = {
  "open": ["wip", "skipped"],
  "wip": ["done", "skipped", "open"],
  "done": ["wip", "open"], // Allow reopening
  "skipped": ["open", "wip"] // Allow reactivating
};

export function createUpdateTaskTool(
  taskStore: TaskStoreService
) {
  return async (input: UpdateTaskInput): Promise<UpdateTaskOutput> => {
    const { taskId, ...updates } = input;

    const task = taskStore.get(taskId);
    if (!task) {
      throw new Error(`Task not found: ${taskId}`);
    }

    const currentStatus = (task as any).status as TaskStatus || "open";
    const changedFields: string[] = [];

    // Validate status transition
    if (updates.status && updates.status !== currentStatus) {
      const allowed = VALID_STATUS_TRANSITIONS[currentStatus];
      if (!allowed.includes(updates.status)) {
        throw new Error(
          `Invalid status transition: ${currentStatus} → ${updates.status}. ` +
          `Allowed: ${allowed.join(", ") || "none"}`
        );
      }
    }

    // Build update object
    const updateData: Partial<Task & { status?: TaskStatus }> = {};

    if (updates.description !== undefined) {
      if (updates.description.trim().length === 0) {
        throw new Error("Task description cannot be empty");
      }
      updateData.description = updates.description.trim();
      changedFields.push("description");
    }

    if (updates.status !== undefined) {
      (updateData as any).status = updates.status;
      changedFields.push("status");
    }

    if (updates.priority !== undefined) {
      updateData.priority = updates.priority;
      changedFields.push("priority");
    }

    if (updates.estimatedDuration !== undefined) {
      updateData.estimatedDuration = updates.estimatedDuration;
      changedFields.push("estimatedDuration");
    }

    if (updates.steps !== undefined) {
      updateData.steps = updates.steps;
      changedFields.push("steps");
    }

    if (updates.files !== undefined) {
      updateData.files = updates.files;
      changedFields.push("files");
    }

    if (updates.metadata !== undefined) {
      updateData.metadata = { ...((task as any).metadata || {}), ...updates.metadata };
      changedFields.push("metadata");
    }

    if (changedFields.length === 0) {
      return { task, changedFields: [] };
    }

    const updatedTask = taskStore.update(taskId, updateData);
    if (!updatedTask) {
      throw new Error(`Failed to update task: ${taskId}`);
    }

    // Add progress note if status changed
    if (updates.status && updates.status !== currentStatus) {
      taskStore.addNote(taskId, {
        content: `Status changed from ${currentStatus} to ${updates.status}`,
        type: "progress",
        author: "taskr-plugin"
      });
    }

    return {
      task: updatedTask,
      previousStatus: currentStatus,
      changedFields
    };
  };
}

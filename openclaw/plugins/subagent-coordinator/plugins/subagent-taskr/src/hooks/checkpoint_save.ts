/**
 * checkpoint_save Hook
 *
 * Triggered when a checkpoint save is requested.
 * Persists task state for later recovery.
 */

import type { CheckpointSaveEvent, CheckpointData, ExecutionResult } from "@subagent-coordinator/types";
import type { TaskStoreService } from "../services/task_store";

export interface CheckpointSaveResult {
  saved: boolean;
  checkpointId?: string;
  error?: string;
}

export async function handleCheckpointSave(
  event: CheckpointSaveEvent,
  taskStore?: TaskStoreService
): Promise<CheckpointSaveResult> {
  const { checkpoint, timestamp } = event;

  try {
    // Validate checkpoint data
    if (!checkpoint.taskId) {
      return {
        saved: false,
        error: "Missing taskId in checkpoint"
      };
    }

    if (!checkpoint.subtasks || checkpoint.subtasks.length === 0) {
      return {
        saved: false,
        error: "No subtasks to save in checkpoint"
      };
    }

    // Generate stable checkpoint ID
    const checkpointId = generateCheckpointId(checkpoint.taskId, timestamp);

    // Add note to task tracking checkpoint creation
    if (taskStore) {
      const noteContent = `Checkpoint saved: ${checkpoint.completedSubtasks.length}/${checkpoint.subtasks.length} subtasks completed`;

      try {
        taskStore.addNote(checkpoint.taskId, {
          content: noteContent,
          type: "progress",
          author: "taskr-plugin"
        });
      } catch (noteError) {
        // Non-critical - checkpoint save can proceed without note
        console.warn("[taskr] Failed to add checkpoint note:", noteError);
      }
    }

    return {
      saved: true,
      checkpointId
    };
  } catch (error) {
    return {
      saved: false,
      error: error instanceof Error ? error.message : "Unknown error"
    };
  }
}

function generateCheckpointId(taskId: string, timestamp: number): string {
  return `cp_${taskId}_${timestamp}`;
}

/**
 * Create a checkpoint from current task state
 */
export function createCheckpointFromState(
  taskId: string,
  subtasks: { id: string; description: string; estimatedDuration?: number }[],
  completedSubtaskIds: string[],
  results: Map<string, ExecutionResult>
): CheckpointData {
  return {
    taskId,
    subtasks: subtasks.map(s => ({
      id: s.id,
      description: s.description,
      estimatedDuration: s.estimatedDuration
    })),
    completedSubtasks: completedSubtaskIds,
    results,
    timestamp: Date.now()
  };
}

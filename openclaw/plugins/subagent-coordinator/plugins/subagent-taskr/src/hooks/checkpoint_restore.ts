/**
 * checkpoint_restore Hook
 *
 * Triggered when a checkpoint restore is requested.
 * Provides recovery suggestions based on checkpoint data.
 */

import type { CheckpointRestoreEvent, CheckpointData } from "@subagent-coordinator/types";

export interface CheckpointRestoreResult {
  canRestore: boolean;
  checkpointData?: CheckpointData;
  remainingSubtasks?: string[];
  progress?: {
    completed: number;
    total: number;
    percentage: number;
  };
  warnings?: string[];
  error?: string;
}

export async function handleCheckpointRestore(
  event: CheckpointRestoreEvent,
  availableCheckpoints: Map<string, CheckpointData>
): Promise<CheckpointRestoreResult> {
  const { checkpointId, taskId } = event;

  // Try to find the checkpoint
  let checkpoint: CheckpointData | null = null;

  // First try exact ID match
  if (checkpointId && availableCheckpoints.has(checkpointId)) {
    checkpoint = availableCheckpoints.get(checkpointId)!;
  }

  // If no exact match, try to find latest checkpoint for taskId
  if (!checkpoint && taskId) {
    const matchingCheckpoints: { id: string; cp: CheckpointData }[] = [];

    for (const [id, cp] of availableCheckpoints.entries()) {
      if (cp.taskId === taskId) {
        matchingCheckpoints.push({ id, cp });
      }
    }

    if (matchingCheckpoints.length > 0) {
      // Sort by timestamp and get the latest
      matchingCheckpoints.sort((a, b) => b.cp.timestamp - a.cp.timestamp);
      checkpoint = matchingCheckpoints[0].cp;
    }
  }

  // No checkpoint found
  if (!checkpoint) {
    return {
      canRestore: false,
      error: `Checkpoint not found: ${checkpointId || `for task ${taskId}`}`
    };
  }

  // Analyze checkpoint for restore feasibility
  const warnings: string[] = [];

  // Check if checkpoint is stale (older than 24 hours)
  const ageMs = Date.now() - checkpoint.timestamp;
  const ageHours = ageMs / (1000 * 60 * 60);

  if (ageHours > 24) {
    warnings.push(`Checkpoint is ${ageHours.toFixed(1)} hours old - task state may have changed significantly`);
  }

  // Check for completed subtasks
  const completed = checkpoint.completedSubtasks.length;
  const total = checkpoint.subtasks.length;

  if (completed === 0) {
    warnings.push("No subtasks were completed in this checkpoint");
  }

  if (completed === total) {
    return {
      canRestore: false,
      checkpointData: checkpoint,
      progress: { completed, total, percentage: 100 },
      warnings: ["All subtasks already completed - nothing to restore"]
    };
  }

  // Calculate remaining subtasks
  const remainingSubtasks = checkpoint.subtasks
    .filter(s => !checkpoint.completedSubtasks.includes(s.id))
    .map(s => s.id);

  return {
    canRestore: true,
    checkpointData: checkpoint,
    remainingSubtasks,
    progress: {
      completed,
      total,
      percentage: Math.round((completed / total) * 100)
    },
    warnings: warnings.length > 0 ? warnings : undefined
  };
}

/**
 * Validate that a checkpoint can be safely restored
 */
export function validateCheckpointRestore(
  checkpoint: CheckpointData,
  currentSubtasks: { id: string; description: string; dependsOn?: string[] }[]
): { valid: boolean; errors: string[] } {
  const errors: string[] = [];

  // Check that all completed subtasks exist in current subtask list
  const currentSubtaskIds = new Set(currentSubtasks.map(s => s.id));
  for (const completedId of checkpoint.completedSubtasks) {
    if (!currentSubtaskIds.has(completedId)) {
      errors.push(`Completed subtask ${completedId} no longer exists in current task`);
    }
  }

  // Check dependency satisfaction for remaining subtasks
  const completedSet = new Set(checkpoint.completedSubtasks);
  const remaining = currentSubtasks.filter(s => !completedSet.has(s.id));

  for (const subtask of remaining) {
    if (subtask.dependsOn) {
      const unsatisfied = subtask.dependsOn.filter(depId => !completedSet.has(depId));
      if (unsatisfied.length > 0) {
        errors.push(`Subtask ${subtask.id} has unsatisfied dependencies: ${unsatisfied.join(", ")}`);
      }
    }
  }

  return {
    valid: errors.length === 0,
    errors
  };
}

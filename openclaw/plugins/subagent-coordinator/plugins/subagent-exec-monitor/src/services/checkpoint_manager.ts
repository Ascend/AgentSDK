/**
 * Checkpoint Manager Service
 *
 * Provides checkpoint save/restore functionality for task execution.
 * Enables resuming failed tasks and tracking execution progress.
 */

import type { CheckpointData, ExecutionResult, Subtask } from "@subagent-coordinator/types";

export interface CheckpointManagerState {
  checkpoints: Map<string, CheckpointData>;
  executionHistory: Map<string, ExecutionResult[]>;
}

export interface CheckpointManagerService {
  save(checkpoint: CheckpointData): Promise<string>;
  restore(checkpointId: string): Promise<CheckpointData | null>;
  list(taskId: string): Promise<string[]>;
  saveExecutionResult(taskId: string, result: ExecutionResult): Promise<void>;
  getExecutionHistory(taskId: string): Promise<ExecutionResult[]>;
  deleteOldCheckpoints(maxAgeMs: number): Promise<number>;
}

export function createCheckpointManager(
  state: CheckpointManagerState
): CheckpointManagerService {
  return {
    async save(checkpoint: CheckpointData): Promise<string> {
      const id = `checkpoint_${checkpoint.taskId}_${checkpoint.timestamp}`;

      state.checkpoints.set(id, {
        ...checkpoint,
        results: new Map(checkpoint.results) // Ensure Map is properly copied
      });

      return id;
    },

    async restore(checkpointId: string): Promise<CheckpointData | null> {
      const checkpoint = state.checkpoints.get(checkpointId);

      if (!checkpoint) {
        return null;
      }

      return {
        ...checkpoint,
        results: new Map(checkpoint.results)
      };
    },

    async list(taskId: string): Promise<string[]> {
      const ids: string[] = [];

      for (const [id, cp] of state.checkpoints.entries()) {
        if (cp.taskId === taskId) {
          ids.push(id);
        }
      }

      return ids.sort();
    },

    async saveExecutionResult(taskId: string, result: ExecutionResult): Promise<void> {
      if (!state.executionHistory.has(taskId)) {
        state.executionHistory.set(taskId, []);
      }

      state.executionHistory.get(taskId)!.push(result);
    },

    async getExecutionHistory(taskId: string): Promise<ExecutionResult[]> {
      return state.executionHistory.get(taskId) || [];
    },

    async deleteOldCheckpoints(maxAgeMs: number): Promise<number> {
      const now = Date.now();
      let deletedCount = 0;

      for (const [id, checkpoint] of state.checkpoints.entries()) {
        if (now - checkpoint.timestamp > maxAgeMs) {
          state.checkpoints.delete(id);
          deletedCount++;
        }
      }

      return deletedCount;
    }
  };
}

// Helper to create a checkpoint from task and current state
export function createCheckpoint(
  taskId: string,
  subtasks: Subtask[],
  completedSubtasks: string[],
  results: Map<string, ExecutionResult>
): CheckpointData {
  return {
    taskId,
    subtasks,
    completedSubtasks,
    results: new Map(results),
    timestamp: Date.now()
  };
}

// Helper to determine if a task can be resumed from checkpoint
export function canResumeFromCheckpoint(
  checkpoint: CheckpointData,
  allSubtasks: Subtask[]
): { canResume: boolean; remainingSubtasks: Subtask[]; reason?: string } {
  // Check if all completed subtasks are actually in the subtask list
  const subtaskIds = new Set(allSubtasks.map(s => s.id));
  for (const completedId of checkpoint.completedSubtasks) {
    if (!subtaskIds.has(completedId)) {
      return {
        canResume: false,
        remainingSubtasks: allSubtasks,
        reason: `Completed subtask ${completedId} not found in current task definition`
      };
    }
  }

  // Determine remaining subtasks
  const remainingSubtasks = allSubtasks.filter(
    s => !checkpoint.completedSubtasks.includes(s.id)
  );

  if (remainingSubtasks.length === 0) {
    return {
      canResume: false,
      remainingSubtasks: [],
      reason: "All subtasks already completed"
    };
  }

  // Check dependency satisfaction
  for (const subtask of remainingSubtasks) {
    if (subtask.dependsOn) {
      const unsatisfiedDeps = subtask.dependsOn.filter(
        depId => !checkpoint.completedSubtasks.includes(depId)
      );
      if (unsatisfiedDeps.length > 0) {
        return {
          canResume: false,
          remainingSubtasks: [],
          reason: `Cannot resume: subtask ${subtask.id} has unsatisfied dependencies: ${unsatisfiedDeps.join(", ")}`
        };
      }
    }
  }

  return {
    canResume: true,
    remainingSubtasks
  };
}

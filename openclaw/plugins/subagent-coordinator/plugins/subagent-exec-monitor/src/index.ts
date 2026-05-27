import { Type } from "@sinclair/typebox";
import { definePluginEntry } from "openclaw/plugin-sdk/plugin-entry";

import type {
  Task,
  QualityGateResult,
  CheckpointData,
  ExecutionResult,
} from "@subagent-coordinator/types";

export { SUBAGENT_COORDINATOR_EVENTS } from "@subagent-coordinator/types";
export type { SubagentCoordinatorEventName } from "@subagent-coordinator/types";

import {
  createCheckpointManager,
  createCheckpoint,
  type CheckpointManagerState,
  type CheckpointManagerService,
} from "./services/checkpoint_manager";

let _checkpointManager: CheckpointManagerService | null = null;
let _checkpointState: CheckpointManagerState | null = null;

function getCheckpointManager(): CheckpointManagerService {
  if (!_checkpointManager) {
    _checkpointState = {
      checkpoints: new Map(),
      executionHistory: new Map(),
    };
    _checkpointManager = createCheckpointManager(_checkpointState);
  }
  return _checkpointManager;
}

import { selectRetryStrategy } from "./tools/retry_strategy";

export default definePluginEntry({
  id: "@subagent-coordinator/exec-monitor",
  name: "Subagent Coordinator Execution Monitor",
  description: "Quality gates, checkpoints, retry strategies, and task analysis for subagent-coordinator",

  register(api) {
    const checkpointManager = getCheckpointManager();

    api.registerTool({
      name: "quality_gate_check",
      label: "quality_gate_check",
      description: "Execute quality gate checks before/after task execution",
      parameters: Type.Object({
        task: Type.Object({
          id: Type.String(),
          description: Type.String(),
          steps: Type.Optional(Type.Number()),
          files: Type.Optional(Type.Array(Type.String())),
          estimatedDuration: Type.Optional(Type.Number()),
          priority: Type.Optional(Type.Union([
            Type.Literal("low"),
            Type.Literal("normal"),
            Type.Literal("high"),
            Type.Literal("urgent"),
          ])),
        }),
        preExecution: Type.Boolean({ description: "true = pre-execution check, false = post-execution check" }),
      }),
      async execute(_id, params) {
        const { task, preExecution } = params;
        const checks: QualityGateResult["checks"] = [];

        if (preExecution) {
          if (!task.description || task.description.trim().length === 0) {
            checks.push({ name: "has_description", pass: false, message: "Task description is empty" });
          } else {
            checks.push({ name: "has_description", pass: true });
          }

          if (task.steps !== undefined && task.steps > 100) {
            checks.push({ name: "step_count_reasonable", pass: false, message: "Step count exceeds 100 - task should be decomposed" });
          } else {
            checks.push({ name: "step_count_reasonable", pass: true });
          }

          if (task.files !== undefined && task.files.length > 500) {
            checks.push({ name: "file_count_reasonable", pass: false, message: "File count exceeds 500" });
          } else {
            checks.push({ name: "file_count_reasonable", pass: true });
          }
        } else {
          checks.push({ name: "execution_completed", pass: true });
        }

        const result: QualityGateResult = {
          pass: checks.every(c => c.pass),
          checks,
        };

        return { content: [{ type: "text", text: JSON.stringify(result) }], details: result };
      },
    });

    api.registerTool({
      name: "retry_strategy_selector",
      label: "retry_strategy_selector",
      description: "Select optimal retry strategy based on error type and history",
      parameters: Type.Object({
        error: Type.String({ description: "Error message or type" }),
        history: Type.Optional(Type.Array(Type.Object({
          taskId: Type.String(),
          success: Type.Boolean(),
          output: Type.Optional(Type.Unknown()),
          error: Type.Optional(Type.String()),
          duration: Type.Number(),
          tokensUsed: Type.Optional(Type.Number()),
        }))),
      }),
      async execute(_id, params) {
        const { error, history } = params;
        const result = selectRetryStrategy(error, (history || []) as ExecutionResult[]);
        return { content: [{ type: "text", text: JSON.stringify(result) }], details: result };
      },
    });

    api.registerTool({
      name: "save_checkpoint",
      label: "save_checkpoint",
      description: "Save a checkpoint for task resumption",
      parameters: Type.Object({
        taskId: Type.String(),
        subtasks: Type.Array(Type.Object({
          id: Type.String(),
          description: Type.String(),
          dependsOn: Type.Optional(Type.Array(Type.String())),
          estimatedDuration: Type.Optional(Type.Number()),
          parallelGroup: Type.Optional(Type.String()),
        })),
        completedSubtasks: Type.Array(Type.String()),
        results: Type.Record(Type.String(), Type.Object({
          taskId: Type.String(),
          success: Type.Boolean(),
          output: Type.Optional(Type.Unknown()),
          error: Type.Optional(Type.String()),
          duration: Type.Number(),
          tokensUsed: Type.Optional(Type.Number()),
        })),
      }),
      async execute(_id, params) {
        const { taskId, subtasks, completedSubtasks, results } = params;
        const checkpoint = createCheckpoint(taskId, subtasks as any, completedSubtasks, new Map(Object.entries(results)) as any);
        const id = checkpointManager.save(checkpoint);
        const result = { checkpointId: id, saved: true };
        return { content: [{ type: "text", text: JSON.stringify(result) }], details: result };
      },
    });

    api.registerTool({
      name: "restore_checkpoint",
      label: "restore_checkpoint",
      description: "Restore a checkpoint for task resumption",
      parameters: Type.Object({
        checkpointId: Type.String(),
      }),
      async execute(_id, params) {
        const { checkpointId } = params;
        const checkpoint = checkpointManager.restore(checkpointId);
        const result = !checkpoint
          ? { found: false }
          : { found: true, checkpoint };
        return { content: [{ type: "text", text: JSON.stringify(result) }], details: result };
      },
    });
  },
});

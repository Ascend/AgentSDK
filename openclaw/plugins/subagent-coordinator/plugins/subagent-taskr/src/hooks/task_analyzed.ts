/**
 * task_analyzed Hook
 *
 * Triggered after task analysis is complete.
 * Records task analysis results and provides suggestions.
 */

import type { TaskAnalyzedEvent, ComplexityScore, OperatorLevel } from "@subagent-coordinator/types";
import type { TaskStoreService } from "../services/task_store";

export interface TaskAnalysisRecord {
  taskId: string;
  analysisTimestamp: number;
  complexity: ComplexityScore;
  operatorLevel: OperatorLevel;
  decompositionTriggered: boolean;
  suggestedStrategies?: string[];
}

export interface TaskAnalyzeResult {
  recorded: boolean;
  recordId?: string;
  suggestions?: {
    type: "decomposition" | "priority" | "delegation";
    message: string;
    action?: string;
  }[];
  error?: string;
}

export async function handleTaskAnalyzed(
  event: TaskAnalyzedEvent,
  taskStore?: TaskStoreService
): Promise<TaskAnalyzeResult> {
  const { task, complexity, operatorLevel, decompositionTriggered, timestamp } = event;

  try {
    const record: TaskAnalysisRecord = {
      taskId: task.id,
      analysisTimestamp: timestamp,
      complexity,
      operatorLevel,
      decompositionTriggered
    };

    // Add analysis as a note to the task
    if (taskStore) {
      const noteContent = formatAnalysisNote(record);
      const noteId = taskStore.addNote(task.id, {
        content: noteContent,
        type: "context",
        author: "taskr-plugin"
      });

      return {
        recorded: true,
        recordId: noteId,
        suggestions: generateSuggestions(task, complexity, operatorLevel, decompositionTriggered)
      };
    }

    return {
      recorded: false,
      error: "Task store not available for recording"
    };
  } catch (error) {
    return {
      recorded: false,
      error: error instanceof Error ? error.message : "Unknown error"
    };
  }
}

function formatAnalysisNote(record: TaskAnalysisRecord): string {
  const lines = [
    "## Task Analysis Record",
    "",
    `**Complexity Score:** ${record.complexity.total}/10`,
    `**Breakdown:** steps=${record.complexity.breakdown.steps}, files=${record.complexity.breakdown.files}, dependency=${record.complexity.breakdown.dependency}, determinism=${record.complexity.breakdown.determinism}`,
    `**Operator Level:** ${record.operatorLevel}`,
    `**Decomposition Triggered:** ${record.decompositionTriggered ? "Yes" : "No"}`,
    ""
  ];

  if (record.complexity.keywords.length > 0) {
    lines.push(`**Keywords:** ${record.complexity.keywords.join(", ")}`);
    lines.push("");
  }

  lines.push(`_Analyzed at ${new Date(record.analysisTimestamp).toISOString()}_`);

  return lines.join("\n");
}

function generateSuggestions(
  task: { id?: string; description: string; steps?: number; files?: string[] },
  complexity: ComplexityScore,
  operatorLevel: OperatorLevel,
  decompositionTriggered: boolean
): TaskAnalyzeResult["suggestions"] {
  const suggestions: TaskAnalyzeResult["suggestions"] = [];

  // Decomposition suggestion
  if (complexity.total >= 7 && !decompositionTriggered) {
    suggestions.push({
      type: "decomposition",
      message: `Task has high complexity (${complexity.total}/10) - decomposition recommended`,
      action: "Consider using decompose_task tool with by_domain strategy"
    });
  }

  if (task.steps && task.steps > 10 && !decompositionTriggered) {
    suggestions.push({
      type: "decomposition",
      message: `Task has ${task.steps} steps - consider breaking down into subtasks`,
      action: "Use decompose_task tool with by_step strategy"
    });
  }

  if (task.files && task.files.length > 10 && !decompositionTriggered) {
    suggestions.push({
      type: "decomposition",
      message: `Task involves ${task.files.length} files - file-based decomposition recommended`,
      action: "Use decompose_task tool with by_file strategy"
    });
  }

  // Priority suggestion
  if (complexity.total >= 8) {
    suggestions.push({
      type: "priority",
      message: "High complexity task should be marked as high priority",
      action: "Use update_task to set priority to 'high' or 'urgent'"
    });
  }

  // Delegation suggestion
  if (operatorLevel === "L4" || operatorLevel === "L5") {
    suggestions.push({
      type: "delegation",
      message: `Task requires ${operatorLevel} operator - ensure appropriate agent is assigned`,
      action: "Consider delegating to specialized agent with L4/L5 capability"
    });
  }

  return suggestions;
}

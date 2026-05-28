/**
 * decomposition_requested Hook
 *
 * Handles task decomposition requests from the subagent-coordinator.
 * Provides decomposition strategy and generates subtasks.
 */

import type {
  DecompositionRequestedEvent,
  DecompositionStrategy,
  Subtask,
} from "@subagent-coordinator/types";

export interface DecompositionRequestedResult {
  recorded: boolean;
  strategy?: DecompositionStrategy;
  subtasks?: Subtask[];
  parallelGroups?: string[];
  estimatedDuration?: number;
  error?: string;
}

/**
 * Handle decomposition_requested event
 *
 * @param event - The decomposition request event
 * @returns Decomposition result with subtasks and strategy
 */
export async function handleDecompositionRequested(
  event: DecompositionRequestedEvent
): Promise<DecompositionRequestedResult> {
  const { task, complexity, suggestedStrategy } = event;

  try {
    const strategy = suggestedStrategy || determineStrategy(task, complexity);
    const subtasks = decomposeTaskByStrategy(task, strategy);
    const parallelGroups = identifyParallelGroups(subtasks);
    const estimatedDuration = subtasks.reduce(
      (sum, s) => sum + (s.estimatedDuration || 60000),
      0
    );

    return {
      recorded: true,
      strategy,
      subtasks,
      parallelGroups,
      estimatedDuration,
    };
  } catch (error) {
    return {
      recorded: false,
      error: error instanceof Error ? error.message : String(error),
    };
  }
}

function determineStrategy(
  task: { description: string; files?: string[] },
  complexity: { total: number }
): DecompositionStrategy {
  const desc = task.description.toLowerCase();
  if (task.files && task.files.length > 1) return "by_file";
  if (/analyze|review|architecture|design|research|investigation/i.test(desc)) return "by_domain";
  return "by_step";
}

function decomposeTaskByStrategy(
  task: { id?: string; description: string; files?: string[] },
  strategy: DecompositionStrategy
): Subtask[] {
  switch (strategy) {
    case "by_file": return decomposeByFile(task);
    case "by_domain": return decomposeByDomain(task);
    default: return decomposeByStep(task);
  }
}

function decomposeByFile(task: { description: string; files?: string[] }): Subtask[] {
  if (!task.files || task.files.length === 0) return decomposeByStep(task);
  const maxPerSubtask = 5;
  const subtasks: Subtask[] = [];
  for (let i = 0; i < task.files.length; i += maxPerSubtask) {
    const batch = task.files.slice(i, i + maxPerSubtask);
    const batchNum = Math.floor(i / maxPerSubtask) + 1;
    const totalBatches = Math.ceil(task.files.length / maxPerSubtask);
    subtasks.push({
      id: `subtask_file_${batchNum}_of_${totalBatches}`,
      description: `Process files: ${batch.join(", ")}`,
      estimatedDuration: batch.length * 30000,
      parallelGroup: totalBatches > 1 ? `file_batch_${batchNum}` : undefined,
    });
  }
  for (let i = 1; i < subtasks.length; i++) {
    subtasks[i].dependsOn = [subtasks[i - 1].id];
  }
  return subtasks;
}

function decomposeByDomain(task: { description: string }): Subtask[] {
  const desc = task.description.toLowerCase();
  const domainPatterns = [
    { pattern: /data|database|storage/i, name: "data_layer" },
    { pattern: /api|endpoint|service/i, name: "api_layer" },
    { pattern: /ui|interface|frontend|visual/i, name: "ui_layer" },
    { pattern: /auth|security|permission/i, name: "security" },
    { pattern: /test|testing|qa/i, name: "testing" },
    { pattern: /deploy|devops|infrastructure/i, name: "deployment" },
  ];
  const matchedDomains = domainPatterns
    .filter(d => d.pattern.test(desc))
    .map(d => d.name);
  if (matchedDomains.length === 0) matchedDomains.push("analysis", "implementation", "verification");
  return matchedDomains.map((domain, idx) => ({
    id: `subtask_domain_${domain}`,
    description: `Work on ${domain.replace("_", " ")}: ${task.description}`,
    estimatedDuration: 300000,
    dependsOn: idx > 0 ? [`subtask_domain_${matchedDomains[idx - 1]}`] : undefined,
  }));
}

function decomposeByStep(task: { description: string }): Subtask[] {
  const separators = [/\s+(?:then|and then|next|after that|接着|然后)\s+/i, /[;\n]+/, /[.。]+(?=\s*[A-Z])/];
  let steps: string[] = [];
  for (const sep of separators) {
    if (sep.test(task.description)) {
      steps = task.description.split(sep).map(s => s.trim()).filter(Boolean);
      break;
    }
  }
  if (steps.length <= 1) {
    steps = task.description.split(/(?<=[.。!?])\s+/).map(s => s.trim()).filter(Boolean);
  }
  if (steps.length <= 1) {
    return [{ id: "subtask_1", description: task.description, estimatedDuration: 60000 }];
  }
  return steps.map((step, idx) => ({
    id: `subtask_${idx + 1}`,
    description: step,
    estimatedDuration: 60000 * (idx + 1),
    dependsOn: idx > 0 ? [`subtask_${idx}`] : undefined,
  }));
}

function identifyParallelGroups(subtasks: Subtask[]): string[] {
  const groups: string[] = [];
  const seen = new Set<string>();
  for (const subtask of subtasks) {
    if (subtask.parallelGroup && !seen.has(subtask.parallelGroup)) {
      groups.push(subtask.parallelGroup);
      seen.add(subtask.parallelGroup);
    }
  }
  return groups;
}

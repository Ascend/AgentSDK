/**
 * decompose_task Tool
 *
 * Decomposes a task into subtasks using various strategies.
 * Supports by_file, by_step, and by_domain decomposition.
 */

import type { Task, Subtask, DecompositionStrategy } from "@subagent-coordinator/types";
import type { TaskStoreService } from "../services/task_store";
import type { TaskGraphService } from "../services/task_graph";

export interface DecomposeTaskInput {
  taskId: string;
  strategy: DecompositionStrategy;
  maxSubtasks?: number;
}

export interface DecomposeTaskOutput {
  parentTask: Task;
  subtasks: Subtask[];
  parallelGroups: string[][];
  estimatedDuration: number;
}

function generateSubtaskId(): string {
  return `subtask_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`;
}

export function createDecomposeTaskTool(
  taskStore: TaskStoreService,
  taskGraph: TaskGraphService
) {
  return async (input: DecomposeTaskInput): Promise<DecomposeTaskOutput> => {
    const { taskId, strategy, maxSubtasks = 10 } = input;

    const parentTask = taskStore.get(taskId);
    if (!parentTask) {
      throw new Error(`Task not found: ${taskId}`);
    }

    // Get existing subtasks
    const existingSubtasks = taskStore.getSubtasks(taskId);
    if (existingSubtasks.length > 0) {
      throw new Error(`Task already has ${existingSubtasks.length} subtasks. Complete or delete them first.`);
    }

    let subtasks: Subtask[] = [];
    let parallelGroups: string[][] = [];

    switch (strategy) {
      case "by_file":
        subtasks = decomposeByFile(parentTask, maxSubtasks);
        break;
      case "by_step":
        subtasks = decomposeByStep(parentTask, maxSubtasks);
        break;
      case "by_domain":
        subtasks = decomposeByDomain(parentTask, maxSubtasks);
        break;
      default:
        throw new Error(`Unknown decomposition strategy: ${strategy}`);
    }

    // Calculate parallel groups based on dependencies
    parallelGroups = calculateParallelGroups(subtasks);

    // Calculate total estimated duration
    const estimatedDuration = subtasks.reduce(
      (sum, s) => sum + (s.estimatedDuration || 60000), // Default 1 minute
      0
    );

    // Create subtasks in store
    // Build temp-id → real-id mapping so setDependencies uses correct references
    const tempToRealId = new Map<string, string>();

    for (const subtask of subtasks) {
      const taskData: Task & { parentId?: string; status?: string; createdAt?: number } = {
        description: subtask.description,
        estimatedDuration: subtask.estimatedDuration,
        parentId: taskId,
        status: "open",
        createdAt: Date.now()
      };

      const created = taskStore.create(taskData);

      // Map the temp id (used in dependsOn) to the real store id
      if (subtask.id) {
        tempToRealId.set(subtask.id, created.id);
      }

      // Translate each dependsOn from temp ids → real ids before saving
      if (subtask.dependsOn && subtask.dependsOn.length > 0) {
        const realDeps = subtask.dependsOn
          .map(dep => tempToRealId.get(dep) ?? dep) // fallback to original if not mapped (defensive)
          .filter(Boolean);
        taskStore.setDependencies(created.id, realDeps);
      }

      // Keep subtask.id in sync so the return value shows real store IDs
      subtask.id = created.id;
    }

    // Add decomposition note to parent task
    taskStore.addNote(taskId, {
      content: `Task decomposed into ${subtasks.length} subtasks using ${strategy} strategy`,
      type: "progress",
      author: "taskr-plugin"
    });

    return {
      parentTask,
      subtasks,
      parallelGroups,
      estimatedDuration
    };
  };
}

function decomposeByFile(task: Task, maxSubtasks: number): Subtask[] {
  const files = task.files || [];
  const subtasks: Subtask[] = [];

  if (files.length === 0) {
    // No files - create a single subtask
    return [{
      id: generateSubtaskId(),
      description: task.description,
      estimatedDuration: task.estimatedDuration
    }];
  }

  // Group files for parallel processing
  const batchSize = Math.ceil(files.length / maxSubtasks);

  for (let i = 0; i < Math.min(files.length, maxSubtasks); i++) {
    const start = i * batchSize;
    const end = Math.min(start + batchSize, files.length);
    const batchFiles = files.slice(start, end);

    subtasks.push({
      id: generateSubtaskId(),
      description: `Process "file" + (batchFiles.length > 1 ? " " + batchFiles.length + " files" : "file"): ${batchFiles.join(", ")}`,
      estimatedDuration: batchFiles.length * 30000, // 30 sec per file
      parallelGroup: `file_batch_${i}`
    });
  }

  // Set sequential dependencies between batches
  for (let i = 1; i < subtasks.length; i++) {
    subtasks[i].dependsOn = [subtasks[i - 1].id];
  }

  return subtasks;
}

function decomposeByStep(task: Task, maxSubtasks: number): Subtask[] {
  const totalSteps = task.steps || 5;
  const subtasks: Subtask[] = [];

  // Determine granularity
  let stepsPerSubtask = Math.ceil(totalSteps / maxSubtasks);
  stepsPerSubtask = Math.max(1, stepsPerSubtask);

  for (let i = 0; i < maxSubtasks && i * stepsPerSubtask < totalSteps; i++) {
    const startStep = i * stepsPerSubtask + 1;
    const endStep = Math.min((i + 1) * stepsPerSubtask, totalSteps);

    subtasks.push({
      id: generateSubtaskId(),
      description: `Execute ${endStep === startStep ? `step ${startStep}` : `steps ${startStep}-${endStep}`} of ${totalSteps}`,
      estimatedDuration: (endStep - startStep + 1) * 60000, // 1 min per step
      dependsOn: i > 0 ? [subtasks[i - 1].id] : undefined
    });
  }

  return subtasks;
}

function decomposeByDomain(task: Task, maxSubtasks: number): Subtask[] {
  // Domain phases split into two parallel tracks:
  //   Track A (sequential): Research → Analysis
  //   Track B (parallel with A): Implementation → Testing → Documentation
  const phases = [
    { name: "Research & Discovery", duration: 120000, track: "A" },
    { name: "Analysis & Design", duration: 180000, track: "A", dependsOnPrev: true },
    { name: "Implementation", duration: 300000, track: "B", parallelGroup: "track-b" },
    { name: "Testing & Validation", duration: 120000, track: "B", dependsOnPrev: true },
    { name: "Documentation & Delivery", duration: 60000, track: "B", dependsOnPrev: true }
  ];

  const subtasks: Subtask[] = [];

  for (let i = 0; i < Math.min(phases.length, maxSubtasks); i++) {
    const phase = phases[i];

    subtasks.push({
      id: generateSubtaskId(),
      description: `${phase.name}: ${task.description}`,
      estimatedDuration: phase.duration,
      parallelGroup: phase.parallelGroup,
      // Only depend on previous phase within the same track
      dependsOn: (phase as any).dependsOnPrev && i > 0 ? [subtasks[i - 1].id] : undefined
    });
  }

  return subtasks;
}

function calculateParallelGroups(subtasks: Subtask[]): string[][] {
  const groups: Map<string, string[]> = new Map();

  for (const subtask of subtasks) {
    if (subtask.parallelGroup) {
      if (!groups.has(subtask.parallelGroup)) {
        groups.set(subtask.parallelGroup, []);
      }
      groups.get(subtask.parallelGroup)!.push(subtask.id);
    }
  }

  return Array.from(groups.values());
}

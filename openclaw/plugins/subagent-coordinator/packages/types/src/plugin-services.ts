import type { Task, ComplexityScore, OperatorLevel, Subtask, QualityGateResult, CheckpointData, DecompositionStrategy } from "./types";

export interface PluginServices {
  taskComplexityScorer?: {
    score(task: Task): Promise<ComplexityScore>;
  };

  operatorClassifier?: {
    classify(complexity: ComplexityScore): Promise<OperatorLevel>;
  };

  taskDecomposer?: {
    decompose(task: Task, strategy: DecompositionStrategy): Promise<Subtask[]>;
    persist(task: Task): Promise<string>;
  };

  checkpointManager?: {
    save(checkpoint: CheckpointData): Promise<string>;
    restore(checkpointId: string): Promise<CheckpointData | null>;
    list(taskId: string): Promise<string[]>;
  };

  qualityGate?: {
    check(task: Task, preExecution: boolean): Promise<QualityGateResult>;
  };
}

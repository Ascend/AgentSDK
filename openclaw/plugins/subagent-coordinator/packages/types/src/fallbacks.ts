import type { Task, ComplexityScore, OperatorLevel, RoutingDecision, Subtask, QualityGateResult, CheckpointData, DecompositionStrategy } from "./types";

export interface FallbackImplementations {
  calculateComplexity(task: Task): ComplexityScore;
  classifyOperatorLevel(complexity: ComplexityScore): OperatorLevel;
  makeRoutingDecision(complexity: ComplexityScore, operatorLevel: OperatorLevel): RoutingDecision;
  decomposeTask(task: Task, strategy: DecompositionStrategy): Subtask[];
  runQualityGate(task: Task, preExecution: boolean): QualityGateResult;
  saveCheckpoint(checkpoint: CheckpointData): string;
  restoreCheckpoint(checkpointId: string): CheckpointData | null;
}

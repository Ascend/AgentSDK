export interface Task {
  id?: string;
  description: string;
  steps?: number;
  files?: string[];
  estimatedDuration?: number;
  priority?: "low" | "normal" | "high" | "urgent";
  metadata?: Record<string, unknown>;
}

export type OperatorLevel = "L1" | "L2" | "L3" | "L4" | "L5";

export interface ComplexityScore {
  total: number;
  breakdown: {
    steps: number;
    files: number;
    dependency: number;
    determinism: number;
  };
  keywords: string[];
}

export type RuntimeType = "subagent" | "acp";

export interface RoutingDecision {
  runtime: RuntimeType;
  agentId: string;
  reason: string;
  complexity: ComplexityScore;
  operatorLevel: OperatorLevel;
}

export interface Subtask {
  id: string;
  description: string;
  dependsOn?: string[];
  estimatedDuration?: number;
  parallelGroup?: string;
}

export type DecompositionStrategy = "by_file" | "by_step" | "by_domain";

export interface ExecutionResult {
  taskId: string;
  success: boolean;
  output?: unknown;
  error?: string;
  duration: number;
  tokensUsed?: number;
}

export interface QualityGateResult {
  pass: boolean;
  checks: {
    name: string;
    pass: boolean;
    message?: string;
  }[];
}

export interface CheckpointData {
  taskId: string;
  subtasks: Subtask[];
  completedSubtasks: string[];
  results: Map<string, ExecutionResult>;
  timestamp: number;
}

export interface RoutingSuggestion {
  runtime: RuntimeType;
  agentId: string;
  reason: string;
  priority?: "low" | "normal" | "high";
}

export interface HookResult {
  recorded?: boolean;
  block?: boolean;
  checks?: { name: string; pass: boolean; message?: string }[];
  enhancedRoutingSuggestion?: RoutingSuggestion;
  metrics?: Record<string, unknown>;
  runtime?: RuntimeType;
  agentId?: string;
  reason?: string;
  enhanced?: boolean;
  originalScore?: ComplexityScore;
  enhancedScore?: ComplexityScore;
}

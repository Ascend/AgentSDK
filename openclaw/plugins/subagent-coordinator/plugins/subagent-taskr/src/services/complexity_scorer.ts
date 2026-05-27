import type { Task, ComplexityScore, OperatorLevel } from "@subagent-coordinator/types";

export interface TaskComplexityScorerConfig {
  weights: {
    steps: number;
    files: number;
    dependency: number;
    determinism: number;
    keywords: number;
  };
  thresholds: {
    stepCountHigh: number;
    fileCountHigh: number;
    contextDependencyHigh: number;
  };
}

const DEFAULT_CONFIG: TaskComplexityScorerConfig = {
  weights: {
    steps: 3,
    files: 3,
    dependency: 2,
    determinism: 1,
    keywords: 1
  },
  thresholds: {
    stepCountHigh: 10,
    fileCountHigh: 10,
    contextDependencyHigh: 5
  }
};

const HIGH_COMPLEXITY_KEYWORDS = [
  "analysis", "analyze", "research", "review", "evaluate", "assess",
  "investigate", "examine", "study", "survey",
  "design", "architecture", "architect", "plan", "architectural",
  "blueprint", "framework", "system design",
  "algorithm", "optimize", "optimize", "refactor", "restructure",
  "migration", "transform", "convert",
  "reasoning", "inference", "hypothesis", "conclusion", "synthesis"
];

const LOW_COMPLEXITY_KEYWORDS = [
  "copy", "move", "rename", "delete", "create", "list",
  "batch", "bulk", "multiple", "simple",
  "file", "directory", "folder", "path",
  "format", "validate", "check", "verify"
];

const CREATIVE_KEYWORDS = [
  "generate", "create", "write", "draw", "invent", "compose",
  "creative", "novel", "new", "original", "design"
];

const DEBUG_KEYWORDS = [
  "debug", "fix", "error", "bug", "issue", "problem",
  "crash", "exception", "fault", "defect"
];

const SECURITY_KEYWORDS = [
  "auth", "authenticate", "authorization", "permission", "credential",
  "secret", "password", "token", "security", "encrypt", "decrypt"
];

export interface ClassifierThresholds {
  l1Max: number;
  l2Max: number;
  l3Max: number;
  l4Max: number;
}

const DEFAULT_THRESHOLDS: ClassifierThresholds = {
  l1Max: 2,
  l2Max: 4,
  l3Max: 6,
  l4Max: 8
};

export interface ClassificationResult {
  level: OperatorLevel;
  name: string;
  description: string;
  delegationRule: string;
  recommendedRuntime: "subagent" | "acp";
  confidence: "high" | "medium" | "low";
}

export const OPERATOR_LEVELS: Record<OperatorLevel, {
  name: string;
  description: string;
  delegationRule: string;
  recommendedRuntime: "subagent" | "acp";
}> = {
  L1: {
    name: "Simple",
    description: "Simple, single-step operations with high determinism",
    delegationRule: "ALWAYS_DELEGATE",
    recommendedRuntime: "subagent"
  },
  L2: {
    name: "Batch",
    description: "Batch operations on multiple files/entities",
    delegationRule: "DELEGATE_WITH_SPLIT",
    recommendedRuntime: "subagent"
  },
  L3: {
    name: "Processing",
    description: "Data processing, analysis, or transformation tasks",
    delegationRule: "DELEGATE_WITH_CHECKPOINT",
    recommendedRuntime: "subagent"
  },
  L4: {
    name: "Analysis",
    description: "Complex analysis, design, or multi-step reasoning",
    delegationRule: "DELEGATE_WITH_SUPERVISION",
    recommendedRuntime: "acp"
  },
  L5: {
    name: "Complex",
    description: "Highly complex tasks requiring full context and reasoning",
    delegationRule: "MAIN_AGENT_ONLY",
    recommendedRuntime: "acp"
  }
};

export interface ComplexityScorer {
  score(task: Task): ComplexityScore;
  classify(complexity: ComplexityScore): ClassificationResult;
}

export class HeuristicComplexityScorer implements ComplexityScorer {
  constructor(
    private readonly config: TaskComplexityScorerConfig = DEFAULT_CONFIG,
    private readonly thresholds: ClassifierThresholds = DEFAULT_THRESHOLDS
  ) {}

  score(task: Task): ComplexityScore {
    const desc = task.description.toLowerCase();

    const stepsScore = calculateStepScore(task.steps, this.config.thresholds.stepCountHigh);
    const filesScore = calculateFileScore(task.files?.length || 0, this.config.thresholds.fileCountHigh);
    const dependencyScore = calculateDependencyScore(desc);
    const determinismScore = calculateDeterminismScore(desc);
    const keywordResults = calculateKeywordScore(desc);

    const rawTotal =
      stepsScore * this.config.weights.steps +
      filesScore * this.config.weights.files +
      dependencyScore * this.config.weights.dependency +
      determinismScore * this.config.weights.determinism +
      keywordResults.score * this.config.weights.keywords;

    const maxPossible =
      3 * this.config.weights.steps +
      3 * this.config.weights.files +
      2 * this.config.weights.dependency +
      2 * this.config.weights.determinism +
      2 * this.config.weights.keywords;

    const normalizedTotal = Math.round((rawTotal / maxPossible) * 10);

    return {
      total: Math.max(1, Math.min(10, normalizedTotal)),
      breakdown: {
        steps: stepsScore,
        files: filesScore,
        dependency: dependencyScore,
        determinism: determinismScore
      },
      keywords: keywordResults.matchedKeywords
    };
  }

  classify(complexity: ComplexityScore): ClassificationResult {
    const score = complexity.total;

    let level: OperatorLevel;
    let confidence: "high" | "medium" | "low" = "high";

    if (score <= this.thresholds.l1Max) {
      level = "L1";
    } else if (score <= this.thresholds.l2Max) {
      level = "L2";
    } else if (score <= this.thresholds.l3Max) {
      level = "L3";
    } else if (score <= this.thresholds.l4Max) {
      level = "L4";
    } else {
      level = "L5";
    }

    const boundaries = [this.thresholds.l1Max, this.thresholds.l2Max, this.thresholds.l3Max, this.thresholds.l4Max];
    const distanceFromBoundary = boundaries.reduce((min, boundary) => {
      return Math.min(min, Math.abs(score - boundary));
    }, Infinity);

    if (distanceFromBoundary <= 1) {
      confidence = "medium";
    }

    if (confidence === "medium") {
      const breakdownValues = Object.values(complexity.breakdown);
      const maxBreakdown = Math.max(...breakdownValues);
      const hasUnbalancedBreakdown = maxBreakdown >= 3 && breakdownValues.filter(v => v >= 2).length >= 2;

      if (hasUnbalancedBreakdown) {
        confidence = "low";
      }
    }

    const levelInfo = OPERATOR_LEVELS[level];

    return {
      level,
      name: levelInfo.name,
      description: levelInfo.description,
      delegationRule: levelInfo.delegationRule,
      recommendedRuntime: levelInfo.recommendedRuntime,
      confidence
    };
  }
}

export function calculateComplexity(
  task: Task,
  config: TaskComplexityScorerConfig = DEFAULT_CONFIG
): ComplexityScore {
  return new HeuristicComplexityScorer(config).score(task);
}

export function classifyOperatorLevel(
  complexity: ComplexityScore,
  thresholds: ClassifierThresholds = DEFAULT_THRESHOLDS
): ClassificationResult {
  return new HeuristicComplexityScorer(DEFAULT_CONFIG, thresholds).classify(complexity);
}

export function enhanceComplexityScore(
  baseScore: ComplexityScore,
  additionalContext: {
    hasCrossSystemOperations?: boolean;
    hasSecuritySensitivity?: boolean;
    hasDataTransformation?: boolean;
  }
): ComplexityScore {
  let bonus = 0;

  if (additionalContext.hasCrossSystemOperations) {
    bonus += 1;
  }
  if (additionalContext.hasSecuritySensitivity) {
    bonus += 1;
  }
  if (additionalContext.hasDataTransformation) {
    bonus += 1;
  }

  return {
    ...baseScore,
    total: Math.max(1, Math.min(10, baseScore.total + bonus))
  };
}

function calculateStepScore(steps: number | undefined, highThreshold: number): number {
  if (!steps) return 0;
  if (steps <= 2) return 1;
  if (steps <= 5) return 2;
  if (steps <= highThreshold) return 3;
  return 3;
}

function calculateFileScore(fileCount: number, highThreshold: number): number {
  if (fileCount <= 1) return 0;
  if (fileCount <= 3) return 1;
  if (fileCount <= highThreshold) return 2;
  return 3;
}

function calculateDependencyScore(description: string): number {
  const dependencyIndicators = [
    /previous|before|after|following/i,
    /based on|depending on|contingent/i,
    /requires|need to know|must have/i,
    /context|background|history/i
  ];

  let score = 0;
  for (const indicator of dependencyIndicators) {
    if (indicator.test(description)) {
      score++;
    }
  }

  return Math.min(2, score);
}

function calculateDeterminismScore(description: string): number {
  for (const keyword of CREATIVE_KEYWORDS) {
    if (description.includes(keyword)) {
      return 2;
    }
  }

  for (const keyword of LOW_COMPLEXITY_KEYWORDS) {
    if (description.includes(keyword)) {
      return 0;
    }
  }

  return 1;
}

function calculateKeywordScore(description: string): { score: number; matchedKeywords: string[] } {
  const matchedKeywords: string[] = [];

  for (const keyword of HIGH_COMPLEXITY_KEYWORDS) {
    if (description.includes(keyword)) {
      matchedKeywords.push(keyword);
    }
  }

  let score = 0;
  if (matchedKeywords.length >= 3) {
    score = 2;
  } else if (matchedKeywords.length >= 1) {
    score = 1;
  }

  return { score, matchedKeywords };
}

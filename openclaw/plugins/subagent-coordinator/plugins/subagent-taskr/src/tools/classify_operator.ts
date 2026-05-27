import type { ComplexityScore } from "@subagent-coordinator/types";
import type { ComplexityScorer, ClassificationResult } from "../services/complexity_scorer";

export function createClassifyOperatorTool(scorer: ComplexityScorer) {
  return async (params: { complexity: ComplexityScore }): Promise<ClassificationResult> => {
    const { complexity } = params;
    return scorer.classify(complexity);
  };
}

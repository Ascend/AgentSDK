import type { Task, ComplexityScore } from "@subagent-coordinator/types";
import type { ComplexityScorer } from "../services/complexity_scorer";

export function createScoreComplexityTool(scorer: ComplexityScorer) {
  return async (params: { task: Task }) => {
    const { task } = params;
    const result = scorer.score(task);
    return result;
  };
}

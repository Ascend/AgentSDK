/**
 * before_delegation Hook
 *
 * Pre-delegation quality check for subagent-coordinator.
 * Validates task parameters and provides routing suggestions.
 */

import type { BeforeDelegationEvent } from "@subagent-coordinator/types";

export async function handleBeforeDelegation(
  event: BeforeDelegationEvent
): Promise<{
  block: boolean;
  checks: { name: string; pass: boolean; message?: string }[];
  enhancedRoutingSuggestion?: {
    runtime: "subagent" | "acp";
    agentId: string;
    reason: string;
  };
}> {
  const checks: { name: string; pass: boolean; message?: string }[] = [];

  // Check 1: Task has description
  if (!event.task.description || event.task.description.trim().length === 0) {
    checks.push({
      name: "has_description",
      pass: false,
      message: "Task description is empty"
    });
  } else {
    checks.push({ name: "has_description", pass: true });
  }

  // Check 2: Task description is not too long
  if (event.task.description && event.task.description.length > 5000) {
    checks.push({
      name: "description_length_reasonable",
      pass: false,
      message: "Task description is unusually long"
    });
  } else {
    checks.push({ name: "description_length_reasonable", pass: true });
  }

  // Check 3: Step count is reasonable
  if (event.task.steps && event.task.steps > 50) {
    checks.push({
      name: "step_count_reasonable",
      pass: false,
      message: "Step count exceeds 50 - consider decomposing"
    });
  } else {
    checks.push({ name: "step_count_reasonable", pass: true });
  }

  // Check 4: Complexity-score alignment check
  const complexityScore = event.complexity.total;
  const operatorLevel = event.operatorLevel;

  // Warn if there's a mismatch between complexity score and operator level
  if (complexityScore >= 7 && operatorLevel !== "L4" && operatorLevel !== "L5") {
    checks.push({
      name: "complexity_operator_alignment",
      pass: false,
      message: `Complexity score ${complexityScore} suggests L4/L5 but got ${operatorLevel}`
    });
  } else if (complexityScore <= 3 && (operatorLevel === "L4" || operatorLevel === "L5")) {
    checks.push({
      name: "complexity_operator_alignment",
      pass: false,
      message: `Complexity score ${complexityScore} suggests L1-L3 but got ${operatorLevel}`
    });
  } else {
    checks.push({ name: "complexity_operator_alignment", pass: true });
  }

  // Check 5: Files exist (if file operations)
  if (event.task.files && event.task.files.length > 0) {
    // Basic check - actual file validation done by subagent
    checks.push({ name: "has_files", pass: true });

    if (event.task.files.length > 100) {
      checks.push({
        name: "file_count_reasonable",
        pass: false,
        message: "File count exceeds 100 - consider batch processing"
      });
    } else {
      checks.push({ name: "file_count_reasonable", pass: true });
    }
  }

  // Determine if we should block delegation
  const hasBlockingFailure = checks.some(c => !c.pass &&
    ["has_description", "description_length_reasonable", "step_count_reasonable"].includes(c.name));

  // Provide enhanced routing suggestion if complexity suggests different runtime
  let enhancedRoutingSuggestion;
  if (complexityScore >= 7 && event.routingDecision.runtime === "subagent") {
    enhancedRoutingSuggestion = {
      runtime: "acp",
      agentId: "researcher",
      reason: `Complexity score ${complexityScore} suggests ACP runtime for better handling`
    };
  }

  return {
    block: hasBlockingFailure,
    checks,
    enhancedRoutingSuggestion
  };
}

/**
 * quality_gate Hook
 *
 * Quality gate validation for pre/post task execution.
 * Ensures tasks meet quality standards before execution
 * and validates results after execution.
 */

import type { Task, QualityGateResult } from "@subagent-coordinator/types";

export interface QualityGateInput {
  task: Task;
  preExecution: boolean;
  executionResult?: {
    success: boolean;
    output?: unknown;
    error?: string;
    duration: number;
  };
}

export async function runQualityGate(input: QualityGateInput): Promise<QualityGateResult> {
  const checks: QualityGateResult["checks"] = [];

  if (input.preExecution) {
    return runPreExecutionChecks(input.task, checks);
  } else {
    return runPostExecutionChecks(input.task, input.executionResult!,checks);
  }
}

function runPreExecutionChecks(
  task: Task,
  checks: QualityGateResult["checks"]
): QualityGateResult {
  // Check 1: Task has non-empty description
  if (!task.description || task.description.trim().length === 0) {
    checks.push({
      name: "has_description",
      pass: false,
      message: "Task description is empty"
    });
  } else {
    checks.push({ name: "has_description", pass: true });
  }

  // Check 2: Description is not excessively long
  if (task.description && task.description.length > 10000) {
    checks.push({
      name: "description_length",
      pass: false,
      message: "Task description exceeds 10000 characters"
    });
  } else {
    checks.push({ name: "description_length", pass: true });
  }

  // Check 3: Step count is within reasonable bounds
  if (task.steps !== undefined) {
    if (task.steps < 1) {
      checks.push({
        name: "step_count_minimum",
        pass: false,
        message: "Step count must be at least 1"
      });
    } else if (task.steps > 100) {
      checks.push({
        name: "step_count_maximum",
        pass: false,
        message: "Step count exceeds 100 - task should be decomposed"
      });
    } else {
      checks.push({ name: "step_count_range", pass: true });
    }
  } else {
    checks.push({ name: "step_count_defined", pass: true }); // steps is optional
  }

  // Check 4: File list is valid if provided
  if (task.files !== undefined) {
    if (!Array.isArray(task.files)) {
      checks.push({
        name: "files_type",
        pass: false,
        message: "Files must be an array"
      });
    } else if (task.files.length === 0) {
      checks.push({
        name: "files_not_empty",
        pass: false,
        message: "Files array is empty"
      });
    } else if (task.files.length > 500) {
      checks.push({
        name: "files_count",
        pass: false,
        message: "File count exceeds 500 - consider batch processing"
      });
    } else {
      checks.push({ name: "files_valid", pass: true });
    }
  } else {
    checks.push({ name: "files_defined", pass: true }); // files is optional
  }

  // Check 5: Estimated duration is reasonable if provided
  if (task.estimatedDuration !== undefined) {
    if (task.estimatedDuration < 0) {
      checks.push({
        name: "estimated_duration_positive",
        pass: false,
        message: "Estimated duration cannot be negative"
      });
    } else if (task.estimatedDuration > 7200000) { // > 2 hours
      checks.push({
        name: "estimated_duration_reasonable",
        pass: false,
        message: "Estimated duration exceeds 2 hours - consider decomposition"
      });
    } else {
      checks.push({ name: "estimated_duration_reasonable", pass: true });
    }
  }

  // Check 6: Priority is valid if provided
  if (task.priority !== undefined) {
    const validPriorities = ["low", "normal", "high", "urgent"];
    if (!validPriorities.includes(task.priority)) {
      checks.push({
        name: "priority_valid",
        pass: false,
        message: `Priority must be one of: ${validPriorities.join(", ")}`
      });
    } else {
      checks.push({ name: "priority_valid", pass: true });
    }
  }

  return {
    pass: checks.every(c => c.pass),
    checks
  };
}

function runPostExecutionChecks(
  task: Task,
  result: NonNullable<QualityGateInput["executionResult"]>,
  checks: QualityGateResult["checks"]
): QualityGateResult {
  // Check 1: Execution completed
  checks.push({
    name: "execution_completed",
    pass: true // If we have result, execution completed
  });

  // Check 2: Success status
  if (result.success) {
    checks.push({
      name: "execution_success",
      pass: true
    });
  } else {
    checks.push({
      name: "execution_success",
      pass: false,
      message: result.error || "Task execution failed"
    });
  }

  // Check 3: Duration is within estimated bounds (if available)
  if (task.estimatedDuration && result.duration) {
    if (result.duration > task.estimatedDuration * 3) {
      checks.push({
        name: "duration_within_estimates",
        pass: false,
        message: `Execution took ${result.duration}ms, significantly exceeding estimate of ${task.estimatedDuration}ms`
      });
    } else {
      checks.push({ name: "duration_within_estimates", pass: true });
    }
  }

  // Check 4: Output validation for generate/create tasks
  const desc = task.description.toLowerCase();
  if (/generate|create|write|output/.test(desc) && result.success) {
    if (!result.output) {
      checks.push({
        name: "has_output",
        pass: false,
        message: "Task expected output but none was produced"
      });
    } else {
      checks.push({ name: "has_output", pass: true });
    }
  }

  // Check 5: Duration is not suspiciously short (for non-trivial tasks)
  if (task.steps && task.steps > 5 && result.duration < 1000) {
    checks.push({
      name: "duration_reasonable",
      pass: false,
      message: "Execution completed suspiciously fast for a multi-step task"
    });
  } else {
    checks.push({ name: "duration_reasonable", pass: true });
  }

  return {
    pass: checks.every(c => c.pass),
    checks
  };
}

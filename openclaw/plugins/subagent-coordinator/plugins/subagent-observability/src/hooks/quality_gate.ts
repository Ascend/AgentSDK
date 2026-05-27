import type {
  QualityGateEvent,
  QualityGateResult,
  HookResult,
} from "@subagent-coordinator/types";

import type { TraceRecorderService } from "../services/trace_recorder";

export function handleQualityGate(
  event: QualityGateEvent,
  traceRecorder: TraceRecorderService
): HookResult {
  const { task, result } = event;

  const currentTrace = traceRecorder.getCurrentTrace();
  if (currentTrace) {
    traceRecorder.recordStep(currentTrace.traceId, {
      type: "quality_check",
      durationMs: 0,
      description: `Quality gate ${event.preExecution ? "pre" : "post"}-execution check: ${result.pass ? "PASS" : "FAIL"}`,
      result: result.checks.map(c => ({
        name: c.name,
        pass: c.pass,
        message: c.message,
      })),
    });
  }

  return {
    recorded: currentTrace !== null,
    checks: result.checks.map(c => ({
      name: c.name,
      pass: c.pass,
      message: c.message,
    })),
  };
}

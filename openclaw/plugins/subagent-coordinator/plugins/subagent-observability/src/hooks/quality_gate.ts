/*!
 * This file is part of the AgentSDK project.
 * Copyright (c) 2026 Huawei Technologies Co.,Ltd.
 *
 * AgentSDK is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *          http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

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

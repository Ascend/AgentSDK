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

export const SUBAGENT_COORDINATOR_EVENTS = {
  BEFORE_DELEGATION: "subagent-coordinator:before_delegation",
  AFTER_EXECUTION: "subagent-coordinator:after_execution",
  TASK_ANALYZED: "subagent-coordinator:task_analyzed",
  DECOMPOSITION_REQUESTED: "subagent-coordinator:decomposition_requested",
  ROUTE_DECISION: "subagent-coordinator:route_decision",
  QUALITY_GATE: "subagent-coordinator:quality_gate",
  CHECKPOINT_SAVE: "subagent-coordinator:checkpoint_save",
  CHECKPOINT_RESTORE: "subagent-coordinator:checkpoint_restore",
} as const;

export type SubagentCoordinatorEventName =
  typeof SUBAGENT_COORDINATOR_EVENTS[keyof typeof SUBAGENT_COORDINATOR_EVENTS];

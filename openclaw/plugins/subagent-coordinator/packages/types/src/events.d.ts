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
export declare const SUBAGENT_COORDINATOR_EVENTS: {
    readonly BEFORE_DELEGATION: "subagent-coordinator:before_delegation";
    readonly AFTER_EXECUTION: "subagent-coordinator:after_execution";
    readonly TASK_ANALYZED: "subagent-coordinator:task_analyzed";
    readonly DECOMPOSITION_REQUESTED: "subagent-coordinator:decomposition_requested";
    readonly ROUTE_DECISION: "subagent-coordinator:route_decision";
    readonly QUALITY_GATE: "subagent-coordinator:quality_gate";
    readonly CHECKPOINT_SAVE: "subagent-coordinator:checkpoint_save";
    readonly CHECKPOINT_RESTORE: "subagent-coordinator:checkpoint_restore";
};
export type SubagentCoordinatorEventName = typeof SUBAGENT_COORDINATOR_EVENTS[keyof typeof SUBAGENT_COORDINATOR_EVENTS];
//# sourceMappingURL=events.d.ts.map

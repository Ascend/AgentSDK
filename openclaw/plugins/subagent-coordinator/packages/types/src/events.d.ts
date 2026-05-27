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

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

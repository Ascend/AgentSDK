import type { SubagentCoordinatorEventName } from "./events";
import type { Task, ComplexityScore, OperatorLevel, RoutingDecision, ExecutionResult, QualityGateResult, CheckpointData, DecompositionStrategy, RuntimeType } from "./types";
interface BeforeDelegationEvent {
    task: Task;
    complexity: ComplexityScore;
    operatorLevel: OperatorLevel;
    routingDecision: RoutingDecision;
    timestamp: number;
}
interface AfterExecutionEvent {
    task: Task;
    result: ExecutionResult;
    complexity: ComplexityScore;
    operatorLevel: OperatorLevel;
    timestamp: number;
}
interface TaskAnalyzedEvent {
    task: Task;
    complexity: ComplexityScore;
    operatorLevel: OperatorLevel;
    decompositionTriggered: boolean;
    timestamp: number;
}
interface DecompositionRequestedEvent {
    task: Task;
    complexity: ComplexityScore;
    suggestedStrategy?: DecompositionStrategy;
    timestamp: number;
}
interface RouteDecisionEvent {
    task: Task;
    complexity: ComplexityScore;
    proposedRuntime: RuntimeType;
    proposedAgentId: string;
    timestamp: number;
}
interface QualityGateEvent {
    task: Task;
    preExecution: boolean;
    result: QualityGateResult;
    timestamp: number;
}
interface CheckpointSaveEvent {
    checkpoint: CheckpointData;
    timestamp: number;
}
interface CheckpointRestoreEvent {
    checkpointId: string;
    taskId: string;
    timestamp: number;
}
export type EventPayloadMap = {
    "subagent-coordinator:before_delegation": BeforeDelegationEvent;
    "subagent-coordinator:after_execution": AfterExecutionEvent;
    "subagent-coordinator:task_analyzed": TaskAnalyzedEvent;
    "subagent-coordinator:decomposition_requested": DecompositionRequestedEvent;
    "subagent-coordinator:route_decision": RouteDecisionEvent;
    "subagent-coordinator:quality_gate": QualityGateEvent;
    "subagent-coordinator:checkpoint_save": CheckpointSaveEvent;
    "subagent-coordinator:checkpoint_restore": CheckpointRestoreEvent;
};
export type GetEventPayload<E extends SubagentCoordinatorEventName> = EventPayloadMap[E];
export type { BeforeDelegationEvent, AfterExecutionEvent, TaskAnalyzedEvent, DecompositionRequestedEvent, RouteDecisionEvent, QualityGateEvent, CheckpointSaveEvent, CheckpointRestoreEvent, };
//# sourceMappingURL=payload-map.d.ts.map

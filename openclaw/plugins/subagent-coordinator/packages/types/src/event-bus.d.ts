import type { SubagentCoordinatorEventName } from "./events";
import type { GetEventPayload } from "./payload-map";
export interface EventBus {
    emit<T extends SubagentCoordinatorEventName>(event: T, payload: GetEventPayload<T>): Promise<void>;
    on<T extends SubagentCoordinatorEventName>(event: T, handler: (payload: GetEventPayload<T>) => Promise<void>): void;
    off<T extends SubagentCoordinatorEventName>(event: T, handler: (payload: GetEventPayload<T>) => Promise<void>): void;
}
//# sourceMappingURL=event-bus.d.ts.map

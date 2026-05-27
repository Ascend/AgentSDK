import type { SubagentCoordinatorEventName } from "@subagent-coordinator/types";

export interface RateLimitConfig {
  maxEventsPerSecond: number;
  burstSize: number;
  windowMs: number;
}

export interface RateLimitEntry {
  tokens: number;
  lastRefill: number;
  agentId: string;
}

export interface RateLimitResult {
  allowed: boolean;
  currentRate: number;
  maxRate: number;
  retryAfterMs?: number;
}

export interface RateLimiterState {
  entries: Map<string, RateLimitEntry>;
  config: RateLimitConfig;
}

const DEFAULT_CONFIG: RateLimitConfig = {
  maxEventsPerSecond: 10,
  burstSize: 20,
  windowMs: 1000,
};

export interface RateLimiterService {
  check(agentId: string, eventType?: SubagentCoordinatorEventName): RateLimitResult;
  record(agentId: string, eventType?: SubagentCoordinatorEventName): RateLimitResult;
  getCurrentRate(agentId: string): number;
  reset(agentId: string): void;
  updateConfig(config: Partial<RateLimitConfig>): void;
  getRateLimitedAgents(): string[];
  cleanup(maxAgeMs: number): number;
}

export function createRateLimiter(
  state: RateLimiterState,
  config: Partial<RateLimitConfig> = {}
): RateLimiterService {
  const effectiveConfig: RateLimitConfig = {
    ...DEFAULT_CONFIG,
    ...config,
  };
  state.config = effectiveConfig;

  const refillTokens = (entry: RateLimitEntry): void => {
    const now = Date.now();
    const elapsed = now - entry.lastRefill;
    const tokensToAdd = (elapsed / state.config.windowMs) * state.config.maxEventsPerSecond;

    entry.tokens = Math.min(
      state.config.burstSize,
      entry.tokens + tokensToAdd
    );
    entry.lastRefill = now;
  };

  const getEntry = (agentId: string): RateLimitEntry => {
    if (!state.entries.has(agentId)) {
      state.entries.set(agentId, {
        tokens: state.config.burstSize,
        lastRefill: Date.now(),
        agentId,
      });
    }
    return state.entries.get(agentId)!;
  };

  return {
    check(agentId) {
      const entry = getEntry(agentId);
      refillTokens(entry);

      const allowed = entry.tokens >= 1;
      const currentRate = state.config.maxEventsPerSecond - entry.tokens;

      return {
        allowed,
        currentRate,
        maxRate: state.config.maxEventsPerSecond,
        retryAfterMs: allowed
          ? undefined
          : Math.ceil((1 - entry.tokens) / state.config.maxEventsPerSecond * state.config.windowMs),
      };
    },

    record(agentId) {
      const entry = getEntry(agentId);
      refillTokens(entry);

      const allowed = entry.tokens >= 1;
      const currentRate = state.config.maxEventsPerSecond - entry.tokens;

      if (allowed) {
        entry.tokens -= 1;
      }

      return {
        allowed,
        currentRate,
        maxRate: state.config.maxEventsPerSecond,
        retryAfterMs: allowed
          ? undefined
          : Math.ceil((1 - entry.tokens) / state.config.maxEventsPerSecond * state.config.windowMs),
      };
    },

    getCurrentRate(agentId) {
      const entry = getEntry(agentId);
      refillTokens(entry);
      return state.config.maxEventsPerSecond - entry.tokens;
    },

    reset(agentId) {
      state.entries.delete(agentId);
    },

    updateConfig(config) {
      state.config = {
        ...state.config,
        ...config,
      };
    },

    getRateLimitedAgents() {
      const limited: string[] = [];
      for (const [agentId, entry] of state.entries.entries()) {
        refillTokens(entry);
        if (entry.tokens < 1) {
          limited.push(agentId);
        }
      }
      return limited;
    },

    cleanup(maxAgeMs) {
      const now = Date.now();
      let cleaned = 0;

      for (const [agentId, entry] of state.entries.entries()) {
        if (now - entry.lastRefill > maxAgeMs) {
          state.entries.delete(agentId);
          cleaned++;
        }
      }

      return cleaned;
    },
  };
}

export function createRateLimiterState(): RateLimiterState {
  return {
    entries: new Map(),
    config: DEFAULT_CONFIG,
  };
}

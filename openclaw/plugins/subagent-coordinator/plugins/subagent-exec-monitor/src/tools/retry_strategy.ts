/**
 * Retry Strategy Tool
 *
 * Selects optimal retry strategy based on error type and execution history.
 */

import type { ExecutionResult } from "@subagent-coordinator/types";

export type RetryStrategy =
  | "exponential_backoff"  // Delay doubles each retry: 1s, 2s, 4s, 8s...
  | "linear_backoff"       // Delay increases linearly: 1s, 2s, 3s, 4s...
  | "immediate"           // Retry immediately
  | "no_retry";           // Do not retry

export interface RetryStrategyResult {
  strategy: RetryStrategy;
  maxRetries: number;
  baseDelayMs: number;
  maxDelayMs: number;
  reason: string;
  jitter: boolean;
}

export interface RetryAnalysis {
  errorType: string;
  isRetryable: boolean;
  suggestedStrategy: RetryStrategy;
  confidence: "high" | "medium" | "low";
}

const ERROR_PATTERNS: { pattern: RegExp; type: string; isRetryable: boolean; suggestedStrategy: RetryStrategy }[] = [
  // Network errors - retryable
  { pattern: /timeout/i, type: "timeout", isRetryable: true, suggestedStrategy: "exponential_backoff" },
  { pattern: /connection refused/i, type: "connection_refused", isRetryable: true, suggestedStrategy: "exponential_backoff" },
  { pattern: /network error/i, type: "network", isRetryable: true, suggestedStrategy: "exponential_backoff" },
  { pattern: /econnreset/i, type: "connection_reset", isRetryable: true, suggestedStrategy: "exponential_backoff" },
  { pattern: /enotfound/i, type: "dns_lookup_failed", isRetryable: true, suggestedStrategy: "linear_backoff" },

  // Rate limiting - retryable with longer delays
  { pattern: /rate limit/i, type: "rate_limit", isRetryable: true, suggestedStrategy: "linear_backoff" },
  { pattern: /429/i, type: "rate_limit_429", isRetryable: true, suggestedStrategy: "linear_backoff" },
  { pattern: /429TooManyRequests/i, type: "rate_limit_429", isRetryable: true, suggestedStrategy: "linear_backoff" },
  { pattern: /retry-after/i, type: "rate_limit_retry_after", isRetryable: true, suggestedStrategy: "linear_backoff" },

  // Service errors - potentially retryable
  { pattern: /500/i, type: "server_error_500", isRetryable: true, suggestedStrategy: "exponential_backoff" },
  { pattern: /502/i, type: "server_error_502", isRetryable: true, suggestedStrategy: "exponential_backoff" },
  { pattern: /503/i, type: "server_error_503", isRetryable: true, suggestedStrategy: "exponential_backoff" },
  { pattern: /504/i, type: "server_error_504", isRetryable: true, suggestedStrategy: "exponential_backoff" },

  // Auth/permission errors - not retryable
  { pattern: /401/i, type: "unauthorized", isRetryable: false, suggestedStrategy: "no_retry" },
  { pattern: /403/i, type: "forbidden", isRetryable: false, suggestedStrategy: "no_retry" },
  { pattern: /auth.*fail/i, type: "auth_failure", isRetryable: false, suggestedStrategy: "no_retry" },
  { pattern: /permission denied/i, type: "permission_denied", isRetryable: false, suggestedStrategy: "no_retry" },
  { pattern: /access denied/i, type: "access_denied", isRetryable: false, suggestedStrategy: "no_retry" },

  // Validation errors - not retryable
  { pattern: /400/i, type: "bad_request", isRetryable: false, suggestedStrategy: "no_retry" },
  { pattern: /invalid.*argument/i, type: "invalid_argument", isRetryable: false, suggestedStrategy: "no_retry" },
  { pattern: /validation.*fail/i, type: "validation_failure", isRetryable: false, suggestedStrategy: "no_retry" },

  // Resource errors - may be retryable after delay
  { pattern: /out of memory/i, type: "out_of_memory", isRetryable: true, suggestedStrategy: "linear_backoff" },
  { pattern: /disk full/i, type: "disk_full", isRetryable: true, suggestedStrategy: "no_retry" }, // Won't resolve on its own
  { pattern: /quota exceeded/i, type: "quota_exceeded", isRetryable: true, suggestedStrategy: "linear_backoff" },

  // Generic errors
  { pattern: /internal error/i, type: "internal_error", isRetryable: true, suggestedStrategy: "exponential_backoff" },
  { pattern: /unknown error/i, type: "unknown", isRetryable: true, suggestedStrategy: "immediate" },
];

export function analyzeError(errorMessage: string): RetryAnalysis {
  for (const { pattern, type, isRetryable, suggestedStrategy } of ERROR_PATTERNS) {
    if (pattern.test(errorMessage)) {
      return {
        errorType: type,
        isRetryable,
        suggestedStrategy,
        confidence: "high"
      };
    }
  }

  return {
    errorType: "unknown",
    isRetryable: true,
    suggestedStrategy: "immediate",
    confidence: "low"
  };
}

export function selectRetryStrategy(
  error: string,
  history: ExecutionResult[] = []
): RetryStrategyResult {
  const analysis = analyzeError(error);

  // Check history for patterns
  if (history.length >= 3) {
    const recentResults = history.slice(-3);
    const failures = recentResults.filter(r => !r.success);

    // If multiple consecutive failures, be more conservative
    if (failures.length >= 2) {
      return {
        strategy: "no_retry",
        maxRetries: 0,
        baseDelayMs: 0,
        maxDelayMs: 0,
        reason: `Multiple consecutive failures (${failures.length}/${recentResults.length}) - persistent issue, not retryable`,
        jitter: false
      };
    }
  }

  // If error is not retryable, return no-retry strategy
  if (!analysis.isRetryable) {
    return {
      strategy: "no_retry",
      maxRetries: 0,
      baseDelayMs: 0,
      maxDelayMs: 0,
      reason: `${analysis.errorType} errors are not retryable`,
      jitter: false
    };
  }

  // Select base configuration based on strategy
  const baseConfigs: Record<RetryStrategy, { maxRetries: number; baseDelayMs: number; maxDelayMs: number; jitter: boolean }> = {
    exponential_backoff: { maxRetries: 3, baseDelayMs: 1000, maxDelayMs: 30000, jitter: true },
    linear_backoff: { maxRetries: 5, baseDelayMs: 2000, maxDelayMs: 60000, jitter: false },
    immediate: { maxRetries: 1, baseDelayMs: 0, maxDelayMs: 0, jitter: false },
    no_retry: { maxRetries: 0, baseDelayMs: 0, maxDelayMs: 0, jitter: false }
  };

  const config = baseConfigs[analysis.suggestedStrategy];

  return {
    strategy: analysis.suggestedStrategy,
    maxRetries: config.maxRetries,
    baseDelayMs: config.baseDelayMs,
    maxDelayMs: config.maxDelayMs,
    reason: `${analysis.errorType} errors - using ${analysis.suggestedStrategy}`,
    jitter: config.jitter
  };
}

// Calculate actual delay for a given retry attempt
export function calculateRetryDelay(
  strategy: RetryStrategy,
  attempt: number,
  baseDelayMs: number,
  maxDelayMs: number
): number {
  let delay: number;

  switch (strategy) {
    case "exponential_backoff":
      delay = baseDelayMs * Math.pow(2, attempt);
      break;
    case "linear_backoff":
      delay = baseDelayMs * (attempt + 1);
      break;
    case "immediate":
      delay = 0;
      break;
    case "no_retry":
      delay = 0;
      break;
    default:
      delay = baseDelayMs;
  }

  return Math.min(delay, maxDelayMs);
}

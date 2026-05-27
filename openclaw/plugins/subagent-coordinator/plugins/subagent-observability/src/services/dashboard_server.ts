import type { ExecutionTrace } from "./trace_recorder";
import type { BudgetStatus } from "./cost_tracker";
import type { TrendAnalysis } from "./trend_analyzer";

export interface DashboardConfig {
  enabled: boolean;
  port: number;
  uiPath: string;
}

export interface DashboardState {
  config: DashboardConfig;
  server?: {
    port: number;
    startedAt: number;
  };
  stats: {
    totalTraces: number;
    activeSessions: number;
    totalCost: number;
  };
}

export interface DashboardSnapshot {
  timestamp: number;
  activeTraces: Array<{
    traceId: string;
    sessionId: string;
    taskId: string;
    startTime: number;
    durationMs: number;
    steps: number;
  }>;
  recentTraces: Array<{
    traceId: string;
    sessionId: string;
    success: boolean;
    durationMs: number;
    cost: number;
    startTime: number;
  }>;
  budgetStatus: BudgetStatus | null;
  stats: {
    totalTraces: number;
    activeSessions: number;
    totalCost: number;
  };
}

export function createInitialState(): DashboardState {
  return {
    config: {
      enabled: false,
      port: 3847,
      uiPath: "/dashboard",
    },
    stats: {
      totalTraces: 0,
      activeSessions: 0,
      totalCost: 0,
    },
  };
}

export interface DashboardServerService {
  updateConfig(config: Partial<DashboardConfig>): void;
  getConfig(): DashboardConfig;
  start(): Promise<void>;
  stop(): Promise<void>;
  isRunning(): boolean;
  getSnapshot(): DashboardSnapshot;
  updateStats(stats: Partial<DashboardState["stats"]>): void;
}

export function createDashboardServer(
  state: DashboardState
): DashboardServerService {
  let running = false;
  let httpServer: { close(): void } | null = null;

  let activeTracesProvider: () => ExecutionTrace[] = () => [];
  let recentTracesProvider: () => ExecutionTrace[] = () => [];
  let budgetStatusProvider: () => BudgetStatus | null = () => null;

  const generateDashboardHTML = (snapshot: DashboardSnapshot): string => {
    const { activeTraces, recentTraces, budgetStatus, stats } = snapshot;
    const now = Date.now();

    return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>OpenClaw Observability - Monitoring Dashboard</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f0f1a; color: #fff; padding: 20px; }
    .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 30px; }
    h1 { color: #6366f1; }
    .timestamp { color: #888; font-size: 14px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 20px; margin-bottom: 30px; }
    .card { background: #1a1a2e; border-radius: 12px; padding: 20px; }
    .card h2 { font-size: 14px; color: #888; margin-bottom: 10px; text-transform: uppercase; }
    .card .value { font-size: 32px; font-weight: bold; }
    .card .sub { color: #888; font-size: 12px; margin-top: 5px; }
    .active-traces { background: #1a1a2e; border-radius: 12px; padding: 20px; margin-bottom: 30px; }
    .active-traces h2 { margin-bottom: 15px; color: #f59e0b; }
    .trace-item { padding: 12px; background: #252540; border-radius: 8px; margin-bottom: 8px; }
    .trace-item .id { color: #6366f1; font-family: monospace; }
    .trace-item .meta { color: #888; font-size: 12px; margin-top: 4px; }
    .recent-traces { background: #1a1a2e; border-radius: 12px; padding: 20px; }
    .recent-traces h2 { margin-bottom: 15px; }
    .recent-item { display: flex; justify-content: space-between; padding: 12px; background: #252540; border-radius: 8px; margin-bottom: 8px; }
    .recent-item .success { color: #22c55e; }
    .recent-item .failure { color: #ef4444; }
    .budget-alert { background: #7c2d12; border: 1px solid #f59e0b; border-radius: 8px; padding: 15px; margin-bottom: 20px; }
    .budget-alert h3 { color: #f59e0b; margin-bottom: 5px; }
    table { width: 100%; border-collapse: collapse; }
    th, td { padding: 10px; text-align: left; border-bottom: 1px solid #333; }
    th { color: #888; font-weight: normal; font-size: 12px; text-transform: uppercase; }
    .refresh { position: fixed; bottom: 20px; right: 20px; background: #6366f1; color: white; border: none; padding: 10px 20px; border-radius: 8px; cursor: pointer; }
  </style>
</head>
<body>
  <div class="header">
    <h1>📊 OpenClaw Observability Dashboard</h1>
    <div class="timestamp">Last updated: ${new Date(now).toLocaleTimeString()}</div>
  </div>

  ${budgetStatus && (budgetStatus.daily.remainingPercent < 0.2 || budgetStatus.monthly.remainingPercent < 0.2) ? `
  <div class="budget-alert">
    <h3>⚠️ Budget Alert</h3>
    <p>Daily: $${budgetStatus.daily.used.toFixed(2)} / $${budgetStatus.daily.limit} (${((1 - budgetStatus.daily.remainingPercent) * 100).toFixed(1)}% used)</p>
    <p>Monthly: $${budgetStatus.monthly.used.toFixed(2)} / $${budgetStatus.monthly.limit} (${((1 - budgetStatus.monthly.remainingPercent) * 100).toFixed(1)}% used)</p>
  </div>
  ` : ""}

  <div class="grid">
    <div class="card">
      <h2>Total Traces</h2>
      <div class="value">${stats.totalTraces}</div>
    </div>
    <div class="card">
      <h2>Active Sessions</h2>
      <div class="value">${stats.activeSessions}</div>
    </div>
    <div class="card">
      <h2>Total Cost</h2>
      <div class="value">$${stats.totalCost.toFixed(2)}</div>
    </div>
    <div class="card">
      <h2>Daily Budget</h2>
      <div class="value">${budgetStatus ? `${((1 - budgetStatus.daily.remainingPercent) * 100).toFixed(0)}%` : "N/A"}</div>
      ${budgetStatus ? `<div class="sub">$${budgetStatus.daily.remaining.toFixed(2)} remaining</div>` : ""}
    </div>
  </div>

  ${activeTraces.length > 0 ? `
  <div class="active-traces">
    <h2>🔄 Active Traces (${activeTraces.length})</h2>
    ${activeTraces.map(t => `
    <div class="trace-item">
      <div class="id">${t.traceId}</div>
      <div class="meta">
        Session: ${t.sessionId} | Task: ${t.taskId} |
        Duration: ${((now - t.startTime) / 1000).toFixed(1)}s | Steps: ${t.steps}
      </div>
    </div>
    `).join("")}
  </div>
  ` : ""}

  <div class="recent-traces">
    <h2>Recent Traces</h2>
    <table>
      <thead>
        <tr>
          <th>Trace ID</th>
          <th>Session</th>
          <th>Status</th>
          <th>Duration</th>
          <th>Cost</th>
          <th>Time</th>
        </tr>
      </thead>
      <tbody>
        ${recentTraces.slice(0, 10).map(t => `
        <tr>
          <td><span class="id">${t.traceId.substring(0, 12)}...</span></td>
          <td>${t.sessionId.substring(0, 12)}...</td>
          <td class="${t.success ? 'success' : 'failure'}">${t.success ? "✓" : "✗"}</td>
          <td>${(t.durationMs / 1000).toFixed(1)}s</td>
          <td>$${t.cost.toFixed(4)}</td>
          <td>${new Date(t.startTime).toLocaleTimeString()}</td>
        </tr>
        `).join("")}
      </tbody>
    </table>
  </div>

  <button class="refresh" onclick="location.reload()">🔄 Refresh</button>

  <script>
    setTimeout(() => location.reload(), 10000);
  </script>
</body>
</html>`;
  };

  return {
    updateConfig(config: Partial<DashboardConfig>): void {
      state.config = { ...state.config, ...config };
    },

    getConfig(): DashboardConfig {
      return { ...state.config };
    },

    start: async function(): Promise<void> {
      if (running || !state.config.enabled) {
        return;
      }

      running = true;
      state.server = {
        port: state.config.port,
        startedAt: Date.now(),
      };

      console.log(`[subagent-observability] Dashboard server started on port ${state.config.port}`);
    },

    stop: async function(): Promise<void> {
      if (!running) return;

      if (httpServer) {
        httpServer.close();
        httpServer = null;
      }

      running = false;
      state.server = undefined;
      console.log("[subagent-observability] Dashboard server stopped");
    },

    isRunning(): boolean {
      return running;
    },

    getSnapshot(): DashboardSnapshot {
      const activeTraces = activeTracesProvider();
      const recentTraces = recentTracesProvider();
      const budgetStatus = budgetStatusProvider();

      const uniqueSessions = new Set(activeTraces.map(t => t.sessionId));

      return {
        timestamp: Date.now(),
        activeTraces: activeTraces.map(t => ({
          traceId: t.traceId,
          sessionId: t.sessionId,
          taskId: t.taskId,
          startTime: t.startTime,
          durationMs: Date.now() - t.startTime,
          steps: t.steps.length,
        })),
        recentTraces: recentTraces.map(t => ({
          traceId: t.traceId,
          sessionId: t.sessionId,
          success: t.success,
          durationMs: t.totalDurationMs,
          cost: t.cost,
          startTime: t.startTime,
        })),
        budgetStatus,
        stats: {
          totalTraces: state.stats.totalTraces,
          activeSessions: uniqueSessions.size,
          totalCost: state.stats.totalCost,
        },
      };
    },

    updateStats(stats: Partial<DashboardState["stats"]>): void {
      state.stats = { ...state.stats, ...stats };
    },
  };
}

export function createDashboardDataProviders(
  traceRecorder: { listTraces: (filters?: any) => string[]; getTrace: (id: string) => any | null },
  costTracker: { getTotalCost: (timeRange?: any) => number },
  trendAnalyzer: { get7DayTrend: () => TrendAnalysis }
) {
  return {
    activeTraces: () => {
      const activeIds = traceRecorder.listTraces();
      return activeIds
        .map(id => traceRecorder.getTrace(id))
        .filter((t): t is ExecutionTrace => t !== null && t.endTime === 0);
    },
    recentTraces: () => {
      const cutoff = Date.now() - 24 * 60 * 60 * 1000;
      const allIds = traceRecorder.listTraces();
      return allIds
        .map(id => traceRecorder.getTrace(id))
        .filter((t): t is ExecutionTrace => t !== null && t.startTime >= cutoff)
        .sort((a, b) => b.startTime - a.startTime);
    },
    budgetStatus: () => null,
  };
}

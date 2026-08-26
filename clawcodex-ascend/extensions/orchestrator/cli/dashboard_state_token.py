#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#           http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

from typing import Any


class DashboardStateTokenActivityMixin:
    def _build_token_activity(self, issues: dict[str, Any]) -> dict[str, Any]:
        """Build token_activity from the registry (active session count + turns/tools)."""
        from .dashboard import ACTIVE_STATUSES

        issue_list = issues.get("issues", [])
        all_with_run = [i for i in issue_list if i.get("run_id")]
        active = [i for i in all_with_run if i.get("status") in ACTIVE_STATUSES]
        return {
            "active_sessions": len(active),
            "total_turns": sum(i.get("run_turn_count", 0) for i in all_with_run),
            "total_tools": sum(i.get("run_tool_count", 0) for i in all_with_run),
        }


# JavaScript payload of the dashboard HTML. Kept in a module-level constant so
# Python syntax-checks the surrounding code independently of the JS body.
DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ClawCodex Orchestrator LiveView</title>
  <style>
    :root {
      --bg-0: #0b0f17;
      --bg-1: #11161f;
      --bg-2: #161c26;
      --bg-3: #1d2532;
      --line: #232c3a;
      --line-soft: #1a212c;
      --fg-0: #e6edf3;
      --fg-1: #c9d1d9;
      --fg-2: #8b949e;
      --fg-3: #6e7681;
      --accent: #58a6ff;
      --accent-2: #79c0ff;
      --good: #3fb950;
      --warn: #d29922;
      --bad: #f85149;
      --vermillion: #db6d28;
      --purple: #a371f7;
      --cyan: #79c0ff;
      --gray: #8b949e;
      --shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
    }
    * { box-sizing: border-box; }
    html, body {
      margin: 0; padding: 0;
      background: var(--bg-0);
      color: var(--fg-1);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue",
                   "PingFang SC", "Microsoft YaHei", Arial, sans-serif;
      font-size: 13px;
      line-height: 1.45;
      min-height: 100vh;
    }
    body {
      background:
        radial-gradient(1200px 600px at 80% -10%, rgba(88,166,255,0.06), transparent 70%),
        radial-gradient(1000px 500px at -10% 110%, rgba(163,113,247,0.05), transparent 70%),
        var(--bg-0);
    }
    a { color: var(--accent); text-decoration: none; }
    a:hover { text-decoration: underline; }
    code, .mono { font-family: "JetBrains Mono", "SF Mono", Menlo, Consolas, monospace; }
    button { font-family: inherit; }

    /* Layout */
    .app {
      display: grid;
      grid-template-rows: auto auto 1fr auto;
      grid-template-columns: 1fr;
      min-height: 100vh;
      padding: 18px 22px 12px;
      gap: 14px;
    }

    /* Header */
    .header {
      display: flex; align-items: center; gap: 18px;
      padding: 14px 18px;
      background: linear-gradient(180deg, rgba(255,255,255,0.02), transparent), var(--bg-1);
      border: 1px solid var(--line);
      border-radius: 12px;
      box-shadow: var(--shadow);
    }
    .header .brand {
      display: flex; align-items: center; gap: 10px;
      font-weight: 600; font-size: 15px; color: var(--fg-0);
    }
    .header .brand .logo {
      width: 26px; height: 26px; border-radius: 7px;
      background: conic-gradient(from 200deg, #58a6ff, #a371f7, #3fb950, #58a6ff);
      box-shadow: 0 0 0 1px rgba(255,255,255,0.06) inset;
    }
    .header .meta { display: flex; gap: 18px; flex: 1; flex-wrap: wrap; }
    .header .meta .kv { color: var(--fg-2); font-size: 12px; }
    .header .meta .kv b { color: var(--fg-0); font-weight: 500; margin-left: 6px; }
    .conn {
      display: inline-flex; align-items: center; gap: 8px;
      padding: 6px 10px; border-radius: 999px;
      background: var(--bg-2); border: 1px solid var(--line);
      font-size: 12px;
    }
    .conn .dot {
      width: 8px; height: 8px; border-radius: 50%;
      background: var(--gray);
      box-shadow: 0 0 0 0 rgba(63,185,80,0.5);
      transition: background 0.2s;
    }
    .conn.ok .dot { background: var(--good); animation: pulse 1.6s infinite; }
    .conn.bad .dot { background: var(--bad); }
    @keyframes pulse {
      0%   { box-shadow: 0 0 0 0 rgba(63,185,80,0.5); }
      70%  { box-shadow: 0 0 0 8px rgba(63,185,80,0); }
      100% { box-shadow: 0 0 0 0 rgba(63,185,80,0); }
    }

    /* Status grid */
    .status-grid {
      display: grid;
      grid-template-columns: repeat(8, minmax(0, 1fr));
      gap: 10px;
    }
    @media (max-width: 1280px) { .status-grid { grid-template-columns: repeat(4, 1fr); } }
    @media (max-width: 720px)  { .status-grid { grid-template-columns: repeat(2, 1fr); } }
    .stat {
      position: relative;
      padding: 12px 14px;
      background: var(--bg-1);
      border: 1px solid var(--line);
      border-radius: 10px;
      overflow: hidden;
      transition: transform 0.15s, border-color 0.2s;
    }
    .stat:hover { transform: translateY(-1px); border-color: var(--line); }
    .stat .label {
      display: flex; align-items: center; gap: 6px;
      font-size: 11px; letter-spacing: 0.06em; text-transform: uppercase;
      color: var(--fg-2);
    }
    .stat .label .ico {
      display: inline-flex; align-items: center; justify-content: center;
      width: 16px; height: 16px; border-radius: 4px;
      background: rgba(255,255,255,0.04);
      font-size: 11px;
    }
    .stat .value {
      font-size: 26px; font-weight: 600; color: var(--fg-0);
      margin-top: 6px; font-variant-numeric: tabular-nums;
    }
    .stat .sub { font-size: 11px; color: var(--fg-3); margin-top: 2px; }
    .stat.zero .value { color: var(--fg-3); }
    .stat::before {
      content: ""; position: absolute; left: 0; top: 0; bottom: 0;
      width: 3px; background: var(--c, var(--accent));
    }

    /* Main panels */
    .main {
      display: grid;
      grid-template-columns: minmax(0, 1.4fr) minmax(360px, 1fr);
      gap: 14px;
      min-height: 0;
    }
    @media (max-width: 1100px) { .main { grid-template-columns: 1fr; } }
    .panel {
      background: var(--bg-1);
      border: 1px solid var(--line);
      border-radius: 12px;
      box-shadow: var(--shadow);
      display: flex; flex-direction: column;
      min-height: 0;
    }
    .panel-header {
      display: flex; align-items: center; gap: 10px;
      padding: 10px 14px;
      border-bottom: 1px solid var(--line);
    }
    .panel-header h3 {
      margin: 0; font-size: 13px; font-weight: 600; color: var(--fg-0);
      letter-spacing: 0.02em;
    }
    .panel-header .actions { margin-left: auto; display: flex; gap: 6px; }
    .panel-header input[type="search"], .panel-header select {
      background: var(--bg-2);
      border: 1px solid var(--line);
      color: var(--fg-1);
      border-radius: 6px;
      padding: 5px 8px;
      font-size: 12px;
      outline: none;
    }
    .panel-header input[type="search"] { width: 180px; }
    .panel-header input[type="search"]:focus, .panel-header select:focus { border-color: var(--accent); }
    .panel-body { flex: 1; overflow: auto; min-height: 0; }

    /* Issue table */
    table.issues { width: 100%; border-collapse: collapse; font-size: 12.5px; }
    table.issues th, table.issues td {
      padding: 9px 12px; text-align: left; border-bottom: 1px solid var(--line-soft);
      white-space: nowrap;
    }
    table.issues th {
      position: sticky; top: 0; z-index: 1;
      background: var(--bg-1); color: var(--fg-2); font-weight: 500;
      font-size: 11px; letter-spacing: 0.05em; text-transform: uppercase;
    }
    table.issues tr { cursor: pointer; transition: background 0.1s; }
    table.issues tr:hover { background: rgba(88,166,255,0.05); }
    table.issues tr.selected { background: rgba(88,166,255,0.10); }
    table.issues td.identifier { color: var(--fg-0); font-weight: 500; }
    table.issues td.branch, table.issues td.workspace { color: var(--fg-2); }
    table.issues td.workspace { max-width: 220px; overflow: hidden; text-overflow: ellipsis; }
    table.issues td.num { text-align: right; font-variant-numeric: tabular-nums; }
    .pill {
      display: inline-flex; align-items: center; gap: 6px;
      padding: 2px 8px; border-radius: 999px;
      font-size: 11px; font-weight: 500;
      background: color-mix(in srgb, var(--c) 14%, transparent);
      color: var(--c);
      border: 1px solid color-mix(in srgb, var(--c) 35%, transparent);
    }
    .pill .ico { font-size: 10px; }

    /* Detail panel */
    .detail { padding: 14px 16px; }
    .detail h2 {
      margin: 0 0 4px; font-size: 18px; color: var(--fg-0);
      display: flex; align-items: center; gap: 10px;
    }
    .detail .sub { color: var(--fg-2); font-size: 12px; }
    .detail .grid {
      display: grid; grid-template-columns: 110px 1fr; gap: 6px 14px;
      margin-top: 14px; font-size: 12.5px;
    }
    .detail .grid .k { color: var(--fg-2); }
    .detail .grid .v { color: var(--fg-1); word-break: break-all; }
    .detail .section-title {
      margin: 18px 0 8px; font-size: 11px; text-transform: uppercase;
      letter-spacing: 0.08em; color: var(--fg-3);
    }
    .events-mini {
      max-height: 240px; overflow: auto;
      border: 1px solid var(--line); border-radius: 8px;
      background: var(--bg-0);
    }
    .events-mini .row {
      display: grid; grid-template-columns: 72px 64px 1fr;
      gap: 10px; padding: 6px 10px; font-size: 12px;
      border-bottom: 1px solid var(--line-soft);
    }
    .events-mini .row:last-child { border-bottom: 0; }
    .events-mini .ts { color: var(--fg-3); font-variant-numeric: tabular-nums; }
    .events-mini .type {
      font-size: 10.5px; text-transform: uppercase; letter-spacing: 0.06em;
      color: var(--c, var(--fg-2));
    }
    .events-mini .row > span:last-child {
      white-space: pre-wrap; word-break: break-word;
    }
    .empty {
      padding: 28px 16px; text-align: center; color: var(--fg-3); font-size: 12px;
    }

    /* Live event feed */
    .feed {
      background: var(--bg-1);
      border: 1px solid var(--line);
      border-radius: 12px;
      box-shadow: var(--shadow);
      max-height: 260px;
      display: flex; flex-direction: column;
    }
    .feed .panel-body { font-family: "JetBrains Mono", "SF Mono", Menlo, monospace; font-size: 12px; }
    .feed .row {
      display: grid;
      grid-template-columns: 78px 60px 80px 1fr;
      gap: 10px; padding: 5px 14px;
      border-bottom: 1px solid var(--line-soft);
      align-items: baseline;
    }
    .feed .row:hover { background: rgba(88,166,255,0.05); }
    .feed .row .ts { color: var(--fg-3); font-variant-numeric: tabular-nums; }
    .feed .row .type { text-transform: uppercase; font-size: 10.5px; letter-spacing: 0.06em; color: var(--c, var(--fg-2)); }
    .feed .row .ident { color: var(--accent-2); }
    .feed .row .msg { color: var(--fg-1); white-space: pre-wrap; word-break: break-word; }

    /* Token + status distribution */
    .charts { display: grid; grid-template-columns: 2fr 1fr; gap: 14px; }
    @media (max-width: 1100px) { .charts { grid-template-columns: 1fr; } }
    .chart-card {
      background: var(--bg-1);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px 16px;
      box-shadow: var(--shadow);
    }
    .chart-card h3 {
      margin: 0 0 8px; font-size: 12px; text-transform: uppercase;
      letter-spacing: 0.06em; color: var(--fg-2); font-weight: 500;
    }
    .dist-bar { display: flex; height: 10px; border-radius: 6px; overflow: hidden; background: var(--bg-2); }
    .dist-bar .seg { transition: flex 0.4s; }
    .dist-legend {
      display: grid; grid-template-columns: repeat(2, 1fr); gap: 4px 12px;
      margin-top: 10px; font-size: 11.5px; color: var(--fg-2);
    }
    .dist-legend .it { display: flex; align-items: center; gap: 6px; }
    .dist-legend .sw { width: 10px; height: 10px; border-radius: 3px; background: var(--c, var(--accent)); }
    .dist-legend .n { color: var(--fg-0); font-variant-numeric: tabular-nums; margin-left: auto; }

    /* Scrollbars */
    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: var(--bg-3); border-radius: 5px; }
    ::-webkit-scrollbar-thumb:hover { background: #2a3445; }

    /* Utility */
    .small { font-size: 11.5px; color: var(--fg-2); }
    .muted { color: var(--fg-3); }
    .nowrap { white-space: nowrap; }
    .right { text-align: right; }
    .hidden { display: none !important; }
    .session-link {
      display: inline-flex; align-items: center; gap: 4px;
      margin-left: 6px; padding: 1px 7px;
      font-size: 11px; font-weight: 500;
      border-radius: 4px;
      background: rgba(88,166,255,0.10);
      color: var(--accent);
      border: 1px solid rgba(88,166,255,0.20);
      text-decoration: none;
      white-space: nowrap;
      transition: background 0.15s;
    }
    .session-link:hover { background: rgba(88,166,255,0.20); text-decoration: none; }
    .session-link::before { content: "▶"; font-size: 8px; }
  </style>
</head>
<body>
  <div class="app">
    <div class="header">
      <div class="brand">
        <div class="logo"></div>
        <div>ClawCodex <span class="muted">·</span> Orchestrator LiveView</div>
      </div>
      <div class="meta">
        <div class="kv">Workspace <b id="hdr-workspace" class="mono">…</b></div>
        <div class="kv">Daemon <b id="hdr-daemon">…</b></div>
        <div class="kv">Uptime <b id="hdr-uptime">—</b></div>
        <div class="kv">Pull Requests <b id="hdr-prs">0</b></div>
        <div class="kv">Event Count <b id="hdr-events">0</b></div>
      </div>
      <div id="conn" class="conn"><span class="dot"></span><span id="conn-text">connecting…</span></div>
    </div>

    <div class="status-grid" id="status-grid"></div>

    <div class="main">
      <div class="panel">
        <div class="panel-header">
          <h3>Issues</h3>
          <span class="small muted" id="issue-count">0</span>
          <div class="actions">
            <input id="filter" type="search" placeholder="Filter by id / branch…">
            <select id="status-filter">
              <option value="">All statuses</option>
            </select>
          </div>
        </div>
        <div class="panel-body">
          <table class="issues">
            <thead>
              <tr>
                <th>Status</th>
                <th>Identifier</th>
                <th>Branch</th>
                <th>PR</th>
                <th>Workspace</th>
                <th class="right">Attempts</th>
                <th class="right">Idle</th>
              </tr>
            </thead>
            <tbody id="issues-body">
              <tr><td colspan="7" class="empty">Loading…</td></tr>
            </tbody>
          </table>
        </div>
      </div>

      <div class="panel">
        <div class="panel-header">
          <h3>Detail</h3>
          <div class="actions">
            <span class="small muted" id="detail-sub">Select an issue to inspect</span>
          </div>
        </div>
        <div class="panel-body">
          <div id="detail" class="detail">
            <div class="empty">Click an issue on the left to see details, branch, commit, PR, and recent events.</div>
          </div>
        </div>
      </div>
    </div>

    <div class="charts">
      <div class="chart-card">
        <h3>Event Type Distribution</h3>
        <div class="dist-bar" id="dist-bar"></div>
        <div class="dist-legend" id="dist-legend"></div>
      </div>
      <div class="chart-card">
        <h3>Token Activity</h3>
        <div class="small" id="tok-summary" style="color:var(--fg-1)">
          <div>Active sessions: <b id="tok-active">0</b></div>
          <div>Turns: <b id="tok-turns">0</b> | Tools: <b id="tok-events">0</b></div>
        </div>
      </div>
    </div>

    <div class="feed">
      <div class="panel-header">
        <h3>Live Event Feed</h3>
        <span class="small muted" id="feed-count">0 events</span>
        <div class="actions">
          <select id="feed-type">
            <option value="">All event types</option>
            <option value="tool_call">Tool Call</option>
            <option value="tool_result">Tool Result</option>
            <option value="agent_text">Agent Text</option>
          </select>
          <button id="feed-clear" style="background:var(--bg-2);border:1px solid var(--line);color:var(--fg-1);border-radius:6px;padding:4px 8px;cursor:pointer;">Clear</button>
        </div>
      </div>
      <div class="panel-body" id="feed-body">
        <div class="empty">Waiting for events…</div>
      </div>
    </div>
  </div>

  <script>
    "use strict";

    // ---- Constants ---------------------------------------------------------
    const STATUS_META = __STATUS_META__;
    const STATUSES = Object.keys(STATUS_META);
    const ACTIVE = new Set(STATUSES.filter(s => STATUS_META[s].group === "active"));
    const TERMINAL = new Set(STATUSES.filter(s => STATUS_META[s].group === "terminal"));

    const EVENT_TYPE_META = {
      tool_call:      { color: "#a371f7", label: "Tool Call" },
      tool_result:    { color: "#3fb950", label: "Tool Result" },
      agent_text:     { color: "#79c0ff", label: "Agent Text" },
    };
    const MAX_FEED_ROWS = 500;

    // ---- State -------------------------------------------------------------
    const state = {
      snapshot: null,
      selectedIssueId: null,
      _lastDetailSig: "",   // signature of last rendered detail panel
      _feedClearedTs: 0,    // Unix ts of last "Clear" click; events older than this are suppressed
      filter: "",
      statusFilter: "",
      feedTypeFilter: "",
      feed: [],            // newest first
      offsets: {},         // last seen file offsets (per issue_id)
    };

    // ---- DOM refs ----------------------------------------------------------
    const $ = (id) => document.getElementById(id);
    const el = {
      conn: $("conn"), connText: $("conn-text"),
      hdrWorkspace: $("hdr-workspace"), hdrDaemon: $("hdr-daemon"),
      hdrUptime: $("hdr-uptime"), hdrPrs: $("hdr-prs"), hdrEvents: $("hdr-events"),
      statusGrid: $("status-grid"),
      issueCount: $("issue-count"),
      issuesBody: $("issues-body"),
      filter: $("filter"),
      statusFilter: $("status-filter"),
      feedBody: $("feed-body"), feedCount: $("feed-count"),
      feedType: $("feed-type"), feedClear: $("feed-clear"),
      detail: $("detail"), detailSub: $("detail-sub"),
      distBar: $("dist-bar"), distLegend: $("dist-legend"),
      tokActive: $("tok-active"), tokEvents: $("tok-events"),
      tokTurns: $("tok-turns"),
    };

    // ---- Utilities ---------------------------------------------------------
    function fmtAge(seconds) {
      if (seconds == null) return "—";
      if (seconds < 60) return seconds + "s";
      if (seconds < 3600) return Math.floor(seconds/60) + "m " + (seconds%60) + "s";
      const h = Math.floor(seconds/3600);
      const m = Math.floor((seconds%3600)/60);
      return h + "h " + m + "m";
    }
    function fmtUptime(seconds) {
      if (!seconds) return "—";
      if (seconds < 60) return seconds + "s";
      if (seconds < 3600) return Math.floor(seconds/60) + "m";
      const h = Math.floor(seconds/3600);
      const m = Math.floor((seconds%3600)/60);
      return h + "h " + m + "m";
    }
    function fmtNumber(n) {
      if (n == null) return "0";
      if (n < 1000) return String(n);
      if (n < 1_000_000) return (n/1000).toFixed(n < 10000 ? 1 : 0) + "k";
      return (n/1_000_000).toFixed(1) + "M";
    }
    function shortSha(sha) { return sha ? sha.slice(0, 7) : ""; }
    function escapeHtml(s) {
      return String(s == null ? "" : s)
        .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
    }
    function truncate(s, n) {
      s = s == null ? "" : String(s);
      return s.length > n ? s.slice(0, n - 1) + "…" : s;
    }

    // ---- Event body helpers (shared by renderFeed + renderDetail) --------
    function toolCallBody(ev) {
      const name = escapeHtml(ev.tool_name || "?");
      const p = ev.params;
      let detail = "";
      if (p && typeof p === "object" && !Array.isArray(p)) {
        // Tool-specific extraction for readability
        const cmd = p.command || p.cmd;
        const path = p.file_path || p.path || p.pathname;
        const pattern = p.pattern || p.query || p.regex;
        const url = p.url;
        if (cmd) detail = truncate(cmd, 120);
        else if (path) detail = truncate(path, 100);
        else if (pattern) detail = truncate(pattern, 80);
        else if (url) detail = truncate(url, 100);
        else {
          const vals = Object.values(p).filter(v => v != null);
          if (vals.length > 0) detail = truncate(String(vals[0]), 100);
        }
      } else if (typeof p === "string") {
        detail = truncate(p, 120);
      }
      return `<b>${name}</b>` + (detail ? ` <span class="muted">${escapeHtml(detail)}</span>` : "");
    }
    function toolResultBody(ev) {
      const name = escapeHtml(ev.tool_name || "?");
      const err = ev.is_error ? ' <span style="color:var(--bad)">[error]</span>' : "";
      let detail = "";
      const c = ev.result_content;
      if (c) {
        detail = typeof c === "string" ? c : (Array.isArray(c) ? c.map(b => (b && b.text) || "").join("") : JSON.stringify(c));
        detail = truncate(detail, 200);
      }
      return `<b>${name}</b>${err}` + (detail ? ` <span class="muted">${escapeHtml(detail)}</span>` : "");
    }
    function eventBody(ev) {
      const t = ev.type || "?";
      if (t === "tool_call") return toolCallBody(ev);
      if (t === "tool_result") return toolResultBody(ev);
      if (t === "agent_text") return `<span>${escapeHtml(truncate(ev.content || "", 160))}</span>`;
      return `<span class="mono">${escapeHtml(truncate(JSON.stringify(ev), 140))}</span>`;
    }

    function pillFor(status) {
      const m = STATUS_META[status] || STATUS_META.pending;
      return `<span class="pill" style="--c:${m.color}"><span class="ico">${m.icon}</span>${m.label}</span>`;
    }

    // ---- Rendering ---------------------------------------------------------
    function renderStatusGrid(byStatus) {
      const html = STATUSES.map(s => {
        const m = STATUS_META[s];
        const n = byStatus[s] || 0;
        const isZero = n === 0 ? " zero" : "";
        const groupLabel = m.group === "active" ? "active" : "terminal";
        return `
          <div class="stat${isZero}" style="--c:${m.color}">
            <div class="label"><span class="ico" style="color:${m.color}">${m.icon}</span>${m.label}<span class="muted" style="margin-left:auto">${groupLabel}</span></div>
            <div class="value">${n}</div>
            <div class="sub">${STATUSES.indexOf(s) >= 0 ? "issues in state" : ""}</div>
          </div>`;
      }).join("");
      el.statusGrid.innerHTML = html;
    }

    function renderIssues(issues) {
      const f = state.filter.trim().toLowerCase();
      const sf = state.statusFilter;
      const filtered = issues.filter(i => {
        if (sf && i.status !== sf) return false;
        if (f) {
          const hay = [i.identifier, i.branch_name, i.workspace_path].filter(Boolean).join(" ").toLowerCase();
          if (!hay.includes(f)) return false;
        }
        return true;
      });
      el.issueCount.textContent = filtered.length + " / " + issues.length;
      if (filtered.length === 0) {
        el.issuesBody.innerHTML = `<tr><td colspan="7" class="empty">No issues match the current filter.</td></tr>`;
        return;
      }
      const rows = filtered.map(i => {
        const pr = i.pr_number
          ? `<a href="${escapeHtml(i.pr_url || "#")}" target="_blank" rel="noopener">#${escapeHtml(i.pr_number)}</a>`
          : `<span class="muted">—</span>`;
        const branch = i.branch_name
          ? `<span class="mono">${escapeHtml(truncate(i.branch_name, 28))}</span>`
          : `<span class="muted">—</span>`;
        const ws = i.workspace_short
          ? `<span class="mono" title="${escapeHtml(i.workspace_path || "")}">${escapeHtml(truncate(i.workspace_short, 30))}</span>`
          : `<span class="muted">—</span>`;
        const sel = i.issue_id === state.selectedIssueId ? " selected" : "";
        const sessionLink = i.run_id
          ? `<a class="session-link" href="#" onclick="openSessionViewer('${escapeHtml(i.run_id)}');return false;">Session</a>`
          : '';
        return `
          <tr class="issue-row${sel}" data-id="${escapeHtml(i.issue_id)}">
            <td>${pillFor(i.status)}</td>
            <td class="identifier">${escapeHtml(i.identifier)} ${sessionLink}</td>
            <td class="branch">${branch}</td>
            <td>${pr}</td>
            <td class="workspace">${ws}</td>
            <td class="num">${i.attempt_count || 0}</td>
            <td class="num">${fmtAge(i.idle_seconds)}</td>
          </tr>`;
      }).join("");
      el.issuesBody.innerHTML = rows;
      el.issuesBody.querySelectorAll(".issue-row").forEach(tr => {
        tr.addEventListener("click", () => selectIssue(tr.getAttribute("data-id")));
      });
    }

    function renderDistribution(byType) {
      const entries = Object.entries(byType || {});
      const total = entries.reduce((a, [, n]) => a + n, 0);
      if (total === 0) {
        el.distBar.innerHTML = `<div class="seg" style="flex:1;background:var(--bg-3)"></div>`;
        el.distLegend.innerHTML = `<div class="muted small">No events yet</div>`;
        return;
      }
      el.distBar.innerHTML = entries.map(([t, n]) => {
        const m = EVENT_TYPE_META[t] || { color: "#8b949e", label: t };
        return `<div class="seg" style="flex:${n};background:${m.color}" title="${m.label}: ${n}"></div>`;
      }).join("");
      el.distLegend.innerHTML = entries.map(([t, n]) => {
        const m = EVENT_TYPE_META[t] || { color: "#8b949e", label: t };
        const pct = ((n / total) * 100).toFixed(1);
        return `<div class="it"><span class="sw" style="--c:${m.color};background:${m.color}"></span>${m.label}<span class="n">${fmtNumber(n)} (${pct}%)</span></div>`;
      }).join("");
    }

    function renderHeader(snap) {
      el.hdrWorkspace.textContent = snap.workspace || "—";
      el.hdrWorkspace.title = snap.workspace || "";
      const m = snap.metadata || {};
      if (m.found) {
        el.hdrDaemon.innerHTML = m.alive
          ? `<span style="color:var(--good)">●</span> running (PID ${m.pid || "?"})`
          : `<span style="color:var(--bad)">●</span> stopped`;
        el.hdrDaemon.title = m.metadata_path || "";
        el.hdrUptime.textContent = m.alive ? fmtUptime(m.uptime_seconds) : "—";
      } else {
        el.hdrDaemon.innerHTML = `<span class="muted">not detected</span>`;
        el.hdrUptime.textContent = "—";
      }
      el.hdrPrs.textContent = (snap.issues && snap.issues.totals && snap.issues.totals.prs) || 0;
      el.hdrEvents.textContent = (snap.events && snap.events.total) || 0;
      // Token Activity: active sessions / turns / tools
      const ta = snap.token_activity || {};
      el.tokActive.textContent = ta.active_sessions || 0;
      el.tokTurns.textContent = ta.total_turns || 0;
      el.tokEvents.textContent = ta.total_tools || 0;
    }

    function renderDetail(issueId) {
      if (!state.snapshot) return;
      const issues = state.snapshot.issues.issues || [];
      const issue = issues.find(i => i.issue_id === issueId);
      if (!issue) {
        el.detail.innerHTML = `<div class="empty">Issue not found.</div>`;
        el.detailSub.textContent = "—";
        return;
      }
      const m = STATUS_META[issue.status] || STATUS_META.pending;
      const events = (state.snapshot.events && state.snapshot.events.recent || [])
        .filter(e => e.issue_id === issueId)
        .slice(0, 20);
      el.detailSub.innerHTML = `${pillFor(issue.status)} <span class="muted small">· ${issue.workspace_strategy || "—"} strategy</span>`;
      const rows = [
        ["Identifier",   `<b>${escapeHtml(issue.identifier)}</b>`],
        ["Issue ID",     `<span class="mono">${escapeHtml(issue.issue_id)}</span>`],
        ["Status",       pillFor(issue.status)],
        ["Session",      issue.run_id
            ? `<a href="#" onclick="openSessionViewer('${escapeHtml(issue.run_id)}');return false;">View Session Timeline →</a>`
            : `<span class="muted">—</span>`],
        ["Branch",       issue.branch_name ? `<span class="mono">${escapeHtml(issue.branch_name)}</span>` : `<span class="muted">—</span>`],
        ["Base branch",  `<span class="mono">${escapeHtml(issue.base_branch || "main")}</span>`],
        ["Commit",       issue.commit_sha ? `<span class="mono">${escapeHtml(shortSha(issue.commit_sha))}</span> <span class="muted">${escapeHtml(issue.commit_sha)}</span>` : `<span class="muted">—</span>`],
        ["Pull request", issue.pr_number ? `<a href="${escapeHtml(issue.pr_url || "#")}" target="_blank" rel="noopener">#${escapeHtml(issue.pr_number)}</a>` : `<span class="muted">—</span>`],
        ["Workspace",    issue.workspace_path ? `<span class="mono" title="${escapeHtml(issue.workspace_path)}">${escapeHtml(issue.workspace_path)}</span>` : `<span class="muted">—</span>`],
        ["Sequence",     issue.sequence_index != null ? `#${issue.sequence_index}` : `<span class="muted">—</span>`],
        ["Intent",       `<span class="mono">${escapeHtml(issue.intent || "none")}</span>`],
        ["Attempts",     String(issue.attempt_count || 0)],
        ["Retries",      String(issue.retry_count || 0)],
        ["Verification", issue.verification_status ? escapeHtml(issue.verification_status) : `<span class="muted">—</span>`],
        ["Clarification",issue.clarification_status ? escapeHtml(issue.clarification_status) : `<span class="muted">—</span>`],
        ["Created",      issue.created_at ? new Date(issue.created_at * 1000).toLocaleString() : `<span class="muted">—</span>`],
        ["Updated",      issue.updated_at ? new Date(issue.updated_at * 1000).toLocaleString() : `<span class="muted">—</span>`],
        ["Idle",         fmtAge(issue.idle_seconds)],
        ["Age",          fmtAge(issue.age_seconds)],
      ];
      const eventsHtml = events.length === 0
        ? `<div class="empty">No recent events for this issue.</div>`
        : events.map(e => {
            const ev = e.event || {};
            const t = ev.type || "?";
            const m2 = EVENT_TYPE_META[t] || { color: "#8b949e", label: t.toUpperCase() };
            const body = eventBody(ev);
            return `<div class="row" style="--c:${m2.color}"><span class="ts">${escapeHtml(e.timestamp || "")}</span><span class="type" style="color:${m2.color}">${m2.label}</span><span>${body}</span></div>`;
          }).join("");

      el.detail.innerHTML = `
        <h2>${escapeHtml(issue.identifier)} ${pillFor(issue.status)}</h2>
        <div class="sub mono">${escapeHtml(issue.issue_id)}</div>
        <div class="grid">
          ${rows.map(([k, v]) => `<div class="k">${k}</div><div class="v">${v}</div>`).join("")}
        </div>
        <div class="section-title">Recent Events (${events.length})</div>
        <div class="events-mini">${eventsHtml}</div>
      `;
    }

    function renderFeed() {
      const f = state.feedTypeFilter;
      const rows = state.feed.filter(e => !f || (e.event && e.event.type === f));
      el.feedCount.textContent = rows.length + " events";
      if (rows.length === 0) {
        el.feedBody.innerHTML = `<div class="empty">No events match the current filter.</div>`;
        return;
      }
      el.feedBody.innerHTML = rows.slice(0, MAX_FEED_ROWS).map(e => {
        const ev = e.event || {};
        const t = ev.type || "?";
        const m = EVENT_TYPE_META[t] || { color: "#8b949e", label: t.toUpperCase() };
        const body = eventBody(ev);
        return `<div class="row" style="--c:${m.color}"><span class="ts">${escapeHtml(e.timestamp || "")}</span><span class="type" style="color:${m.color}">${m.label}</span><span class="ident">${escapeHtml(identifierFor(e.issue_id))}</span><span class="msg">${body}</span></div>`;
      }).join("");
    }

    function identifierFor(issueId) {
      if (!state.snapshot) return issueId;
      const issue = (state.snapshot.issues.issues || []).find(i => i.issue_id === issueId);
      return issue ? issue.identifier : issueId;
    }

    function renderStatusFilter() {
      if (!state.snapshot) return;
      const counts = state.snapshot.issues.by_status || {};
      const current = state.statusFilter;
      const opts = [`<option value="">All statuses</option>`].concat(
        STATUSES.map(s => `<option value="${s}" ${s === current ? "selected" : ""}>${STATUS_META[s].label} (${counts[s] || 0})</option>`)
      );
      el.statusFilter.innerHTML = opts.join("");
    }

    function renderAll() {
      if (!state.snapshot) return;
      renderHeader(state.snapshot);
      renderStatusGrid(state.snapshot.issues.by_status || {});
      renderStatusFilter();
      renderIssues(state.snapshot.issues.issues || []);
      renderDistribution((state.snapshot.events && state.snapshot.events.by_type) || {});
      // Only re-render detail panel when its data actually changed.
      // This prevents the "Recent Events" list from being rebuilt every 0.5s,
      // which would reset scroll position and make it unreadable.
      if (state.selectedIssueId) {
        const issue = (state.snapshot.issues.issues || []).find(i => i.issue_id === state.selectedIssueId);
        if (issue) {
          const evtCount = (state.snapshot.events && state.snapshot.events.recent || [])
            .filter(e => e.issue_id === state.selectedIssueId).length;
          const sig = state.selectedIssueId + '|' + issue.status + '|' + issue.updated_at + '|' + evtCount;
          if (sig !== state._lastDetailSig) {
            renderDetail(state.selectedIssueId);
            state._lastDetailSig = sig;
          }
        }
      }
      renderFeed();
    }

    function applySnapshot(snap) {
      state.snapshot = snap;
      if (!snap.events) snap.events = { by_type: {}, total: 0, recent: [] };

      // Normalize events.recent in-place (EventTailerManager format → renderFeed/renderDetail format)
      if (snap.events.recent) {
        snap.events.recent = snap.events.recent.map(function(e) {
          // Already normalized (e.g. from a previous applySnapshot pass)
          if (e.event) return e;
          return {
            event: {
              type: e.event_type || '?',
              tool_name: (e.data && e.data.tool) || (e.data && e.data.tool_name) || '?',
              approved: e.data && e.data.approved,
              turn: e.data && e.data.turn,
              deny_reason: e.data && e.data.deny_reason,
              is_error: e.data && e.data.is_error,
              params: e.data && e.data.params,
              result_content: e.data && e.data.result_content,
              content: e.data && e.data.content,
            },
            timestamp: e.ts ? new Date(e.ts * 1000).toLocaleTimeString("en-GB", { hour12: false }) : "",
            issue_id: e.issue_id,
            ts: e.ts,
          };
        });
      }

      // Incremental feed update: de-duplicate by ts+issue_id, respect Clear timestamp
      if (snap.events.recent && snap.events.recent.length > 0) {
        const clearedTs = state._feedClearedTs || 0;
        const existingKeys = new Set(state.feed.map(e => e.ts + '|' + e.issue_id));
        const newEvents = snap.events.recent
          .filter(e => !existingKeys.has(e.ts + '|' + e.issue_id))
          .filter(e => !clearedTs || e.ts > clearedTs)
          .reverse();
        newEvents.forEach(e => state.feed.unshift(e));
        if (state.feed.length > MAX_FEED_ROWS * 2) state.feed.length = MAX_FEED_ROWS * 2;
      }

      renderAll();
    }

    function selectIssue(issueId) {
      state.selectedIssueId = state.selectedIssueId === issueId ? null : issueId;
      state._lastDetailSig = "";  // force re-render on manual selection
      renderIssues(state.snapshot.issues.issues || []);
      if (state.selectedIssueId) renderDetail(state.selectedIssueId);
      else {
        el.detail.innerHTML = `<div class="empty">Click an issue on the left to see details, branch, commit, PR, and recent events.</div>`;
        el.detailSub.textContent = "Select an issue to inspect";
      }
    }

    // ---- Event handling ----------------------------------------------------
    function pushEvent(evt) {
      state.feed.unshift(evt);
      if (state.feed.length > MAX_FEED_ROWS * 2) state.feed.length = MAX_FEED_ROWS * 2;
      // Increment by_type counter and update distribution live.
      if (state.snapshot && evt.event && evt.event.type) {
        if (!state.snapshot.events) state.snapshot.events = { by_type: {}, total: 0, recent: [] };
        const bt = state.snapshot.events.by_type || (state.snapshot.events.by_type = {});
        bt[evt.event.type] = (bt[evt.event.type] || 0) + 1;
        state.snapshot.events.total = (state.snapshot.events.total || 0) + 1;
        renderDistribution(bt);
        el.hdrEvents.textContent = state.snapshot.events.total;
      }
      renderFeed();
    }

    // ---- Session Viewer ----------------------------------------------------
    let _viewerPort = null;

    async function openSessionViewer(runId) {
      try {
        // If we have a cached port, probe it first; reset on failure.
        if (_viewerPort) {
          try {
            const probe = await fetch(`http://localhost:${_viewerPort}/api/viz/health`, { method: 'HEAD' });
            if (probe.ok) {
              window.open(`http://localhost:${_viewerPort}/session/${encodeURIComponent(runId)}`, '_blank');
              return;
            }
          } catch (_) { /* viewer was killed (idle timeout) — reset and retry */ }
          _viewerPort = null;
        }
        const resp = await fetch('/api/session-viewer/start', { method: 'POST' });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const { port } = await resp.json();
        _viewerPort = port;
        window.open(`http://localhost:${port}/session/${encodeURIComponent(runId)}`, '_blank');
      } catch (e) {
        console.error('Failed to start session viewer:', e);
        _viewerPort = null;
        alert('Failed to start the session viewer, please retry later.');
      }
    }

    // ---- SSE ---------------------------------------------------------------
    let es = null;
    function connect() {
      if (es) { try { es.close(); } catch (e) {} }
      es = new EventSource("/events");
      es.onopen = () => {
        el.conn.classList.remove("bad"); el.conn.classList.add("ok");
        el.connText.textContent = "connected";
      };
      es.onerror = () => {
        el.conn.classList.remove("ok"); el.conn.classList.add("bad");
        el.connText.textContent = "disconnected — retrying…";
      };
      es.onmessage = (msg) => {
        let data;
        try { data = JSON.parse(msg.data); } catch (e) { return; }
        if (data.type === "snapshot") {
          applySnapshot(data);
        } else if (data.type === "event") {
          pushEvent(data);
        }
      };
    }

    // ---- Wire up -----------------------------------------------------------
    el.filter.addEventListener("input", (e) => {
      state.filter = e.target.value;
      if (state.snapshot) renderIssues(state.snapshot.issues.issues || []);
    });
    el.statusFilter.addEventListener("change", (e) => {
      state.statusFilter = e.target.value;
      if (state.snapshot) renderIssues(state.snapshot.issues.issues || []);
    });
    el.feedType.addEventListener("change", (e) => {
      state.feedTypeFilter = e.target.value;
      renderFeed();
    });
    el.feedClear.addEventListener("click", () => {
      state._feedClearedTs = Date.now() / 1000;
      state.feed = [];
      renderFeed();
    });

    connect();
    // Periodic refresh to keep header idle/uptime counters ticking even when
    // the server is idle.
    setInterval(() => {
      if (state.snapshot) renderHeader(state.snapshot);
    }, 1000);
  </script>
</body>
</html>"""

# Trae IDE Integration Guide

> Lets Trae CN call clawcodex's Orchestrator / SOP Compiler / Skills / stability gate directly from the chat panel.

## Architecture

```text
Trae CN (Windows process)
  └─ mcp.json registration → wsl.exe -d Ubuntu-24.04 -- bash -lc "python3 -m extensions.trae.mcp_bridge"
       └─ extensions/trae/mcp_bridge.py (stdio MCP server inside WSL)
            ├─ clawcodex_orchestrator_run_issue  (fire-and-forget, returns run_id)
            ├─ clawcodex_sop_compile             (calls convert_sop_to_agent)
            ├─ clawcodex_skill_invoke            (SkillRegistryExt parses the prompt)
            └─ clawcodex_stability_gate          (subprocess pytest)
```

**Why wsl.exe**: Trae CN is a native Windows process, while clawcodex's dependencies (pytest, extensions/sop_converter, extensions/skills_ext) are installed inside WSL Ubuntu-24.04. The MCP stdio protocol communicates over stdin/stdout, and `wsl.exe` transparently forwards stdin/stdout to the in-WSL process, so cross-environment operation works.

**Automatic path conversion**: `{workspaceFolder}` passed by Trae CN is a Windows path (`C:\xxx`). `BridgeConfig.from_env` automatically calls `_win_to_wsl` to convert it to `/mnt/c/xxx`. To disable this (pure-Linux deploys), set `CLAWCODEX_AUTO_WIN_TO_WSL=0`.

## Setup Steps

### 1. Confirm the WSL distro name

```bash
wsl.exe -l -v
# The first non-header output line is the distro name; this machine uses Ubuntu-24.04
```

### 2. Write Trae CN's mcp.json

File location: `%APPDATA%\Trae CN\User\mcp.json` (i.e. `C:\Users\<username>\AppData\Roaming\Trae CN\User\mcp.json`).

```jsonc
{
  "mcpServers": {
    "clawcodex": {
      "command": "C:\\Windows\\System32\\wsl.exe",
      "args": [
        "-d", "Ubuntu-24.04",            // ← replace with your distro name
        "--",
        "bash", "-lc",
        "cd /mnt/c/WorkSpace/clawcodex && CLAWCODEX_WORKSPACE=/mnt/c/WorkSpace/clawcodex CLAWCODEX_REPORTS_DIR=/mnt/c/WorkSpace/clawcodex/.reports/ python3 -m extensions.trae.mcp_bridge"
      ],
      "env": {
        "CLAWCODEX_AUTO_WIN_TO_WSL": "1"  // automatic Windows→WSL path conversion
      }
    }
  }
}
```

> **Note**: `cd /mnt/c/WorkSpace/clawcodex` is the clawcodex repo path inside WSL — adjust it to your actual location. `python3` must be able to find the extensions.trae module (run from the repo root; no pip install needed).

### 3. Restart Trae CN

Trae CN loads mcp.json at startup. Fully quit Trae CN (tray icon → Quit) after editing, then start it again.

### 4. Verify the integration

Type into the Trae CN chat panel:

> Run the stability gate with clawcodex

Trae AI should call the `clawcodex_stability_gate` tool and return a summary like `exit=0 | 345 passed in 48.23s`.

Alternatively, in Trae CN's MCP panel (Settings → AI → MCP Servers), confirm the `clawcodex` server status is `connected` and the tool list shows the 4 `clawcodex_*` tools.

## Tool Reference

| Tool | Inputs | Returns | Latency |
|------|--------|---------|---------|
| `clawcodex_orchestrator_run_issue` | `issue_url` (required), `workflow_path` | `queued run_id=<uuid>` | immediate (fire-and-forget) |
| `clawcodex_sop_compile` | `sdk_spec` (required), `requirements`, `agent_name` | `compiled agent=... skills=N persist=...` | seconds |
| `clawcodex_skill_invoke` | `skill_name` (required), `params` | skill prompt text | immediate |
| `clawcodex_stability_gate` | (none) | `exit=0 \| N passed in Xs` | 30-60s |

**Long-running task queries**: `clawcodex_orchestrator_run_issue` is fire-and-forget — it returns a run_id immediately. Actual progress is written to `<reports_dir>/<run_id>.ndjson`; ask "check the progress of run_id xxx" in Trae again to trigger file polling.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Trae MCP panel shows clawcodex failed | wrong wsl.exe distro name | run `wsl.exe -l -v`, fix the `-d` argument in mcp.json |
| Tool call reports `mcp SDK not installed` | mcp not installed inside WSL | `pip install mcp` |
| stability_gate reports `pytest not found` | pytest not installed inside WSL | `pip install pytest` |
| Tool call reports `ModuleNotFoundError: extensions.trae` | wrong cd path | confirm `cd` in mcp.json points at the clawcodex repo root |
| Trae chat times out | stability_gate runs >120s | `CLAWCODEX_REPORTS_DIR` in mcp.json args has no effect; shrink the test set or raise `stability_gate_timeout_s` |

## Rollback

Delete the `clawcodex` section from mcp.json and restart Trae CN. `extensions/trae/` lives entirely in Layer 2 and does not affect `src/` or `clawcodex_ext/`.

## Acceptance Checklist

- [x] `python -m extensions.trae.mcp_bridge` starts standalone; `tools/list` returns 4 tools
- [x] Unit tests: `tests/trae/test_mcp_bridge.py` 31 passed + 2 skipped
- [x] E2E: full Trae CN chain (wsl.exe → bash -lc → python -m) `tools/list` returns 4 tools; `clawcodex_stability_gate` real run 345 passed
- [x] Windows→WSL automatic path conversion: `_win_to_wsl` + `BridgeConfig.from_env` verified
- [ ] `mcp inspector` schema check — pending manual run (needs `npx @modelcontextprotocol/inspector`)

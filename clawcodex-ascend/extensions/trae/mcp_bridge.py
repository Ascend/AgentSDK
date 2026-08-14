# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------
#
# Copyright (c) 2026 Clawd Codex Team
# SPDX-License-Identifier: MIT
# Source: https://github.com/agentforce314/clawcodex
# ClawCodex-derived portions remain licensed under the MIT License.
# See clawcodex-ascend/LICENSE.clawcodex.
#
"""Trae IDE MCP reverse bridge.

Lets Trae IDE users call clawcodex downstream capabilities (Orchestrator,
SOP Compiler, Skills bridge, stability gate) from the chat pane. Trae IDE
will not implement ACP in the near term (see trae-agent #344) but natively
supports MCP (``byted-solo.builtin-mcp``), so clawcodex exposes a stdio
MCP server and Trae connects to it. Layer 2 decoupling — no pollution of
``src/`` or ``clawcodex_ext/``.

``mcp`` is an optional dependency (``pip install mcp``). When absent,
:class:`TraeMcpBridge` can still be instantiated and its tool specs and
dispatch logic unit-tested; only :meth:`run_stdio` requires ``mcp``.

Trae IDE side setup (mcp.json config for Trae CN via wsl.exe, or direct
byted-solo.builtin-mcp connection for Trae AI): see
``extensions/trae/README.md``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import subprocess
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from extensions.capabilities.acp_protocol import ACPToolSpec

logger = logging.getLogger(__name__)

__all__ = [
    "TraeMcpBridge",
    "BridgeConfig",
    "MCP_UNAVAILABLE",
    "mcp_available",
    "build_tool_specs",
]

MCP_UNAVAILABLE = "mcp SDK not installed (pip install mcp). TraeMcpBridge.run_stdio() unavailable."


def mcp_available() -> bool:
    """Return True if the optional ``mcp`` SDK is importable."""
    try:
        import mcp  # noqa: F401
    except ImportError:
        return False
    return True


@dataclass
class BridgeConfig:
    """MCP bridge runtime config (constructible from environment variables)."""

    workspace: str = ""
    reports_dir: str = ""
    stability_gate_args: list[str] = field(
        default_factory=lambda: [
            sys.executable,
            "-m",
            "pytest",
            "tests/stability_gate/",
            "-q",
            "--tb=line",
            "-x",
        ]
    )
    stability_gate_cwd: str = ""
    stability_gate_timeout_s: float = 120.0
    progress_poll_interval_s: float = 0.5
    # Auto-convert Windows paths (C:\xxx) to WSL paths (/mnt/c/xxx)
    auto_win_to_wsl: bool = True

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "BridgeConfig":
        env = env if env is not None else dict(os.environ)
        workspace = env.get("CLAWCODEX_WORKSPACE", "")
        reports_dir = env.get("CLAWCODEX_REPORTS_DIR", "")
        if env.get("CLAWCODEX_AUTO_WIN_TO_WSL", "1") not in ("0", "false", "False"):
            workspace = _win_to_wsl(workspace) if workspace else workspace
            reports_dir = _win_to_wsl(reports_dir) if reports_dir else reports_dir
        return cls(
            workspace=workspace,
            reports_dir=reports_dir,
            stability_gate_cwd=workspace,
        )


def _win_to_wsl(path: str) -> str:
    """Convert a Windows path ``C:\\foo\\bar`` to WSL ``/mnt/c/foo/bar``.

    Non-Windows paths (already starting with ``/``, empty, or UNC-like)
    are returned unchanged. Backslashes are normalized to forward
    slashes; a drive ``D:\\proj`` becomes ``/mnt/d/proj``.
    """
    if not path:
        return path
    p = path.strip().strip('"').strip("'")
    if p.startswith("/") or p.startswith("\\\\wsl"):
        return p
    if len(p) >= 2 and p[1] == ":" and p[0].isalpha():
        drive = p[0].lower()
        rest = p[2:].replace("\\", "/")
        if rest.startswith("/"):
            rest = rest[1:]
        return f"/mnt/{drive}/{rest}"
    return p


TOOL_ORCH_RUN = "clawcodex_orchestrator_run_issue"
TOOL_SOP_COMPILE = "clawcodex_sop_compile"
TOOL_SKILL_INVOKE = "clawcodex_skill_invoke"
TOOL_STABILITY_GATE = "clawcodex_stability_gate"


def build_tool_specs() -> list[ACPToolSpec]:
    """Return the 4 MCP tool specifications exposed to Trae.

    Exposed as a standalone function so unit tests can assert the
    schemas without instantiating the bridge.
    """
    return [
        ACPToolSpec(
            name=TOOL_ORCH_RUN,
            description=(
                "Derive a git workspace from the current workspace, run the "
                "clawcodex agent to process the issue and push a PR. Equivalent "
                "to a one-shot `clawcodex-dev orchestrator server start`. "
                "Returns a run_id; poll .reports/{run_id}.ndjson for progress."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "issue_url": {
                        "type": "string",
                        "description": "GitHub/Gitee/GitCode/Linear issue URL",
                    },
                    "workflow_path": {
                        "type": "string",
                        "description": "Optional SOP workflow.md path",
                    },
                },
                "required": ["issue_url"],
            },
        ),
        ACPToolSpec(
            name=TOOL_SOP_COMPILE,
            description=(
                "Compile an SDK spec into a reusable agent (calls "
                "sop_converter.convert_sop_to_agent); returns the agent "
                "definition and skill list."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "sdk_spec": {
                        "type": "string",
                        "description": "OpenAPI dict JSON / URL / method list",
                    },
                    "requirements": {
                        "type": "string",
                        "description": "Business requirements, used for skill grouping",
                    },
                    "agent_name": {"type": "string"},
                },
                "required": ["sdk_spec"],
            },
        ),
        ACPToolSpec(
            name=TOOL_SKILL_INVOKE,
            description=(
                "Invoke a registered Skill (resolved through the skill bridge "
                "layer: the SkillRegistryExt resolves the skill markdown prompt)."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "skill_name": {"type": "string"},
                    "params": {"type": "object", "default": {}},
                },
                "required": ["skill_name"],
            },
        ),
        ACPToolSpec(
            name=TOOL_STABILITY_GATE,
            description=(
                "Run the stability gate in the current workspace and return a "
                "Stage 1-9 pass/fail summary. Equivalent to `pytest tests/stability_gate/ -q`."
            ),
            input_schema={"type": "object", "properties": {}},
        ),
    ]


class TraeMcpBridge:
    """MCP server bridge — lets Trae IDE call clawcodex capabilities via MCP.

    Design notes:
      - ``mcp`` optional: the bridge stays constructible and unit-testable
        without it; only :meth:`run_stdio` requires ``mcp`` at call time.
      - Orchestrator integration is fire-and-forget: ``enqueue_issue``
        returns the run_id immediately (risk mitigation).
      - SOP compilation calls the real :func:`convert_sop_to_agent`.
      - Skill calls resolve the registered skill's prompt via :class:`SkillRegistryExt`.
      - The stability gate runs pytest in a subprocess and returns a summary.
    """

    def __init__(
        self,
        config: BridgeConfig | None = None,
        *,
        orchestrator_enqueue: Callable[[str, str | None], str] | None = None,
        sop_compiler: Callable[..., dict[str, Any]] | None = None,
        skill_invoker: Callable[[str, dict[str, Any]], str] | None = None,
        stability_runner: Callable[[], str] | None = None,
    ) -> None:
        self._config = config or BridgeConfig.from_env()
        self._orchestrator_enqueue = orchestrator_enqueue
        self._sop_compiler = sop_compiler
        self._skill_invoker = skill_invoker
        self._stability_runner = stability_runner
        self._runs: dict[str, Path] = {}
        self._server: Any | None = None

    # ---- Tool listing -----------------------------------------------------

    def list_tools(self) -> list[ACPToolSpec]:
        """Return tool specs (mcp-agnostic, used by both MCP layer & tests)."""
        return build_tool_specs()

    # ---- Tool dispatch ----------------------------------------------------

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Dispatch a tool call by name. Returns text content.

        Branches: fire-and-forget enqueue, real SOP compilation, skill
        bridge, and the pytest stability-gate subprocess.
        """
        if name == TOOL_ORCH_RUN:
            return await self._handle_orch_run(arguments)
        if name == TOOL_SOP_COMPILE:
            return await self._handle_sop_compile(arguments)
        if name == TOOL_SKILL_INVOKE:
            return await self._handle_skill_invoke(arguments)
        if name == TOOL_STABILITY_GATE:
            return await self._handle_stability_gate(arguments)
        return f"error: unknown tool: {name}"

    # ---- Per-tool handlers ------------------------------------------------

    async def _handle_orch_run(self, arguments: dict[str, Any]) -> str:
        issue_url = arguments.get("issue_url")
        if not issue_url:
            return "error: issue_url is required"
        workflow_path = arguments.get("workflow_path")
        enqueue = self._orchestrator_enqueue or self._default_orchestrator_enqueue
        # Fire-and-forget: enqueue returns the run_id immediately; long tasks run in the background
        try:
            run_id = enqueue(issue_url, workflow_path)
        except Exception as exc:  # noqa: BLE001 — boundary, surface to Trae
            logger.exception("orchestrator enqueue failed for %s", issue_url)
            return f"error: enqueue failed: {exc}"
        # Record the progress file path for later polling (Trae can call tools/call again)
        reports = Path(self._config.reports_dir or ".reports")
        self._runs[run_id] = reports / f"{run_id}.ndjson"
        return f"queued run_id={run_id} (progress: {self._runs[run_id]})"

    async def _handle_sop_compile(self, arguments: dict[str, Any]) -> str:
        sdk_spec = arguments.get("sdk_spec")
        if not sdk_spec:
            return "error: sdk_spec is required"
        compile_fn = self._sop_compiler or self._default_sop_compiler
        try:
            result = compile_fn(
                sdk_spec=sdk_spec,
                requirements=arguments.get("requirements", ""),
                agent_name=arguments.get("agent_name", ""),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("sop compile failed")
            return f"error: compile failed: {exc}"
        if isinstance(result, dict) and result.get("status") == "error":
            return f"error: {result.get('error', 'unknown')}"
        # Compact summary (avoid flooding the Trae chat pane with huge JSON)
        skill_count = len(result.get("skills", [])) if isinstance(result, dict) else 0
        agent_name = result.get("agent_type", "") if isinstance(result, dict) else ""
        persist = result.get("persist_status", "") if isinstance(result, dict) else ""
        return f"compiled agent={agent_name} skills={skill_count} persist={persist}"

    async def _handle_skill_invoke(self, arguments: dict[str, Any]) -> str:
        skill_name = arguments.get("skill_name")
        if not skill_name:
            return "error: skill_name is required"
        params = arguments.get("params", {}) or {}
        invoke = self._skill_invoker or self._default_skill_invoker
        try:
            return invoke(skill_name, params)
        except Exception as exc:  # noqa: BLE001
            logger.exception("skill invoke failed: %s", skill_name)
            return f"error: skill '{skill_name}' failed: {exc}"

    async def _handle_stability_gate(self, _arguments: dict[str, Any]) -> str:
        runner = self._stability_runner or self._default_stability_runner
        try:
            return await asyncio.to_thread(runner)
        except Exception as exc:  # noqa: BLE001
            logger.exception("stability gate failed")
            return f"error: stability gate failed: {exc}"

    # ---- Default production implementations (lazily load the real modules) --

    def _default_orchestrator_enqueue(self, issue_url: str, workflow_path: str | None) -> str:
        """Default orchestrator enqueue — generates run_id and records intent.

        A real Orchestrator needs heavy dependencies unsuitable for the
        long-lived MCP process; production should inject a thin
        ``orchestrator_enqueue=`` that hands the task to the daemon (e.g.
        via a LocalTracker inbox). This default just writes the intent to
        the reports directory for later polling — the daemon side must
        listen on the inbox separately. No heavy imports, rollback-safe.
        """
        run_id = str(uuid.uuid4())
        reports = Path(self._config.reports_dir or ".reports")
        reports.mkdir(parents=True, exist_ok=True)
        ndjson = reports / f"{run_id}.ndjson"
        record = {
            "run_id": run_id,
            "issue_url": issue_url,
            "workflow_path": workflow_path,
            "event": "queued",
        }
        ndjson.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
        logger.info(
            "orchestrator enqueue (default): run_id=%s issue=%s -> %s",
            run_id,
            issue_url,
            ndjson,
        )
        return run_id

    def _default_sop_compiler(self, **kwargs: Any) -> dict[str, Any]:
        """Default SOP compiler — calls the real ``convert_sop_to_agent``."""
        from extensions.sop_converter.convert_sop_skill import convert_sop_to_agent

        return convert_sop_to_agent(
            sdk_spec=kwargs["sdk_spec"],
            requirements=kwargs.get("requirements", ""),
            agent_name=kwargs.get("agent_name", ""),
        )

    def _default_skill_invoker(self, skill_name: str, params: dict[str, Any]) -> str:
        """Default skill invoker — resolves the skill prompt via SkillRegistryExt.

        Returns the skill's prompt text (or an error if missing). The
        bridge stops at the prompt layer; the caller decides how to run it.
        """
        from extensions.skills_ext.registry_ext import SkillRegistryExt

        registry = SkillRegistryExt(project_root=self._config.workspace or ".")
        skills = registry.get_all_skills()
        for skill in skills:
            if getattr(skill, "name", None) == skill_name:
                prompt = getattr(skill, "prompt", "") or ""
                # Params are appended as metadata (prompt templates are not rewritten)
                if params:
                    prompt = f"{prompt}\n\n--- params ---\n{json.dumps(params, ensure_ascii=False)}"
                return prompt or f"(skill '{skill_name}' has empty prompt)"
        return f"error: skill '{skill_name}' not found in registry"

    def _default_stability_runner(self) -> str:
        """Default stability gate runner — subprocess pytest, parse summary."""
        args = list(self._config.stability_gate_args)
        cwd = self._config.stability_gate_cwd or None
        try:
            proc = subprocess.run(
                args,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=self._config.stability_gate_timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return f"error: stability gate timed out after {self._config.stability_gate_timeout_s}s"
        except FileNotFoundError as exc:
            return f"error: pytest not found: {exc}"
        out = (proc.stdout or "") + (proc.stderr or "")
        lines = out.strip().splitlines()
        summary = next(
            (ln for ln in reversed(lines) if re.search(r"\d+ passed|\d+ failed", ln)),
            None,
        )
        if summary is None:
            summary = lines[-1] if lines else "(no output)"
        return f"exit={proc.returncode} | {summary}"

    # ---- MCP server assembly (only when mcp is installed) ------------------

    def _build_mcp_server(self) -> Any:
        """Construct the ``mcp.server.Server`` and register handlers.

        Deferred until :meth:`run_stdio` so the module imports cleanly
        without ``mcp``; unit tests bypass this via :meth:`list_tools`.
        """
        if not mcp_available():
            raise ImportError(MCP_UNAVAILABLE)
        from mcp.server import Server
        from mcp.types import TextContent, Tool

        server: Server = Server("clawcodex-trae-bridge")

        @server.list_tools()
        async def _list_tools() -> list[Tool]:
            return [
                Tool(
                    name=spec.name,
                    description=spec.description,
                    inputSchema=spec.input_schema,
                )
                for spec in build_tool_specs()
            ]

        @server.call_tool()
        async def _call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
            text = await self.call_tool(name, arguments or {})
            return [TextContent(type="text", text=text)]

        return server

    async def run_stdio(self) -> None:
        """Expose the MCP server over stdio for the Trae IDE builtin-mcp.

        Entry point: ``python -m extensions.trae.mcp_bridge``. Requires
        ``mcp``; raises :class:`ImportError` with install hints otherwise.
        """
        if self._server is None:
            self._server = self._build_mcp_server()
        from mcp.server.stdio import stdio_server

        async with stdio_server() as (read_stream, write_stream):
            await self._server.run(
                read_stream,
                write_stream,
                self._server.create_initialization_options(),
            )


# Module entry: python -m extensions.trae.mcp_bridge


def _main() -> int:
    logging.basicConfig(
        level=os.environ.get("CLAWCODEX_BRIDGE_LOG", "INFO"),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    if not mcp_available():
        print(MCP_UNAVAILABLE, file=sys.stderr)
        return 2
    config = BridgeConfig.from_env()
    bridge = TraeMcpBridge(config=config)
    try:
        asyncio.run(bridge.run_stdio())
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

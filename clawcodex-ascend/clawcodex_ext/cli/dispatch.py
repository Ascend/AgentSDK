#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSES/Clawd-Codex-MIT.txt.
#
# Portions copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Licensed under Mulan PSL v2. You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

# pylint: disable=no-name-in-module,too-many-lines
"""Downstream CLI dispatch — owns run_cli(argv)."""

from __future__ import annotations

import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Callable


def _telemetry_record_session(*, session_id: str, entrypoint: str, is_non_interactive: bool) -> None:
    """Best-effort session_start.

    Local import keeps ``telemetry`` out of the CLI dispatch
    module surface for ``--help`` cold start.
    """
    try:
        from telemetry import record_session_start

        record_session_start(
            session_id=session_id,
            entrypoint=entrypoint,
            client_type=os.environ.get("CLAUDE_CODE_ENTRYPOINT", "cli"),
            is_non_interactive=is_non_interactive,
        )
    except Exception:  # nosec B110
        # Telemetry MUST NEVER block the CLI; failures are best-effort.
        pass


def _telemetry_record_end(
    *,
    session_id: str,
    command_name: str,
    mode: str,
    success: bool,
    duration_s: float,
    exit_status: int,
) -> None:
    try:
        from telemetry import record_command_run, record_session_end

        record_session_end(
            session_id=session_id,
            duration_s=duration_s,
            exit_status=exit_status,
        )
        record_command_run(
            session_id=session_id,
            command_name=command_name,
            mode=mode,
            success=success,
            duration_s=duration_s,
            exit_status=exit_status,
        )
    except Exception:  # nosec B110
        pass  # Telemetry is optional and must not affect command execution.


def _run_and_record(
    *,
    session_id: str,
    command_name: str,
    mode: str,
    start: float,
    fn: Callable[[], int],
) -> int:
    """Run a CLI action and close the telemetry session around its result."""
    rc = fn()
    _telemetry_record_end(
        session_id=session_id,
        command_name=command_name,
        mode=mode,
        success=(rc == 0),
        duration_s=time.monotonic() - start,
        exit_status=rc,
    )
    return rc


def _print_version() -> None:
    """Print the ``claw-codex version ...`` banner used by both --version channels."""
    from src import __version__

    print(f"claw-codex version {__version__} (Python)")


def _derive_session_id() -> str:
    """Resolve a session id, preferring the bootstrap one when set."""
    try:
        from src.bootstrap.state import get_session_id

        sid = get_session_id()
        if isinstance(sid, str) and sid:
            return sid
    except Exception:  # nosec B110
        pass  # Bootstrap state is optional here; use the existing generated session identifier.
    return uuid.uuid4().hex


def _apply_feature_gate_overrides(args: object) -> None:
    """Apply ``--enable-feature`` / ``--disable-feature`` CLI args."""
    try:
        from clawcodex_ext.feature_gate import get_registry

        reg = get_registry()
        enable_flags = getattr(args, "enable_feature", None) or []
        disable_flags = getattr(args, "disable_feature", None) or []
        for name in enable_flags:
            reg.enable_feature(name)
        for name in disable_flags:
            reg.disable_feature(name)
    except Exception:  # nosec B110
        # Feature-gate failures MUST NEVER block the CLI.
        pass


def _apply_agent_debug_if_requested(argv: list[str]) -> None:
    if "--agent-debug" not in argv[1:]:
        return
    try:
        from clawcodex_ext.debug.agent_debug import apply_agent_debug_environment

        apply_agent_debug_environment(os.environ)
    except Exception:
        os.environ["CLAWCODEX_AGENT_DEBUG"] = "1"


def _is_provider_free_goal_summary_print(args: object) -> bool:
    """Return True for local-only ``-p /goal`` status and clear commands."""
    if not getattr(args, "print", False):
        return False
    if getattr(args, "input_format", "text") != "text":
        return False
    prompt = getattr(args, "prompt", None)
    if not isinstance(prompt, str):
        return False
    words = prompt.strip().lower().split()
    if not words or words[0] != "/goal":
        return False
    return len(words) == 1 or (len(words) == 2 and words[1] in {"clear", "stop", "off", "reset", "none", "cancel"})


def _has_multimodel_selection(args: Any) -> bool:
    """Return True when a multimodel group would activate for this run."""
    if getattr(args, "multimodel", None) or getattr(args, "runtime_multimodel", None):
        return True
    from clawcodex_ext.multimodel.config import MultiModelConfigError, load_config

    try:
        return bool(load_config().default_group)
    except MultiModelConfigError:
        return True


def _run_with_worktree_keep_note(callback, worktree_session):
    """Run a local frontend and always report its retained worktree."""
    try:
        return callback()
    finally:
        if worktree_session is not None:
            from clawcodex_ext.cli.worktree import print_worktree_keep_note

            print_worktree_keep_note(worktree_session)


def _maybe_argcomplete_top_level(argv: list[str]) -> None:
    """If argcomplete is active, expose the fast-path subcommand nouns.

    The flat top-level parser at ``build_parser()`` does not know about
    the subcommand sieve in ``run_cli`` (login/config/mcp/.../provider/
    model/sop/viz). When ``_ARGCOMPLETE`` is set, argcomplete's
    ``autocomplete()`` will only complete the flat parser's tokens. This
    hook attaches the sieve's noun set as the first-positional choice
    list so the shell can offer the full subcommand set. No-op when
    ``_ARGCOMPLETE`` is unset; the lazy import keeps ``--help`` under
    the 5-second budget enforced by the stability gate.
    """

    if os.environ.get("_ARGCOMPLETE") != "1":
        return
    import argcomplete  # noqa: F401

    from clawcodex_ext.cli.parser import build_parser
    from clawcodex_ext.cli.subcommand_registry import (
        _SUBCOMMANDS,
        load_builtin_subcommands,
    )

    parser = build_parser()
    load_builtin_subcommands()
    top_level = tuple(_SUBCOMMANDS.keys())
    # Override the first-positional ``prompt`` argument's choice list
    # so argcomplete offers the subcommand nouns. argcomplete reads the
    # parser's own argument table for flag completion automatically.
    for action in parser._actions:  # type: ignore[attr-defined]
        if action.dest == "prompt":
            action.choices = top_level  # type: ignore[attr-defined]
            break
    argcomplete.autocomplete(parser, always_complete_options=False)


def _dispatch_frontend(
    *,
    args: Any,
    ctx: Any,
    argv: list[str],
    worktree_session: Any,
    telemetry_session_id: str,
    telemetry_start: float,
) -> int:
    """Run the mode frontend — headless (``-p/--print``), TUI, or REPL."""
    if args.print:
        # telemetry notice — shown once on stderr for headless/CLI mode
        # so users know when collection + reporting are active.
        try:
            from telemetry.config import load_config as _load_telemetry_cfg

            _tc = _load_telemetry_cfg()
            if _tc.enabled and _tc.reporting.reporting_enabled:
                print(
                    "Telemetry: stats ✓ · error reporting ✓  — /telemetry to configure",
                    file=sys.stderr,
                )
                print(
                    "Collects usage data & error reports; may be uploaded periodically.",
                    file=sys.stderr,
                )
        except Exception:  # nosec B110
            pass  # Telemetry is optional and must not affect command execution.
        name, command_name, mode = "headless", "print", "non_interactive"
    else:
        from src.entrypoints.tui import should_use_tui

        # Interactive path: decide between the Textual TUI (new default)
        # and the legacy Rich REPL. Explicit flags win; otherwise
        # auto-detect a compatible TTY.
        explicit_tui: bool | None = None
        if args.tui:
            explicit_tui = True
        elif getattr(args, "legacy_repl", False) or args.no_tui:
            explicit_tui = False

        if should_use_tui(explicit_tui):
            name, command_name, mode = "tui", "tui", "interactive"
        else:
            name, command_name, mode = "repl", "repl", "interactive"

    from src.utils.startup_profiler import profile_checkpoint

    profile_checkpoint(f"mode_dispatch_{name}")
    profile_checkpoint("phase4_dispatch")

    from clawcodex_ext.frontend import get_frontend

    frontend = get_frontend(name)
    return _run_and_record(
        session_id=telemetry_session_id,
        command_name=command_name,
        mode=mode,
        start=telemetry_start,
        fn=lambda: _run_with_worktree_keep_note(lambda: frontend.run(ctx, argv[1:]), worktree_session),
    )


def _maybe_print_startup_diagnostic(args: Any) -> None:
    """Print a Provider/Model diagnostic line for ``-p/--print`` runs."""
    if not getattr(args, "print", False):
        return
    try:
        from src.config import get_default_provider, get_provider_config

        provider_name = getattr(args, "provider", None) or get_default_provider()
        provider_cfg = get_provider_config(provider_name) or {}
        model = getattr(args, "model", None) or provider_cfg.get("default_model")
        print(
            f"Provider: {provider_name}, Model: {model}",
            file=sys.stderr,
            flush=True,
        )
    except Exception:  # nosec B110
        # Config lookup failure must not block init; subsequent paths
        # will surface authoritative values when they succeed.
        pass


def _apply_swarm_mode(args: Any, parser: Any) -> None:
    """Rewrite args for a ``--swarm`` / ``--effort swarm`` invocation."""
    if not (bool(getattr(args, "swarm", False)) or getattr(args, "effort", None) == "swarm"):
        return
    if not getattr(args, "prompt", None):
        parser.error("--swarm/--decompose requires a prompt")
    from extensions.orchestrator.issue import Issue
    from extensions.orchestrator.task_decomposition import (
        TaskDecomposer,
        build_swarm_prompt,
        write_task_plan,
    )

    workspace_root = Path.cwd()
    issue = Issue(
        id="cli-swarm",
        identifier="cli-swarm",
        title=str(args.prompt)[:160],
        description=str(args.prompt),
    )
    import asyncio

    plan = asyncio.run(TaskDecomposer().decompose_issue(issue))
    plan_path = write_task_plan(plan, workspace_root)
    args.prompt = build_swarm_prompt(issue, plan, plan_path)
    args.print = True
    os.environ["CLAUDE_CODE_COORDINATOR_MODE"] = "1"
    print(
        "NOTE: CLI --swarm/--decompose mode runs outside the orchestrator's "
        "normal issue tracking pipeline. There will be no IssueRecord "
        "persistence, no summary comment, and no PR creation. "
        "The decomposed plan has been written to "
        f"{plan_path.relative_to(workspace_root)}.",
        file=sys.stderr,
    )


def _resolve_session_args(args: Any) -> int | None:
    """Resolve --continue / --resume references to a concrete session ID."""
    # Resolve --continue: auto-detect the most recent session (S-R3).
    if getattr(args, "continue", None) and not getattr(args, "resume", None):
        from src.services.session_storage import SessionStorage

        try:
            metas = SessionStorage.list_sessions(limit=1)
            if metas:
                args.resume = metas[0].session_id
            else:
                print("No previous sessions found to continue.", file=sys.stderr)
        except Exception:
            print("Unable to list sessions for --continue.", file=sys.stderr)

    # Resolve --resume: if the value is not a known session ID, try it
    # as a tag prefix so `--resume cron:task:build` works directly.
    resume_val = getattr(args, "resume", None)
    if resume_val and resume_val != "browse":
        from clawcodex_ext.services.session_storage import (
            SessionStorage,
            resolve_sessions_dir,
        )

        session_dir = resolve_sessions_dir() / resume_val
        if not session_dir.is_dir():
            # Not a session directory — try tag prefix lookup.
            metas = SessionStorage.list_sessions(tag_filter=str(resume_val), limit=1)
            if metas:
                print(
                    f"Resuming session {metas[0].session_id[:8]}... (matched by tag '{resume_val}')",
                    file=sys.stderr,
                )
                args.resume = metas[0].session_id
            else:
                print(
                    f"No session found for ID or tag '{resume_val}'.",
                    file=sys.stderr,
                )
                return 1
    return None


def _bootstrap_cli_environment(argv: list[str] | None) -> list[str]:
    """Normalize argv and run the pre-telemetry environment setup."""
    # WI-0.1 (ch17 Phase 0): instrument cold-start phases. Env-gated by
    # ``CLAUDE_CODE_PROFILE_STARTUP``; a no-op import + no-op call when
    # disabled (~ns overhead). On exit the profiler writes a Markdown
    # report to ``$CLAUDE_CONFIG_DIR/startup-perf/{session_id}.txt``.
    from src.utils.startup_profiler import profile_checkpoint

    profile_checkpoint("cli_main_entry")

    # A nested CLI inherits the outer session's worktree variables. Strip
    # them before any prefetch/bootstrap child can snapshot the environment;
    # only a worktree created by this invocation may be advertised later.
    from src.utils.worktree_session import strip_worktree_env

    strip_worktree_env()

    if os.environ.get("CLAWCODEX_DEBUG", "").lower() in ("1", "true", "yes"):
        import logging

        logging.basicConfig(
            level=logging.WARNING,
            format="%(asctime)s %(name)s %(message)s",
            stream=sys.stderr,
        )

    if argv is None:
        argv = sys.argv

    _apply_agent_debug_if_requested(argv)
    return argv


def _run_pre_parse_channels(
    argv: list[str],
    telemetry_session_id: str,
    telemetry_start: float,
) -> int | None:
    """Match ``clawcodex <token>`` subcommands against the registry."""
    _maybe_argcomplete_top_level(argv)

    # Match the first token only; flag values that equal a subcommand
    # name (``clawcodex --model mcp``) must not mis-route the sieve.
    rest = argv[1:]
    if rest and not rest[0].startswith("-"):
        token = rest[0]
        rest_args = rest[1:]

        from clawcodex_ext.cli.subcommand_registry import (
            get_subcommand,
            telemetry_mode_for,
        )

        subcommand = get_subcommand(token)
        if subcommand is not None:
            return _run_and_record(
                session_id=telemetry_session_id,
                command_name=token,
                mode=telemetry_mode_for(token),
                start=telemetry_start,
                fn=lambda: subcommand(rest_args),
            )
    return None


def _run_post_parse_channels(
    args: Any,
    parser: Any,
    telemetry_session_id: str,
    telemetry_start: float,
) -> int | None:
    """Run the post-parse arg rewrites and flag fast exits."""
    from src.utils.startup_profiler import profile_checkpoint

    profile_checkpoint("argparse_done")

    _apply_swarm_mode(args, parser)

    if getattr(args, "prompt", None) and not getattr(args, "print", False):
        parser.error(f"unknown command: {args.prompt} (use -p/--print to send a prompt)")

    exit_code = _resolve_session_args(args)
    if exit_code is not None:
        return exit_code

    if args.version:
        _print_version()
        return _run_and_record(
            session_id=telemetry_session_id,
            command_name="version",
            mode="non_interactive",
            start=telemetry_start,
            fn=lambda: 0,
        )

    if args.config:
        from clawcodex_ext.cli.subcommand_registry import (
            get_subcommand,
            telemetry_mode_for,
        )

        subcommand = get_subcommand("config")
        if subcommand is not None:
            return _run_and_record(
                session_id=telemetry_session_id,
                command_name="config",
                mode=telemetry_mode_for("config"),
                start=telemetry_start,
                fn=lambda: subcommand([]),
            )
    return None


def _run_init_and_resolve_permissions(args: Any) -> None:
    """Run the init hook, then resolve permission state once."""
    from src.utils.startup_profiler import profile_checkpoint

    # Plan-phase-1 wiring (ch02-bootstrap-refactoring-plan.md P1.5):
    # ``run_pre_action(args)`` is the Python analog of Commander's
    # ``preAction`` hook: memoized ``init()`` (safe env vars +
    # graceful-shutdown + API preconnect) + interactive bootstrap state.
    #
    profile_checkpoint("phase0_end_phase2_start")
    from src.init import run_pre_action

    run_pre_action(args)
    profile_checkpoint("phase2_end_phase3_start")

    from clawcodex_ext.cli.permissions import resolve_permission_state

    resolve_permission_state(args)
    profile_checkpoint("permissions_resolved")
    profile_checkpoint("phase3_end_phase4_start")


def _enter_worktree(args: Any) -> Any | None:
    """Create or adopt the ``--worktree`` session and enter it."""
    from clawcodex_ext.cli.worktree import WORKTREE_FAILED, maybe_create_worktree

    worktree_session = maybe_create_worktree(args)
    if worktree_session is WORKTREE_FAILED:
        return WORKTREE_FAILED
    if worktree_session is not None:
        os.chdir(worktree_session.worktree_path)
        os.environ.update(worktree_session.to_env())
    return worktree_session


def _build_runtime_options(args: Any, *, worktree_session: Any) -> Any:
    """Project resolved CLI args into the shared RuntimeOptions."""
    from clawcodex_ext.cli.runners import split_csv
    from clawcodex_ext.runtime.context import RuntimeOptions

    # ``--resume`` without a SESSION_ID means "browse" mode.
    resume_val = getattr(args, "resume", None)

    # An explicit --agent value that names an existing directory is a
    # slash-command bundle path; ``auto`` and flagless runs stay None.
    bundle_path: Path | None = None
    agent_type_raw = getattr(args, "agent", None)
    if agent_type_raw is not None and agent_type_raw != "auto":
        candidate = Path(str(agent_type_raw)).resolve()
        if candidate.is_dir():
            bundle_path = candidate

    return RuntimeOptions(
        provider_name=getattr(args, "provider", None),
        model=getattr(args, "model", None),
        fallback_model=getattr(args, "fallback_model", None),
        effort=(args.effort if getattr(args, "effort", None) not in (None, "normal", "swarm") else None),
        prompt=getattr(args, "prompt", None),
        output_format=getattr(args, "output_format", "text"),
        input_format=getattr(args, "input_format", "text"),
        include_partial_messages=getattr(args, "include_partial_messages", False),
        max_turns=getattr(args, "max_turns", 20),
        max_turns_explicit=bool(getattr(args, "max_turns_explicit", False)),
        allowed_tools=tuple(split_csv(getattr(args, "allowed_tools", None))),
        disallowed_tools=tuple(split_csv(getattr(args, "disallowed_tools", None))),
        stream=getattr(args, "stream", False),
        permission_mode=args._resolved_permission_mode,
        is_bypass_permissions_mode_available=args._resolved_is_bypass_available,
        skip_permissions=getattr(args, "dangerously_skip_permissions", False),
        resume_session_id=resume_val if resume_val and resume_val != "browse" else None,
        resume_browse=(resume_val == "browse"),
        fork_session_id=getattr(args, "fork_session", None),
        resume_session_at=_parse_resume_at(getattr(args, "resume_session_at", None)),
        verbose=getattr(args, "verbose", False),
        gateway=getattr(args, "gateway", False),
        gateway_origin=getattr(args, "gateway_origin", None),
        gateway_sock=getattr(args, "gateway_sock", None),
        bundle_path=bundle_path,
        record=getattr(args, "record", None),
        record_width=getattr(args, "record_width", None),
        record_height=getattr(args, "record_height", None),
        workspace_root=(Path(worktree_session.worktree_path) if worktree_session is not None else None),
        worktree_session=worktree_session,
        multimodel_cli_group=getattr(args, "multimodel", None),
        multimodel_runtime_group=getattr(args, "runtime_multimodel", None),
    )


def _run_goal_summary_headless(
    args: Any,
    runtime_opts: Any,
    worktree_session: Any,
    telemetry_session_id: str,
    telemetry_start: float,
) -> int | None:
    """Run provider-free ``-p /goal`` status/clear commands headless."""
    if not _is_provider_free_goal_summary_print(args):
        return None
    if _has_multimodel_selection(args):
        return None
    from src.entrypoints.headless import HeadlessOptions, run_headless

    headless_options = HeadlessOptions.from_runtime_opts(runtime_opts)
    return _run_and_record(
        session_id=telemetry_session_id,
        command_name="print",
        mode="non_interactive",
        start=telemetry_start,
        fn=lambda: _run_with_worktree_keep_note(lambda: run_headless(headless_options), worktree_session),
    )


def _build_runtime_context(runtime_opts: Any, worktree_session: Any) -> Any | None:
    """Build the RuntimeContext shared by all frontends."""
    from clawcodex_ext.runtime.context import RuntimeContext

    try:
        return RuntimeContext.build(runtime_opts)
    except RuntimeError as exc:
        message = str(exc).strip() or "Provider configuration is missing."
        print(f"warning: {message}", file=sys.stderr)
        if sys.stdin.isatty() and sys.stdout.isatty():
            print(
                "hint: run `clawcodex login` to configure credentials interactively.",
                file=sys.stderr,
            )
        if worktree_session is not None:
            from clawcodex_ext.cli.worktree import print_worktree_keep_note

            print_worktree_keep_note(worktree_session)
        return None


def run_cli(argv: list[str] | None = None) -> int:
    """CLI main entry point, parameterized to avoid sys.argv mutation in tests."""
    argv = _bootstrap_cli_environment(argv)

    _telemetry_session_id = _derive_session_id()
    _telemetry_start = time.monotonic()
    _telemetry_record_session(
        session_id=_telemetry_session_id,
        entrypoint="cli",
        is_non_interactive=False,
    )

    rc = _run_pre_parse_channels(argv, _telemetry_session_id, _telemetry_start)
    if rc is not None:
        return rc

    from clawcodex_ext.cli.parser import build_parser

    parser = build_parser()
    args = parser.parse_args(argv[1:])

    _maybe_print_startup_diagnostic(args)

    rc = _run_post_parse_channels(args, parser, _telemetry_session_id, _telemetry_start)
    if rc is not None:
        return rc

    _apply_feature_gate_overrides(args)

    _run_init_and_resolve_permissions(args)

    if getattr(args, "fallback_model", None) and args.fallback_model == getattr(args, "model", None):
        print("error: --fallback-model must differ from --model", file=sys.stderr)
        return 2

    from clawcodex_ext.cli.worktree import WORKTREE_FAILED

    worktree_session = _enter_worktree(args)
    if worktree_session is WORKTREE_FAILED:
        return 1

    runtime_opts = _build_runtime_options(args, worktree_session=worktree_session)

    rc = _run_goal_summary_headless(args, runtime_opts, worktree_session, _telemetry_session_id, _telemetry_start)
    if rc is not None:
        return rc

    ctx = _build_runtime_context(runtime_opts, worktree_session)
    if ctx is None:
        return 1

    _resolve_startup_agent(args, ctx)

    return _dispatch_frontend(
        args=args,
        ctx=ctx,
        argv=argv,
        worktree_session=worktree_session,
        telemetry_session_id=_telemetry_session_id,
        telemetry_start=_telemetry_start,
    )


# ---------------------------------------------------------------------------
# Agent resolution: --agent flag or auto-detect clawcodex-overview.md
# ---------------------------------------------------------------------------


def _apply_sop_startup(
    ctx,
    agent: dict,
    *,
    bundle_path: Path | None,
    workspace: Path,
    force_bundle: bool = False,
) -> None:
    """Inject SOP routing, optional bundle isolation, and proxy tool allowlist."""
    from extensions.sop_converter.bundle_context import (
        activate_bundle_isolation,
        apply_sdk_source_working_directory,
        build_bundle_context,
    )
    from extensions.sop_converter.bundle_agents import register_bundle_agents
    from extensions.sop_converter.bundle_discovery import overview_has_sop_skills
    from extensions.sop_converter.bundle_skills import register_bundle_skills
    from extensions.sop_converter.sop_prompts import (
        append_sop_overview_routing,
        format_sdk_source_dir_block,
    )
    from extensions.sop_converter.startup_agent import build_bundle_overview_agent_definition

    is_sop = overview_has_sop_skills(agent) or force_bundle
    bundle_ctx = None

    if bundle_path is not None and bundle_path.is_dir() and is_sop:
        bundle_path = bundle_path.resolve()
        ctx.options.agent_dir_override = bundle_path
        ctx.tool_context._agent_dir_override = bundle_path

        agent_names = register_bundle_agents(bundle_path)
        if agent_names:
            sample_agents = ", ".join(agent_names[:4])
            if len(agent_names) > 4:
                sample_agents += ", …"
            domain_agents = [a for a in agent_names if a.endswith("-agent") and not a.startswith("clawcodex-")]
            stage_agents = [
                a
                for a in agent_names
                if a.endswith("-agent")
                and any(
                    stage in a
                    for stage in [
                        "topic-init",
                        "problem-decompose",
                        "search-strategy",
                        "literature-collect",
                        "literature-screen",
                        "knowledge-extract",
                        "synthesis",
                        "hypothesis-gen",
                        "experiment-design",
                        "code-generation",
                        "resource-planning",
                        "experiment-run",
                        "iterative-refine",
                        "result-analysis",
                        "research-decision",
                        "paper-outline",
                        "paper-draft",
                        "peer-review",
                        "paper-revision",
                        "quality-gate",
                        "knowledge-archive",
                        "export-publish",
                        "citation-verify",
                    ]
                )
            ]

            if stage_agents:
                agent_type_desc = f"stage agents ({len(stage_agents)}) + domain agents ({len(domain_agents)})"
            else:
                agent_type_desc = "domain agents"

            print(
                f"🤖 Loaded {len(agent_names)} SOP {agent_type_desc} from bundle",
                file=sys.stderr,
            )
            print(f"   agents: {sample_agents}", file=sys.stderr)

        load_result = register_bundle_skills(bundle_path, workspace)
        registered = load_result.skill_names
        if registered:
            from extensions.sop_converter.workflow_project import (
                read_workflow_first_stage_skill_name,
            )

            stage1_skill = read_workflow_first_stage_skill_name(bundle_path)
            marker_text = ""
            if stage1_skill and stage1_skill in registered:
                marker_text = f"; stage-1 {stage1_skill} ✓"
            sample = ", ".join(registered[:4])
            if len(registered) > 4:
                sample += ", …"
            print(
                f"📦 Loaded {len(registered)} SOP skills from bundle{marker_text}",
                file=sys.stderr,
            )
            print(f"   skills: {sample}", file=sys.stderr)
            try:
                from src.skills.loader import get_all_skills

                get_all_skills(project_root=workspace)
            except Exception:  # nosec B110
                pass  # This optional feature is unavailable; continue with the core command path.
        bundle_ctx = build_bundle_context(
            bundle_path=bundle_path,
            skill_names=load_result.skill_names,
            skill_dirs=load_result.skill_dirs,
            tool_names=load_result.tool_names,
            workspace_root=workspace,
        )
        activate_bundle_isolation(ctx.tool_registry, bundle_ctx)
        ctx.options.bundle_path = bundle_path
        ctx.tool_context.bundle_context = bundle_ctx
        apply_sdk_source_working_directory(ctx.tool_context, bundle_ctx)

    if is_sop:
        startup_def = build_bundle_overview_agent_definition(
            agent,
            bundle_dir=bundle_path if bundle_path is not None else workspace,
        )
        ctx.options.startup_agent = startup_def
        ctx.tool_context.startup_agent = startup_def
        ctx.tool_context.agent_type = startup_def.agent_type
        if bundle_ctx is not None:
            ctx.tool_context.bundle_context = bundle_ctx
        try:
            from extensions.sop_converter.runtime.catalog_tools import (
                register_resource_catalog_tool,
            )

            register_resource_catalog_tool(getattr(ctx, "tool_registry", None))
        except Exception:  # nosec B110
            pass  # This optional feature is unavailable; continue with the core command path.

    body = (agent.get("system_prompt_body") or "").strip()
    sdk_source_dir = bundle_ctx.sdk_source_dir if bundle_ctx is not None else None
    if body:
        if is_sop:
            body = append_sop_overview_routing(
                body,
                sdk_source_dir=sdk_source_dir,
                bundle_path=bundle_path,
            )
        existing = getattr(ctx.options, "append_system_prompt", "")
        ctx.options.append_system_prompt = f"{existing}\n\n{body}" if existing else body
    elif is_sop and sdk_source_dir is not None:
        sdk_block = format_sdk_source_dir_block(sdk_source_dir)
        if sdk_block:
            existing = getattr(ctx.options, "append_system_prompt", "")
            ctx.options.append_system_prompt = f"{existing}\n\n{sdk_block}" if existing else sdk_block

    agent_name = agent.get("name", "unknown")
    skills = agent.get("skills", [])
    sub_count = len([s for s in skills if isinstance(s, str) and s.endswith("-skill")])
    if sub_count:
        print(f"⚡ Using agent: {agent_name} ({sub_count} sub-agents)", file=sys.stderr)
    else:
        print(f"⚡ Using agent: {agent_name}", file=sys.stderr)


def _resolve_startup_agent(args, ctx) -> None:
    """Resolve agent type from ``--agent`` flag or auto-detect.

    Priority:
      1. ``--agent <name>``  → ``resolve_agent_by_type(cwd, name)``
      2. ``--agent /path``   → load from directory's ``.claude/agents/``
      3. ``--agent`` (const) → ``resolve_default_agent(cwd)``
      4. No ``--agent``      → ``resolve_default_agent(cwd)`` (auto-detect)
      5. Nothing found       → keep the default GENERAL_PURPOSE_AGENT

    Injects the agent's ``system_prompt_body`` into
    ``ctx.options.append_system_prompt``.  Prints a startup banner
    showing the resolved agent name and sub-agent count.
    """

    from extensions.sop_converter.default_agent import (
        resolve_agent_by_type,
        resolve_default_agent,
    )

    from extensions.sop_converter.bundle_discovery import (
        discover_workspace_bundle,
        overview_has_sop_skills,
    )

    cwd = ctx.workspace_root or Path.cwd()
    agent_type = getattr(args, "agent", None)

    # Check if agent_type is a directory path
    agent_dir_override: Path | None = None
    if agent_type is not None and agent_type != "auto":
        agent_type_path = Path(str(agent_type)).resolve()
        if agent_type_path.is_dir():
            agent_dir_override = agent_type_path
            ctx.options.agent_dir_override = agent_dir_override
            ctx.tool_context._agent_dir_override = agent_dir_override
            workspace = ctx.workspace_root or Path.cwd()
            bundle_path = agent_type_path
            agent = resolve_default_agent(bundle_path)
            if agent is None:
                agent = _resolve_first_agent_in_dir(bundle_path)
            if agent is None:
                agent = resolve_default_agent(workspace)
            if agent is None:
                agent = _resolve_first_agent_in_dir(workspace)
            if agent is None:
                from extensions.sop_converter.bundle_skills import register_bundle_skills

                load_result = register_bundle_skills(bundle_path, workspace)
                agent = {
                    "name": "clawcodex-overview",
                    "description": "SOP bundle session",
                    "skills": load_result.skill_names,
                    "system_prompt_body": "",
                }
            _apply_sop_startup(
                ctx,
                agent,
                bundle_path=bundle_path,
                workspace=workspace,
                force_bundle=True,
            )
            return

    if agent_type is not None and agent_type != "auto":
        # Explicit ``--agent <name>``
        agent = resolve_agent_by_type(cwd, agent_type, agent_dir_override=agent_dir_override)
    else:
        # Auto-detect (always check, even with ``--agent`` bare)
        agent = resolve_default_agent(cwd)

    if agent:
        workspace = ctx.workspace_root or Path.cwd()
        bundle_path: Path | None = None
        if overview_has_sop_skills(agent):
            bundle_path = discover_workspace_bundle(
                workspace,
                agent_skills=agent.get("skills"),
            )
            if bundle_path is not None:
                print(
                    f"📦 Auto-activated SOP bundle: {bundle_path.name}",
                    file=sys.stderr,
                )
        _apply_sop_startup(
            ctx,
            agent,
            bundle_path=bundle_path,
            workspace=workspace,
        )


def _resolve_first_agent_in_dir(cwd: Path) -> dict[str, Any] | None:
    """Scan ``.claude/agents/`` in *cwd* for the best overview agent.

    Serves as fallback when ``resolve_default_agent()`` (which looks for the
    hardcoded ``clawcodex-overview.md`` name) finds nothing, but the directory
    contains agent files with different names (e.g. ``ascend-data-forge.md``).

    Uses a scoring heuristic: the overview agent is the file with the most
    ``skill-`` prefixed items in its ``skills`` list + the longest body.
    This reliably distinguishes the overview (dozens of skills, >>1k body)
    from sub-agents (1 skill, <200 body).
    """
    from extensions.sop_converter.default_agent import parse_agent_file

    agents_dir = cwd / ".claude" / "agents"
    if not agents_dir.is_dir():
        return None

    best: dict[str, Any] | None = None
    best_score = -1

    for md_file in sorted(agents_dir.glob("*.md")):
        try:
            agent = parse_agent_file(md_file)
            if not agent:
                continue
            skills = agent.get("skills", [])
            sub_skills = len([s for s in skills if isinstance(s, str) and s.endswith("-skill")])
            body_len = len(agent.get("system_prompt_body", "") or "")
            score = sub_skills * 1000 + body_len
            if score > best_score:
                best_score = score
                best = agent
        except Exception:
            continue

    return best


def _parse_resume_at(raw: str | None) -> int | None:
    """Parse ``--resume-session-at`` value to an integer index.

    Returns ``None`` when the argument is not given or cannot be parsed.
    """
    if raw is None:
        return None
    try:
        val = int(raw)
        if val < 0:
            return None
        return val
    except (ValueError, TypeError):
        return None

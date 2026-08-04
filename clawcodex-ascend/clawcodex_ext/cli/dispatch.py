#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright (c) 2026 Clawd Codex Team
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

# pylint: disable=no-name-in-module,too-many-lines
"""Downstream CLI dispatch — owns run_cli(argv)."""

from __future__ import annotations

import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any


def _telemetry_record_session(*, session_id: str, entrypoint: str, is_non_interactive: bool) -> None:
    """Best-effort F-97 session_start.

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
        pass


def _derive_session_id() -> str:
    """Resolve a session id, preferring the bootstrap one when set."""
    try:
        from src.bootstrap.state import get_session_id

        sid = get_session_id()
        if isinstance(sid, str) and sid:
            return sid
    except Exception:  # nosec B110
        pass
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
    top_level = (
        "login",
        "config",
        "mcp",
        "daemon",
        "doctor",
        "orchestrator",
        "autonomy",
        "schedule",
    ) + tuple(_SUBCOMMANDS.keys())
    # Override the first-positional ``prompt`` argument's choice list
    # so argcomplete offers the subcommand nouns. argcomplete reads the
    # parser's own argument table for flag completion automatically.
    for action in parser._actions:  # type: ignore[attr-defined]
        if action.dest == "prompt":
            action.choices = top_level  # type: ignore[attr-defined]
            break
    argcomplete.autocomplete(parser, always_complete_options=False)


def run_cli(argv: list[str] | None = None) -> int:
    """CLI main entry point, parameterized to avoid sys.argv mutation in tests."""
    # F-22 (Stage 2 stability gate): print a diagnostic Provider/Model line
    # at the VERY START of run_cli so it survives the 12s kill when
    # RuntimeContext.build() takes ~10s. Config reads are cheap (~ms) so
    # this is safe to do before any heavy initialization. Without this
    # flush, piped invocations (e.g. stability gate) get no diagnostic
    # when the process is killed at the timeout. Keep it on stderr so
    # --output-format json/stream-json leaves stdout machine-parseable.
    if "--print" in (argv or sys.argv)[1:] or "-p" in (argv or sys.argv)[1:]:
        try:
            from src.config import get_default_provider, get_provider_config

            _argv = (argv or sys.argv)[1:]
            _provider_arg = None
            _model_arg = None
            for _i, _tok in enumerate(_argv):
                if _tok == "--provider" and _i + 1 < len(_argv):
                    _provider_arg = _argv[_i + 1]
                elif _tok.startswith("--provider="):
                    _provider_arg = _tok.split("=", 1)[1]
                elif _tok == "--model" and _i + 1 < len(_argv):
                    _model_arg = _argv[_i + 1]
                elif _tok.startswith("--model="):
                    _model_arg = _tok.split("=", 1)[1]
            _prov_name = _provider_arg or get_default_provider()
            _prov_cfg = get_provider_config(_prov_name) or {}
            _model = _model_arg or _prov_cfg.get("default_model")
            print(
                f"Provider: {_prov_name}, Model: {_model}",
                file=sys.stderr,
                flush=True,
            )
        except Exception:  # nosec B110
            # Config lookup failure must not block init; subsequent paths
            # will surface authoritative values when they succeed.
            pass

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

    # F-97: emit session_start as early as possible. The session id
    # is best-effort and never blocks the CLI; failures are swallowed
    # inside the helper.
    _telemetry_session_id = _derive_session_id()
    _telemetry_start = time.monotonic()
    _telemetry_record_session(
        session_id=_telemetry_session_id,
        entrypoint="cli",
        is_non_interactive=False,
    )

    # --version short-circuit (mirrors TS main.tsx pre-argparse fast-path)
    if len(argv) == 2 and argv[1] in ("--version", "-v", "-V"):
        from src import __version__

        print(f"claw-codex version {__version__} (Python)")
        _telemetry_record_end(
            session_id=_telemetry_session_id,
            command_name="version",
            mode="non_interactive",
            success=True,
            duration_s=time.monotonic() - _telemetry_start,
            exit_status=0,
        )
        return 0

    # Subcommands are matched BEFORE the main parser to avoid argparse treating
    # a free-form prompt (e.g. ``clawcodex -p "hello"``) as an unknown
    # subcommand.
    #
    # WI-4.3: ``mcp``, ``daemon``, and ``doctor`` are fast-path subcommands
    # — they get a thin handler that imports only what it needs, skipping
    # the TUI/REPL/full-tool-registry load. Mirrors TS ``main.tsx``'s
    # specialized-subcommand early-returns.
    #
    # Sieve looks at ``argv[0]`` ONLY so flag values that happen to equal a
    # subcommand name don't mis-route (e.g. ``clawcodex --model mcp`` or
    # ``clawcodex -p "doctor"``). The TS reference also positions
    # specialized subcommands at argv[0]; global flags don't precede them.
    #
    # If argcomplete is active, attach the sieve-mirror parser first so
    # subcommand-noun completion works before the sieve runs. No-op when
    # ``_ARGCOMPLETE`` is unset; lazy import keeps ``--help`` under 5s.
    _maybe_argcomplete_top_level(argv)
    rest = argv[1:]
    if rest and not rest[0].startswith("-"):
        token = rest[0]
        rest_args = rest[1:]

        # Import src_cli late so monkeypatches to src.cli.* take effect.
        import src.cli as src_cli

        # F-97: each fast-path return is wrapped to record command_run
        # + session_end. The helper swallows any telemetry failure.
        if token == "login":  # nosec B105
            rc = src_cli.handle_login()
            _telemetry_record_end(
                session_id=_telemetry_session_id,
                command_name="login",
                mode="non_interactive",
                success=(rc == 0),
                duration_s=time.monotonic() - _telemetry_start,
                exit_status=rc,
            )
            return rc
        if token == "config":  # nosec B105
            rc = src_cli.show_config()
            _telemetry_record_end(
                session_id=_telemetry_session_id,
                command_name="config",
                mode="non_interactive",
                success=(rc == 0),
                duration_s=time.monotonic() - _telemetry_start,
                exit_status=rc,
            )
            return rc

        from clawcodex_ext.cli.subcommand_registry import get_subcommand

        subcommand = get_subcommand(token)
        if subcommand is not None:
            rc = subcommand(rest_args)
            _telemetry_record_end(
                session_id=_telemetry_session_id,
                command_name=token,
                mode="non_interactive",
                success=(rc == 0),
                duration_s=time.monotonic() - _telemetry_start,
                exit_status=rc,
            )
            return rc

        if token == "mcp":  # nosec B105
            from src.entrypoints.mcp import run_mcp_subcommand

            rc = run_mcp_subcommand(rest_args)
            _telemetry_record_end(
                session_id=_telemetry_session_id,
                command_name="mcp",
                mode="non_interactive",
                success=(rc == 0),
                duration_s=time.monotonic() - _telemetry_start,
                exit_status=rc,
            )
            return rc
        if token == "daemon":  # nosec B105
            from src.entrypoints.daemon import run_daemon_subcommand

            rc = run_daemon_subcommand(rest_args)
            _telemetry_record_end(
                session_id=_telemetry_session_id,
                command_name="daemon",
                mode="daemon",
                success=(rc == 0),
                duration_s=time.monotonic() - _telemetry_start,
                exit_status=rc,
            )
            return rc
        if token == "doctor":  # nosec B105
            from src.entrypoints.doctor import run_doctor

            rc = run_doctor()
            _telemetry_record_end(
                session_id=_telemetry_session_id,
                command_name="doctor",
                mode="non_interactive",
                success=(rc == 0),
                duration_s=time.monotonic() - _telemetry_start,
                exit_status=rc,
            )
            return rc
        if token == "orchestrator":  # nosec B105
            from src.entrypoints.orchestrator import run_orchestrator_subcommand

            rc = run_orchestrator_subcommand(rest_args)
            _telemetry_record_end(
                session_id=_telemetry_session_id,
                command_name="orchestrator",
                mode="daemon",
                success=(rc == 0),
                duration_s=time.monotonic() - _telemetry_start,
                exit_status=rc,
            )
            return rc
        if token == "autonomy":  # nosec B105
            from clawcodex_ext.cron_system.status import build_autonomy_runs, build_autonomy_status

            deep = "--deep" in rest_args
            filtered_args = [arg for arg in rest_args if arg != "--deep"]
            command = filtered_args[0] if filtered_args else "status"
            rc = 0
            if command == "status":
                print(build_autonomy_status(Path.cwd(), deep=deep))
                rc = 0
            elif command == "runs":
                print(build_autonomy_runs(Path.cwd(), deep=deep))
                rc = 0
            else:
                print("usage: clawcodex autonomy [status|runs] [--deep]", file=sys.stderr)
                rc = 2
            _telemetry_record_end(
                session_id=_telemetry_session_id,
                command_name="autonomy",
                mode="non_interactive",
                success=(rc == 0),
                duration_s=time.monotonic() - _telemetry_start,
                exit_status=rc,
            )
            return rc
        if token == "schedule":  # nosec B105
            from clawcodex_ext.cron_system.schedule import (
                format_cron_task_detail,
                format_manual_fire_result,
                get_cron_task_detail,
                manual_fire_cron_task,
            )
            from clawcodex_ext.cron_system.status import build_schedule_list

            command = rest_args[0] if rest_args else "list"
            rc = 0
            if command == "list":
                print(build_schedule_list(Path.cwd()))
                rc = 0
            elif command == "get" and len(rest_args) >= 2:
                cwd = Path.cwd()
                detail = get_cron_task_detail(cwd, rest_args[1])
                if detail is None:
                    print(f"No scheduled job with id '{rest_args[1]}'", file=sys.stderr)
                    rc = 1
                else:
                    print(format_cron_task_detail(detail))
                    rc = 0
            elif command == "run" and len(rest_args) >= 2:
                cwd = Path.cwd()
                run = manual_fire_cron_task(cwd, rest_args[1], current_dir=cwd)
                if run is None and get_cron_task_detail(cwd, rest_args[1]) is None:
                    print(f"No scheduled job with id '{rest_args[1]}'", file=sys.stderr)
                    rc = 1
                else:
                    print(format_manual_fire_result(rest_args[1], run))
                    rc = 0
            else:
                print("usage: clawcodex schedule [list|get ID|run ID]", file=sys.stderr)
                rc = 2
            _telemetry_record_end(
                session_id=_telemetry_session_id,
                command_name="schedule",
                mode="non_interactive",
                success=(rc == 0),
                duration_s=time.monotonic() - _telemetry_start,
                exit_status=rc,
            )
            return rc

    from clawcodex_ext.cli.parser import build_parser

    parser = build_parser()
    args = parser.parse_args(argv[1:])
    profile_checkpoint("argparse_done")

    swarm_requested = bool(getattr(args, "swarm", False)) or getattr(args, "effort", None) == "swarm"
    if swarm_requested:
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

    if getattr(args, "prompt", None) and not getattr(args, "print", False):
        parser.error(f"unknown command: {args.prompt} (use -p/--print to send a prompt)")

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

    if args.version:
        from src import __version__

        print(f"claw-codex version {__version__} (Python)")
        return 0

    if args.config:
        import src.cli as src_cli

        return src_cli.show_config()

    # ---- Feature Gate CLI overrides ----------------------------------
    # Apply ``--enable-feature`` / ``--disable-feature`` before the
    # agent loop starts.  These programmatic overrides take priority
    # over env-vars and config-file values.
    _apply_feature_gate_overrides(args)

    # Plan-phase-1 wiring (ch02-bootstrap-refactoring-plan.md P1.5):
    # ``run_pre_action(args)`` is the Python analog of Commander's
    # ``preAction`` hook. It runs the memoized ``init()`` (chapter
    # phase 2 — safe env vars + graceful-shutdown + API preconnect)
    # and mutates interactive bootstrap state.
    #
    # MUST PRECEDE permission resolution so init-side env-var
    # application can affect permission resolution. ``--version`` /
    # ``--config`` short-circuit above, so the chapter's
    # "fast paths skip init" property is preserved.
    #
    # The API-preconnect call previously lived here at module level;
    # it now runs inside ``init()`` so it overlaps with any callers
    # of ``init()`` (REPL, headless, etc.), not just the cli.py path.
    profile_checkpoint("phase0_end_phase2_start")
    from src.init import run_pre_action

    run_pre_action(args)
    profile_checkpoint("phase2_end_phase3_start")

    # Resolve permission state ONCE here so all modes (print/TUI/REPL) honor
    # ``--dangerously-skip-permissions`` consistently. Mirrors
    # ``typescript/src/main.tsx:1383-1389``.
    from clawcodex_ext.cli.permissions import resolve_permission_state
    from clawcodex_ext.cli.runners import _split_csv
    from clawcodex_ext.frontend import get_frontend
    from clawcodex_ext.runtime.context import RuntimeContext, RuntimeOptions

    resolve_permission_state(args)
    profile_checkpoint("permissions_resolved")
    profile_checkpoint("phase3_end_phase4_start")

    if getattr(args, "fallback_model", None) and args.fallback_model == getattr(args, "model", None):
        print("error: --fallback-model must differ from --model", file=sys.stderr)
        return 2

    from clawcodex_ext.cli.worktree import WORKTREE_FAILED, maybe_create_worktree

    worktree_session = maybe_create_worktree(args)
    if worktree_session is WORKTREE_FAILED:
        return 1
    if worktree_session is not None:
        os.chdir(worktree_session.worktree_path)
        os.environ.update(worktree_session.to_env())

    # Interactive path: decide between the Textual TUI (new default) and the
    # legacy Rich REPL. Explicit flags win; otherwise auto-detect a compatible TTY.
    explicit_tui: bool | None = None
    if args.tui:
        explicit_tui = True
    elif getattr(args, "legacy_repl", False) or args.no_tui:
        explicit_tui = False

    # ``--resume`` without a SESSION_ID means "browse" mode.
    # REPL mode now has its own session browser, so no need to force TUI.
    resume_val = getattr(args, "resume", None)

    bundle_path: Path | None = None
    agent_type_raw = getattr(args, "agent", None)
    if agent_type_raw is not None and agent_type_raw != "auto":
        candidate = Path(str(agent_type_raw)).resolve()
        if candidate.is_dir():
            bundle_path = candidate

    # F-157: command-line selection wins over an in-process runtime choice,
    # which in turn wins over config.yaml's default_group.
    from clawcodex_ext.multimodel.config import MultiModelConfigError  # fmt: skip

    try:
        from clawcodex_ext.multimodel.config import load_config, resolve_active_group

        _multimodel_config = load_config()
        _multimodel_group = resolve_active_group(
            cli_group=getattr(args, "multimodel", None),
            runtime_group=getattr(args, "runtime_multimodel", None),
            config=_multimodel_config,
        )
        if _multimodel_group and _multimodel_group not in _multimodel_config.groups:
            raise MultiModelConfigError(f"unknown model group '{_multimodel_group}'")
        if _multimodel_group:
            from clawcodex_ext.multimodel.feature import require_multimodel_enabled

            require_multimodel_enabled()
    except (MultiModelConfigError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # Build RuntimeContext once from resolved args — shared by all frontends.
    runtime_opts = RuntimeOptions(
        provider_name=getattr(args, "provider", None),
        model=getattr(args, "model", None),
        fallback_model=getattr(args, "fallback_model", None),
        effort=(args.effort if getattr(args, "effort", None) not in (None, "normal", "swarm") else None),
        prompt=getattr(args, "prompt", None),
        output_format=getattr(args, "output_format", "text"),
        input_format=getattr(args, "input_format", "text"),
        include_partial_messages=getattr(args, "include_partial_messages", False),
        max_turns=getattr(args, "max_turns", 20),
        max_turns_explicit=any(token == "--max-turns" or token.startswith("--max-turns=") for token in argv[1:]),  # nosec B105
        allowed_tools=tuple(_split_csv(getattr(args, "allowed_tools", None))),
        disallowed_tools=tuple(_split_csv(getattr(args, "disallowed_tools", None))),
        stream=getattr(args, "stream", False),
        permission_mode=getattr(args, "_resolved_permission_mode", "default"),
        is_bypass_permissions_mode_available=getattr(args, "_resolved_is_bypass_available", False),
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
    )
    runtime_opts.multimodel_group = _multimodel_group
    if _is_provider_free_goal_summary_print(args) and not _multimodel_group:
        from src.entrypoints.headless import HeadlessOptions, run_headless

        headless_options = HeadlessOptions(
            prompt=runtime_opts.prompt,
            output_format=runtime_opts.output_format,
            input_format=runtime_opts.input_format,
            provider_name=runtime_opts.provider_name,
            model=runtime_opts.model,
            fallback_model=runtime_opts.fallback_model,
            effort=runtime_opts.effort,
            max_turns=runtime_opts.max_turns,
            max_turns_explicit=runtime_opts.max_turns_explicit,
            permission_mode=runtime_opts.permission_mode,
            is_bypass_permissions_mode_available=runtime_opts.is_bypass_permissions_mode_available,
            skip_permissions=runtime_opts.skip_permissions,
            allowed_tools=runtime_opts.allowed_tools,
            disallowed_tools=runtime_opts.disallowed_tools,
            include_partial_messages=runtime_opts.include_partial_messages,
            verbose=runtime_opts.verbose,
            workspace_root=runtime_opts.workspace_root or Path.cwd(),
            append_system_prompt=runtime_opts.append_system_prompt,
            startup_agent=runtime_opts.startup_agent,
            resume_session_id=runtime_opts.resume_session_id,
            resume_session_at=runtime_opts.resume_session_at,
            record=runtime_opts.record,
            record_width=runtime_opts.record_width,
            record_height=runtime_opts.record_height,
        )
        rc = _run_with_worktree_keep_note(lambda: run_headless(headless_options), worktree_session)
        _telemetry_record_end(
            session_id=_telemetry_session_id,
            command_name="print",
            mode="non_interactive",
            success=(rc == 0),
            duration_s=time.monotonic() - _telemetry_start,
            exit_status=rc,
        )
        return rc

    try:
        ctx = RuntimeContext.build(runtime_opts)
        ctx.multimodel_group = _multimodel_group
    except RuntimeError as exc:
        # Configuration errors (missing API key, no provider selected, etc.)
        # are not programmer errors — surface a clean warning instead of a
        # traceback so the user knows exactly how to recover.
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
        return 1

    # ---- Agent type resolution: --agent flag or auto-detect ----
    _resolve_startup_agent(args, ctx)

    # Select frontend by name; dispatch stays as the thin orchestration layer.
    if args.print:
        # F-97 telemetry notice — shown once on stderr for headless/CLI mode
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
            pass
        profile_checkpoint("mode_dispatch_print")
        profile_checkpoint("phase4_dispatch")
        frontend = get_frontend("headless")
        rc = _run_with_worktree_keep_note(lambda: frontend.run(ctx, argv[1:]), worktree_session)
        _telemetry_record_end(
            session_id=_telemetry_session_id,
            command_name="print",
            mode="non_interactive",
            success=(rc == 0),
            duration_s=time.monotonic() - _telemetry_start,
            exit_status=rc,
        )
        return rc

    from src.entrypoints.tui import should_use_tui

    if should_use_tui(explicit_tui):
        profile_checkpoint("mode_dispatch_tui")
        profile_checkpoint("phase4_dispatch")
        frontend = get_frontend("tui")
        rc = _run_with_worktree_keep_note(lambda: frontend.run(ctx, argv[1:]), worktree_session)
        _telemetry_record_end(
            session_id=_telemetry_session_id,
            command_name="tui",
            mode="interactive",
            success=(rc == 0),
            duration_s=time.monotonic() - _telemetry_start,
            exit_status=rc,
        )
        return rc

    profile_checkpoint("mode_dispatch_repl")
    profile_checkpoint("phase4_dispatch")

    frontend = get_frontend("repl")
    rc = _run_with_worktree_keep_note(lambda: frontend.run(ctx, argv[1:]), worktree_session)
    _telemetry_record_end(
        session_id=_telemetry_session_id,
        command_name="repl",
        mode="interactive",
        success=(rc == 0),
        duration_s=time.monotonic() - _telemetry_start,
        exit_status=rc,
    )
    return rc


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
                pass
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
            pass

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

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

# pylint: disable=no-name-in-module
"""Fast-path downstream CLI subcommand registry."""

from __future__ import annotations

from collections.abc import Callable

SubcommandHandler = Callable[[list[str]], int]

_SUBCOMMANDS: dict[str, SubcommandHandler] = {}
_LOADED = False


def register(name: str) -> Callable[[SubcommandHandler], SubcommandHandler]:
    """Register a fast-path subcommand handler."""

    def decorator(handler: SubcommandHandler) -> SubcommandHandler:
        _SUBCOMMANDS[name] = handler
        return handler

    return decorator


def get_subcommand(name: str) -> SubcommandHandler | None:
    load_builtin_subcommands()
    return _SUBCOMMANDS.get(name)


def load_builtin_subcommands() -> None:
    global _LOADED
    if _LOADED:
        return
    _LOADED = True

    from clawcodex_ext.cli.provider_cmd import commands as _provider_commands  # noqa: F401
    from clawcodex_ext.cli.model_cmd import commands as _model_commands  # noqa: F401
    from clawcodex_ext.multimodel import cli as _multimodel_cli  # noqa: F401
    from clawcodex_ext.cli.sop_cmd import commands as _sop_commands  # noqa: F401
    from clawcodex_ext.cli import telemetry_cmd as _telemetry_cmd  # noqa: F401

    # IM Message Gateway: `clawcodex-dev gateway start|stop|status|restart` (daemon)
    # and `clawcodex-dev gateway setup|status|restart|disconnect|login [<name>]` (channel config).
    from clawcodex_ext.cli.gateway_cmd import commands as _gateway_commands  # noqa: F401

    # ``clawcodex auth logout|status|zeroize``
    from clawcodex_ext.cli import auth_cmd as _auth_cmd  # noqa: F401

    # `clawcodex template list|show|create` subcommand.
    # Some downstream checkouts do not ship the CLI wrapper yet even though
    # the template service modules are present. Treat it as optional so a
    # missing template command cannot break unrelated entry points such as
    # `clawcodex-dev viz`.
    try:
        from clawcodex_ext.cli import template_cmd as _template_cmd  # noqa: F401
    except ImportError:
        _template_cmd = None

    # ``clawcodex-dev session migrate`` subcommand for
    # converting legacy 3-file sessions to the unified 2-file format.
    from clawcodex_ext.cli import session_migrate_cmd as _session_migrate_cmd  # noqa: F401

    # ``clawcodex viz`` subcommand for the Multi-Session Visualizer
    # wrapped in try/except so a partial checkout that lacks
    # the visualizer package (e.g. CI smoke that runs only the core CLI
    # surfaces) does not break unrelated subcommand discovery. The
    # inner ``register_viz_subcommand()`` already self-contains a
    # try/except, so the outer guard only protects against the import
    # itself raising.
    try:
        from extensions.visualizer.cli import register_viz_subcommand  # noqa: F401

        register_viz_subcommand()
    except Exception:  # nosec B110
        pass  # This optional feature is unavailable; continue with the core command path.

    # ``clawcodex stats`` subcommand for tool/skill usage statistics
    from clawcodex_ext.cli import stats_cmd as _stats_cmd  # noqa: F401

    # ``clawcodex forecast`` Intent Forecast subcommand.
    from clawcodex_ext.intent_forecast import cli as _intent_forecast_cli  # noqa: F401

    # ``clawcodex feature`` subcommand for runtime feature-gate management
    from clawcodex_ext.feature_gate.cli import run_feature_command  # noqa: F401

    from extensions.remote_api.cli import register_api_subcommand

    register_api_subcommand()

    # SR-5.1: ``clawcodex-dev community-radar scan|status|source|config``
    from clawcodex_ext.community_radar.cli import register_community_radar_subcommand

    register_community_radar_subcommand()

    # ``clawcodex-dev diag freeze-report|viewer|status`` subcommand
    # for Layer-4 freeze-detection diagnostics. Imports the module so the
    # ``@register('diag')`` decorator fires; the handler is looked up
    # by name in the sieve above.
    from clawcodex_ext.cli import diag_cmd as _diag_cmd  # noqa: F401  # noqa: F401

    # ``clawcodex-dev tool <name> [--args]`` subcommand that
    # auto-dispatches to any discoverable tool. Idempotent: a single
    # subcommand name is registered (``tool``) — per-tool name routing
    # happens inside the handler to keep the sieve deterministic.
    from clawcodex_ext.cli.tool_cmd.hooks import install_tool_subcommand

    install_tool_subcommand()

    # F-REC: ``clawcodex record --sources ... --out <path>`` asciicast v2
    # recorder. Imported for side-effect — the @register("record")
    # decorator inside the module wires the subcommand into the
    # registry. Wrapped in try/except so a partial checkout that lacks
    # the recording extension (e.g. CI smoke) does not break unrelated
    # subcommand discovery.
    try:
        from extensions.recording import cli as _recording_cli  # noqa: F401
    except ImportError:
        _recording_cli = None

    # F-REC: ``clawcodex cast-to-mp4 --cast <in.cast> --out <out.mp4>``
    # post-processor that turns a recorded .cast into a video. Imported
    # for side-effect — the @register("cast-to-mp4") decorator wires it
    # into the registry. Same try/except guard so a missing Pillow (the
    # cast-to-mp4 dependency) does not break unrelated subcommands.
    try:
        from extensions.recording import cast_to_mp4_cli as _cast_to_mp4_cli  # noqa: F401
    except ImportError:
        _cast_to_mp4_cli = None

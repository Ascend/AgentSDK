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

"""Downstream CLI parser — owns build_parser()."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="clawcodex",
        description="ClawCodex - Claude Code Python Implementation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  clawcodex --version                   Show version
  clawcodex login                       Configure API keys or ChatGPT OAuth
  clawcodex config                      Show current configuration
  clawcodex --stream                    Start REPL with live response rendering
  clawcodex                             Start interactive REPL
  clawcodex -p "hello"                  Non-interactive mode (text output)
  clawcodex -p "hi" --output-format json
  clawcodex -p --output-format stream-json --input-format stream-json < input.ndjson
""",
    )

    parser.add_argument("prompt", nargs="?", help="Prompt to send in non-interactive mode")
    parser.add_argument("--version", action="store_true", help="Show version information")
    parser.add_argument("--config", action="store_true", help="Show current configuration")
    parser.add_argument(
        "-w",
        "--worktree",
        nargs="?",
        const=True,
        default=None,
        metavar="NAME",
        help=(
            "Run this session in an isolated git worktree at "
            ".clawcodex/worktrees/NAME (a name is generated when omitted). "
            "Prefer --worktree=NAME when combining it with a positional prompt."
        ),
    )
    parser.add_argument("--stream", action="store_true", help="Enable live rendering in REPL")
    parser.add_argument(
        "--agent-debug",
        action="store_true",
        help="Enable external-agent REPL debug markers and sandbox-safe local state",
    )
    decomposition = parser.add_mutually_exclusive_group()
    decomposition.add_argument(
        "--swarm",
        dest="swarm",
        action="store_true",
        help="Dynamically decompose the prompt and run it through coordinator workers",
    )
    decomposition.add_argument(
        "--decompose",
        dest="swarm",
        action="store_true",
        help="Alias for --swarm",
    )
    parser.add_argument(
        "--effort",
        choices=("normal", "swarm", "low", "medium", "high", "xhigh", "max"),
        default=None,
        help=(
            "Reasoning effort for this session; 'swarm' enables downstream "
            "task decomposition and 'normal' keeps automatic model defaults"
        ),
    )

    # ---- Interactive UI selection ----
    #
    # The default interactive experience is the prompt_toolkit + rich REPL,
    # which matches the TS Ink reference's terminal-native behavior:
    # transcript flows into scrollback, only the prompt + status row are
    # live, and native mouse copy works. ``--tui`` opts into the Textual
    # in-app experience; ``--legacy-repl`` / ``--no-tui`` are kept as
    # no-op aliases for back-compat (they already select the default).
    ui_group = parser.add_mutually_exclusive_group()
    ui_group.add_argument(
        "--tui",
        action="store_true",
        help="Use the Textual in-app TUI (opt-in; default is the inline REPL)",
    )
    ui_group.add_argument(
        "--legacy-repl",
        dest="legacy_repl",
        action="store_true",
        help="Use the inline prompt_toolkit + rich REPL (this is the default)",
    )
    ui_group.add_argument(
        "--no-tui",
        dest="no_tui",
        action="store_true",
        help="Alias for --legacy-repl (kept for backward compatibility)",
    )

    # ---- Non-interactive / print mode (Phase 1 parity) ----
    noninteractive = parser.add_argument_group("non-interactive mode")
    noninteractive.add_argument(
        "-p",
        "--print",
        action="store_true",
        help="Print response and exit (useful for pipes)",
    )
    noninteractive.add_argument(
        "--output-format",
        choices=("text", "json", "stream-json"),
        default="text",
        help="Output format for --print mode (default: text)",
    )
    noninteractive.add_argument(
        "--input-format",
        choices=("text", "stream-json"),
        default="text",
        help="Input format for --print mode (default: text)",
    )
    noninteractive.add_argument(
        "--include-partial-messages",
        action="store_true",
        help="Include incremental assistant text chunks in stream-json output",
    )
    noninteractive.add_argument(
        "--max-turns",
        type=int,
        default=20,
        help="Maximum number of agent tool turns (default: 20)",
    )
    noninteractive.add_argument(
        "--model",
        type=str,
        default=None,
        help="Override the model used for this run",
    )
    noninteractive.add_argument(
        "--fallback-model",
        type=str,
        default=None,
        dest="fallback_model",
        help="Model to use after repeated overloaded (529) errors",
    )
    noninteractive.add_argument(
        "--provider",
        type=str,
        default=None,
        help="Override the provider (anthropic, openai, openai-codex, glm, minimax, openrouter, deepseek)",
    )
    noninteractive.add_argument(
        "--multimodel",
        type=str,
        default=None,
        metavar="GROUP",
        help="Use a configured multi-model group for this run (highest selection priority)",
    )
    noninteractive.add_argument(
        "--allowed-tools",
        type=str,
        default=None,
        help="Comma-separated list of tools allowed to run",
    )
    noninteractive.add_argument(
        "--disallowed-tools",
        type=str,
        default=None,
        help="Comma-separated list of tools that must NOT run",
    )
    noninteractive.add_argument(
        "--resume",
        nargs="?",
        default=None,
        const="browse",
        metavar="SESSION_ID",
        help="Resume a previous session by session ID; without SESSION_ID, browse sessions",
    )
    noninteractive.add_argument(
        "-c",
        "--continue",
        action="store_true",
        default=None,
        help="Resume the most recent session (alias for --resume with auto-detected latest)",
    )
    noninteractive.add_argument(
        "--fork-session",
        type=str,
        default=None,
        metavar="SESSION_ID",
        help="Create a new session ID but preserve conversation history from the given session",
    )
    noninteractive.add_argument(
        "--resume-session-at",
        type=str,
        default=None,
        metavar="MSG_INDEX",
        help="Resume a session at a specific message index (0-based); requires --resume or --continue",
    )
    noninteractive.add_argument(
        "--verbose",
        action="store_true",
        help="Emit verbose diagnostics to stderr",
    )

    im_group = parser.add_argument_group("IM gateway opt-in")
    im_group.add_argument(
        "--gateway",
        dest="gateway",
        action="store_true",
        help="Bind this REPL session to all WeChat direct/private messages.",
    )
    im_group.add_argument(
        "--im-gateway",
        dest="gateway",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    im_group.add_argument(
        "--gateway-origin",
        dest="gateway_origin",
        type=str,
        default=None,
        metavar="ORIGIN",
        help=("Advanced: bind this REPL session to a specific IM gateway origin, e.g. wechat:direct:default:user_id"),
    )
    im_group.add_argument(
        "--im-gateway-origin",
        dest="gateway_origin",
        type=str,
        default=None,
        metavar="ORIGIN",
        help=argparse.SUPPRESS,
    )
    im_group.add_argument(
        "--gateway-sock",
        dest="gateway_sock",
        type=str,
        default=None,
        metavar="PATH",
        help=("Gateway daemon Unix socket for --gateway-origin (default: ~/.clawcodex/gateway/gateway.sock)"),
    )
    im_group.add_argument(
        "--im-gateway-sock",
        dest="gateway_sock",
        type=str,
        default=None,
        metavar="PATH",
        help=argparse.SUPPRESS,
    )

    # ---- Permissions ----
    # ``--dangerously-skip-permissions`` and ``--allow-dangerously-skip-permissions``
    # apply to all UI modes (REPL, TUI, headless), so they live in a top-level
    # group rather than under ``noninteractive``. Mirrors the TS reference at
    # ``typescript/src/main.tsx:970``.
    permissions_group = parser.add_argument_group("permissions")
    permissions_group.add_argument(
        "--dangerously-skip-permissions",
        dest="dangerously_skip_permissions",
        action="store_true",
        help=("Bypass all permission checks. Recommended only for sandboxes with no internet access."),
    )
    permissions_group.add_argument(
        "--allow-dangerously-skip-permissions",
        dest="allow_dangerously_skip_permissions",
        action="store_true",
        help=(
            "Enable bypassing all permission checks as an option, without it "
            "being enabled by default. Recommended only for sandboxes with "
            "no internet access."
        ),
    )
    parser.add_argument(
        "--record",
        type=str,
        default=None,
        metavar="CAST_PATH",
        help=(
            "Record the REPL session to an asciicast v2 .cast file. "
            "Captures console output (Rich) and user input; plays back "
            "with asciinema-player. Requires the recording extension."
        ),
    )
    parser.add_argument(
        "--record-width",
        type=int,
        default=None,
        metavar="COLS",
        help="Terminal width for the recorded .cast header (default: auto-detect).",
    )
    parser.add_argument(
        "--record-height",
        type=int,
        default=None,
        metavar="ROWS",
        help="Terminal height for the recorded .cast header (default: auto-detect).",
    )

    # ---- Agent type selection ----
    parser.add_argument(
        "--agent",
        type=str,
        default=None,
        nargs="?",
        const="auto",
        metavar="AGENT_TYPE",
        help=(
            "Start with a specific agent type. Without a value, auto-detect "
            "clawcodex-overview.md in .claude/agents/. Default: general-purpose agent."
        ),
    )

    permissions_group.add_argument(
        "--permission-mode",
        dest="permission_mode",
        choices=("default", "plan", "acceptEdits", "bypassPermissions", "dontAsk", "auto"),
        default=None,
        help='Initial permission mode (default: default). "auto" uses LLM classifier.',
    )

    # ---- Feature Gates ----
    # Runtime feature-toggle controls. These are applied before the
    # agent loop starts and take highest priority (above env vars and
    # config file).
    feature_group = parser.add_argument_group("feature gates")
    feature_group.add_argument(
        "--enable-feature",
        dest="enable_feature",
        type=str,
        action="append",
        metavar="NAME",
        help="Enable a runtime feature flag by name (may be repeated)",
    )
    feature_group.add_argument(
        "--disable-feature",
        dest="disable_feature",
        type=str,
        action="append",
        metavar="NAME",
        help="Disable a runtime feature flag by name (may be repeated)",
    )

    # Subcommands are intercepted in ``main`` before argparse runs so that a
    # free-form prompt argument cannot be misinterpreted as a subcommand.
    # Listing them here purely for ``--help`` documentation.
    commands_group = parser.add_argument_group("subcommands")
    commands_group.add_argument(
        "--_commands_doc",
        help=argparse.SUPPRESS,
    )
    parser.epilog = (parser.epilog or "") + (
        "\nSubcommands:\n"
        "  login    Configure API keys or ChatGPT OAuth (interactive)\n"
        "  config   Show current configuration\n"
        "  autonomy Show scheduled-task status and runs\n"
        "  feature  Manage runtime feature flags\n"
    )
    return parser

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

"""``clawcodex-dev gateway`` subcommand — manage the Gateway daemon and IM channels.

Flattened single-level verbs. Disambiguation: a verb followed by a
positional channel name is a channel operation; a verb alone (with only
optional flags) is a daemon operation.

Usage:
  clawcodex-dev gateway start [--state-dir X] [-v|--verbose]
  clawcodex-dev gateway stop [--state-dir X]
  clawcodex-dev gateway restart [--state-dir X] [-v|--verbose]
  clawcodex-dev gateway status [--state-dir X]
      Bare status: daemon health + all-channels overview (unified view).
  clawcodex-dev gateway status <name>         Channel health/status
  clawcodex-dev gateway restart <name>        Rebuild a channel adapter
  clawcodex-dev gateway disconnect <name>     Remove REPL/orchestrator connection
  clawcodex-dev gateway login <name>          WeChat iLink QR login
  clawcodex-dev gateway setup                 Guided channel configuration wizard

All commands are idempotent. v1 runs POSIX UDS only; acceptance is
limited to POSIX/WSL/Git Bash.
"""

from __future__ import annotations

import sys

from clawcodex_ext.cli.subcommand_registry import register

_USAGE = (
    "usage: clawcodex-dev gateway {start|stop|status|restart|setup|disconnect|login} [options]\n"
    "       clawcodex-dev gateway {status|restart|disconnect|login} <name>\n\n"
    "Manage the IM Message Gateway daemon and channel configuration.\n"
    "  start            Start the daemon (idempotent; --state-dir, -v/--verbose).\n"
    "  stop             Stop the daemon (idempotent; --state-dir).\n"
    "  restart          Restart the daemon (no positional).\n"
    "  restart <name>   Rebuild a channel adapter and reload its config.\n"
    "  status           Daemon health + all-channels overview (unified view).\n"
    "  status <name>    Show channel health/status.\n"
    "  setup            Configure channels, then restart the Gateway daemon.\n"
    "  disconnect <name> Remove the active REPL/orchestrator connection.\n"
    "  login <name>     WeChat iLink QR login.\n"
    "  --state-dir PATH  Override gateway state directory.\n"
    "  -v, --verbose     Enable DEBUG-level IM logging (default: only WARNING and above).\n"
    "                    Also set via CLAWCODEX_GATEWAY_LOG_LEVEL=INFO|DEBUG.\n"
)


@register("gateway")
def run_gateway_command(args: list[str]) -> int:
    from extensions.im_gateway.server import DaemonPaths, GatewayDaemon

    if not args or args[0] in ("help", "--help", "-h"):
        print(_USAGE)
        return 0

    # Parse global flags: --state-dir <path> and -v/--verbose.
    state_dir: str | None = None
    verbose = False
    positional: list[str] = []
    i = 0
    while i < len(args):
        if args[i] == "--state-dir" and i + 1 < len(args):
            state_dir = args[i + 1]
            i += 2
            continue
        if args[i] in ("-v", "--verbose"):
            verbose = True
            i += 1
            continue
        positional.append(args[i])
        i += 1

    if not positional:
        print(_USAGE)
        return 0

    verb = positional[0]
    rest = positional[1:]

    # -- channel operations (verb + positional name) ------------------------

    if verb == "status" and rest:
        from clawcodex_ext.cli.channels_cmd.commands import format_status

        print(format_status(None, rest[0], state_dir=state_dir))
        return 0

    if verb == "restart" and rest:
        from clawcodex_ext.cli.channels_cmd.commands import restart_channel

        return restart_channel(rest[0], state_dir=state_dir)

    if verb in ("disconnect", "unbind"):
        if not rest:
            print(f"error: gateway {verb} <name>", file=sys.stderr)
            return 2
        from clawcodex_ext.cli.channels_cmd.commands import _disconnect_gateway_connection

        return _disconnect_gateway_connection(rest[0], state_dir=state_dir)

    if verb == "login":
        if not rest:
            print("error: gateway login <wechat-name>", file=sys.stderr)
            return 2
        from clawcodex_ext.cli.channels_cmd.commands import wechat_login

        return wechat_login(rest[0], state_dir=state_dir)

    # -- daemon / setup operations (verb alone) -----------------------------

    paths = DaemonPaths.for_state_dir(state_dir)
    daemon = GatewayDaemon(paths)

    if verb == "status":
        # Unified view: daemon status line + all channels.
        daemon.status()
        from clawcodex_ext.cli.channels_cmd.commands import format_status

        print(format_status(None, None, state_dir=state_dir))
        return 0

    if verb == "start":
        if rest:
            print("error: start takes no channel name", file=sys.stderr)
            return 2
        return daemon.start(verbose=verbose)

    if verb == "stop":
        if rest:
            print("error: stop takes no channel name", file=sys.stderr)
            return 2
        return daemon.stop()

    if verb == "restart":
        return daemon.restart(verbose=verbose)

    if verb == "setup":
        from clawcodex_ext.cli.channels_cmd.commands import run_wizard

        config_path = paths.state_dir / "channels.yaml"
        setup_result = run_wizard(str(config_path))
        if setup_result != 0:
            return setup_result
        return daemon.restart(verbose=verbose)

    print(f"error: unknown gateway subcommand {verb!r}", file=sys.stderr)
    print(_USAGE, file=sys.stderr)
    return 2


__all__ = ["run_gateway_command"]

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright (c) 2026 Clawd Codex Team
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

"""IM Message Gateway daemon process.

Hosts :class:`MessageGateway` as a long-running daemon, listening on a
POSIX UDS socket for REPL/orchestrator opt-in clients. Lifecycle is
managed by ``clawcodex-dev gateway server start|stop|status|restart``
via :mod:`clawcodex_ext.cli.gateway_cmd`.

v1 (P1) ships lifecycle + PID/lock/stale-socket/health. WeChat adapter
hosting lands in P2; the full ``GatewayIpcProtocol`` listener + agent
registry in P2/P3; reliability hardening in P4; default host agent in P5.
"""

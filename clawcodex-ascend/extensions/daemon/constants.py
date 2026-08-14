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
"""Constants for the daemon subsystem."""

from __future__ import annotations

EXIT_CODE_PERMANENT: int = 78

EXIT_CODE_TRANSIENT: int = 1

EXIT_CODE_OK: int = 0

BACKOFF_INITIAL_MS: int = 2_000

BACKOFF_CAP_MS: int = 120_000

BACKOFF_MULTIPLIER: int = 2

MAX_RAPID_FAILURES: int = 5

RAPID_FAILURE_WINDOW_MS: int = 10_000

GRACEFUL_SHUTDOWN_TIMEOUT_MS: int = 30_000

DAEMON_STATE_DIRNAME: str = ".clawcodex"

DAEMON_STATE_SUBDIR: str = "daemon"

DAEMON_STATE_FILENAME_EXT: str = ".json"

ENV_VAR_SUPERVISOR_PID: str = "CLAWCODEX_SUPERVISOR_PID"
ENV_VAR_DAEMON_NAME: str = "CLAWCODEX_DAEMON_NAME"
ENV_VAR_DAEMON_DIR: str = "CLAWCODEX_DAEMON_DIR"
ENV_VAR_DAEMON_SPAWN_MODE: str = "CLAWCODEX_DAEMON_SPAWN_MODE"
ENV_VAR_DAEMON_CAPACITY: str = "CLAWCODEX_DAEMON_CAPACITY"
ENV_VAR_DAEMON_PERMISSION_MODE: str = "CLAWCODEX_DAEMON_PERMISSION_MODE"
ENV_VAR_DAEMON_SANDBOX: str = "CLAWCODEX_DAEMON_SANDBOX"
ENV_VAR_DAEMON_TIMEOUT_MS: str = "CLAWCODEX_DAEMON_TIMEOUT_MS"
ENV_VAR_DAEMON_SESSION_KIND: str = "CLAWCODEX_DAEMON_SESSION_KIND"

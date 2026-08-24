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

"""Focused construction test for Swarm mode after A.12 is available."""

import sys
from types import ModuleType, SimpleNamespace


git_sync = ModuleType("extensions.orchestrator.git_sync")


class VerificationFailed(RuntimeError):
    """Minimal A.3 boundary used while Git Sync migrates independently."""


git_sync.VerificationFailed = VerificationFailed
sys.modules.setdefault("extensions.orchestrator.git_sync", git_sync)

from extensions.orchestrator.multi_agent.modes.swarm import (  # noqa: E402
    SwarmModeRunner,
    _control_namespace,
)


def test_swarm_configures_bounded_decomposer() -> None:
    runner = SwarmModeRunner(object(), max_subtasks=5, max_parallel=2, max_waves=4)

    assert runner._decomposer.max_subtasks == 5
    assert runner._decomposer.max_parallel == 2
    assert runner._decomposer.max_waves == 4


def test_control_namespace_is_stable_and_issue_scoped() -> None:
    first = SimpleNamespace(issue=SimpleNamespace(id="owner/issue-1", identifier="ISSUE-1"))
    second = SimpleNamespace(issue=SimpleNamespace(id="owner/issue-2", identifier="ISSUE-2"))

    assert _control_namespace(first) == "swarm-owner_issue-1"
    assert _control_namespace(first) != _control_namespace(second)

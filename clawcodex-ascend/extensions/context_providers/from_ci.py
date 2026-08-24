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
"""from_ci — CI pipeline status context provider.

Registers a ``register_section`` builder that reads ``runtime_ctx["ci_status"]``
and injects a brief "CI Status" block at ``order=56``.

Usage
-----
Importing this module triggers registration at module-load time::

    from extensions.context_providers import from_ci  # noqa: F401

The builder returns ``None`` when ``ci_status`` is absent, so importing
the module is safe in non-CI environments.

Tags
----
``ci``
"""

from __future__ import annotations

from clawcodex_ext.context_system.section_registry import (
    SectionScope,
    register_section,
)

__all__: list[str] = []


def _ci_status_builder(runtime_ctx: dict) -> str | None:
    """Build the CI-status section block.

    Returns ``None`` (skip section) when ``ci_status`` is not set in the
    runtime context, or a markdown summary otherwise.
    """
    ci = runtime_ctx.get("ci_status")
    if ci is None:
        return None

    ci_str = str(ci).strip()
    if not ci_str:
        return None

    return f"## CI Status\n- Current: {ci_str}\n"


register_section(
    "ci-status",
    builder=_ci_status_builder,
    order=56,
    cache_scope=SectionScope.REQUEST,
    tags=["ci"],
)

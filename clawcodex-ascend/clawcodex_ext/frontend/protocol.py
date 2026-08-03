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

"""Downstream Frontend protocol and plugin base class."""

from __future__ import annotations
# pylint: disable=W2301

from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Frontend(Protocol):
    """Protocol for CLI frontends (REPL, TUI, headless)."""

    @property
    def name(self) -> str:
        """Unique identifier, e.g. 'repl', 'tui', 'headless'."""
        ...

    @property
    def display_name(self) -> str:
        """Human-readable name, e.g. 'Interactive REPL'."""
        ...

    def run(self, ctx: Any, argv: list[str]) -> int:
        """Run the frontend with the given RuntimeContext and CLI argv.

        Args:
            ctx: RuntimeContext built from CLI args.
            argv: Remaining command-line arguments (after subcommand).

        Returns:
            CLI exit code.
        """
        ...


class FrontendPlugin(ABC):
    """Base class for frontend plugins.

    Subclass this to create a new frontend. Use :func:`register_frontend`
    as a decorator to register it with the frontend registry.
    """

    name: str
    display_name: str

    @abstractmethod
    def run(self, ctx: Any, argv: list[str]) -> int:
        """Run the frontend with the given RuntimeContext and CLI argv."""
        ...

    def argparse_group(self, parser) -> None:
        """Add frontend-specific argparse arguments (optional hook)."""
        pass

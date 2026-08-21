#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
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

# pylint: disable=relative-beyond-top-level

"""Workflow singleton store.

Port of Symphony's WorkflowStore.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .config.schema import WorkflowConfig
from .workflow import WorkflowLoader

logger = logging.getLogger(__name__)


class WorkflowStore:
    """Singleton store for the currently-loaded workflow."""

    _instance: "WorkflowStore | None" = None
    _config: WorkflowConfig | None = None
    _prompt_template: str | None = None
    _workflow_path: str | Path | None = None

    def __new__(cls) -> "WorkflowStore":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(self, path: str | Path) -> tuple[WorkflowConfig, str]:
        """Load a workflow file into the store."""
        config, prompt = WorkflowLoader.load(path)
        self._config = config
        self._prompt_template = prompt
        self._workflow_path = str(path)
        return config, prompt

    @property
    def config(self) -> WorkflowConfig | None:
        return self._config

    @property
    def prompt_template(self) -> str | None:
        return self._prompt_template

    @property
    def workflow_path(self) -> str | None:
        return self._workflow_path

    def force_reload(self) -> None:
        """Reload the current workflow file."""
        if self._workflow_path:
            self.load(self._workflow_path)
        else:
            logger.warning("force_reload called before any workflow was loaded; nothing to reload")

    def current(self) -> tuple[WorkflowConfig, str] | None:
        """Return (config, prompt_template) if loaded."""
        if self._config is None or self._prompt_template is None:
            return None
        return self._config, self._prompt_template

    def set_prompt_template(self, template: str) -> str | None:
        """Temporarily override the prompt template; returns the previous value.

        Used by workflow stage runners to provide a stage-specific template
        that suppresses commit/push instructions (the orchestrator's git_sync
        handles all git operations after the workflow completes).
        """
        prev = self._prompt_template
        self._prompt_template = template
        return prev

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (mainly for tests)."""
        cls._instance = None
        cls._config = None
        cls._prompt_template = None
        cls._workflow_path = None


def get_workflow_store() -> WorkflowStore:
    """Get the global WorkflowStore singleton."""
    return WorkflowStore()

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

"""F-97 LODESTONE — built-in target set and registry factory.

The defaults registered here are:

* ``vscode``              — local VS Code URI scheme
* ``vscode-insiders``     — local VS Code Insiders URI scheme
* ``cursor``              — Cursor IDE scheme
* ``idea``                — IntelliJ IDEA ``idea://open?file=…&line=…``
* ``subl``                — Sublime Text ``subl://open?url=…&line=…``
* ``file``                — plain ``file://`` (last-resort fallback)
* ``github``              — gitcode-style ``{remote}/blob/{branch}/{rel}#L{line}``
* ``gitcode``             — explicit ``gitcode.com`` URL
* ``gitee``               — explicit ``gitee.com`` URL
* ``linear``              — Linear issue URL
* ``tracker:gitcode``     — tracker issue → ``gitcode.com/{owner}/{repo}/issues/{n}``
* ``tracker:linear``      — tracker issue → ``linear.app/...``

Call ``build_default_registry(cfg)`` from :class:`LodestoneService`
construction — never re-instantiate these manually except in tests.
"""

from __future__ import annotations

from .models import (
    AnchorTarget,
    AnchorTargetRegistry,
    LodestoneConfig,
)

# Common host whitelist for trusted remote URLs.
GITCODE_HOST = "gitcode.com"
GITHUB_HOST = "github.com"
GITEE_HOST = "gitee.com"
LINEAR_HOST = "linear.app"

_TRACKER_TARGET_KIND = "tracker_issue"


def _editor_targets() -> tuple[AnchorTarget, ...]:
    return (
        AnchorTarget(
            kind="file_path",
            target_id="vscode",
            template="vscode://file/{abs}:{line}:{col}",
            requires=(),
            description="Open in VS Code via the vscode:// URI scheme.",
        ),
        AnchorTarget(
            kind="file_path",
            target_id="vscode-insiders",
            template="vscode-insiders://file/{abs}:{line}:{col}",
            requires=(),
            description="Open in VS Code Insiders.",
        ),
        AnchorTarget(
            kind="file_path",
            target_id="cursor",
            template="cursor://file/{abs}:{line}:{col}",
            requires=(),
            description="Open in Cursor IDE.",
        ),
        AnchorTarget(
            kind="file_path",
            target_id="idea",
            template="idea://open?file={abs}&line={line}&column={col}",
            requires=(),
            description="Open in JetBrains IDE (IntelliJ / PyCharm / …).",
        ),
        AnchorTarget(
            kind="file_path",
            target_id="subl",
            template="subl://open?url=file://{abs}&line={line}&column={col}",
            requires=(),
            description="Open in Sublime Text.",
        ),
        AnchorTarget(
            kind="file_path",
            target_id="file",
            template="file://{abs}",
            requires=(),
            description="Plain file:// link (last-resort fallback).",
        ),
    )


def _remote_targets() -> tuple[AnchorTarget, ...]:
    return (
        AnchorTarget(
            kind="file_path",
            target_id="github",
            is_remote=True,
            template="{remote}/blob/{branch}/{rel}#L{line}",
            hosts=(GITHUB_HOST,),
            description="GitHub-style blob URL with line anchor.",
        ),
        AnchorTarget(
            kind="file_path",
            target_id="gitcode",
            is_remote=True,
            template="{remote}/blob/{branch}/{rel}#L{line}",
            hosts=(GITCODE_HOST,),
            description="GitCode-style blob URL with line anchor.",
        ),
        AnchorTarget(
            kind="file_path",
            target_id="gitee",
            is_remote=True,
            template="{remote}/blob/{branch}/{rel}#L{line}",
            hosts=(GITEE_HOST,),
            description="Gitee-style blob URL with line anchor.",
        ),
        AnchorTarget(
            kind="git_commit",
            target_id="github",
            is_remote=True,
            template="{remote}/commit/{ref}",
            hosts=(GITHUB_HOST,),
            description="GitHub commit URL.",
        ),
        AnchorTarget(
            kind="git_commit",
            target_id="gitcode",
            is_remote=True,
            template="{remote}/commit/{ref}",
            hosts=(GITCODE_HOST,),
            description="GitCode commit URL.",
        ),
        AnchorTarget(
            kind="git_commit",
            target_id="gitee",
            is_remote=True,
            template="{remote}/commit/{ref}",
            hosts=(GITEE_HOST,),
            description="Gitee commit URL.",
        ),
        AnchorTarget(
            kind="git_blob",
            target_id="github",
            is_remote=True,
            template="{remote}/blob/{ref}/{rel}#L{line}",
            hosts=(GITHUB_HOST,),
            description="GitHub blob-at-sha URL.",
        ),
        AnchorTarget(
            kind="git_blob",
            target_id="gitcode",
            is_remote=True,
            template="{remote}/blob/{ref}/{rel}#L{line}",
            hosts=(GITCODE_HOST,),
            description="GitCode blob-at-sha URL.",
        ),
        AnchorTarget(
            kind="git_blob",
            target_id="gitee",
            is_remote=True,
            template="{remote}/blob/{ref}/{rel}#L{line}",
            hosts=(GITEE_HOST,),
            description="Gitee blob-at-sha URL.",
        ),
    )


def _tracker_targets() -> tuple[AnchorTarget, ...]:
    return (
        AnchorTarget(
            kind=_TRACKER_TARGET_KIND,
            target_id="tracker:gitcode",
            is_remote=True,
            template="https://{host}/{owner}/{repo}/issues/{key}",
            hosts=(GITCODE_HOST,),
            description="GitCode issues tracker URL.",
        ),
        AnchorTarget(
            kind=_TRACKER_TARGET_KIND,
            target_id="tracker:linear",
            is_remote=True,
            template="https://{workspace}/issue/{key}",
            hosts=(LINEAR_HOST,),
            description="Linear workspace issue URL.",
        ),
        AnchorTarget(
            kind=_TRACKER_TARGET_KIND,
            target_id="tracker:github",
            is_remote=True,
            template="{remote}/issues/{key}",
            hosts=(GITHUB_HOST,),
            description="GitHub issues tracker URL.",
        ),
        AnchorTarget(
            kind=_TRACKER_TARGET_KIND,
            target_id="tracker:gitee",
            is_remote=True,
            template="{remote}/issues/{key}",
            hosts=(GITEE_HOST,),
            description="Gitee issues tracker URL.",
        ),
    )


def _function_targets() -> tuple[AnchorTarget, ...]:
    """Editors that accept a function-symbol arg (so far only VS Code can
    route by symbol via ``vscode://`` — others fall back to path-anchor).
    """
    return (
        AnchorTarget(
            kind="function_ref",
            target_id="vscode-symbol",
            template="vscode://file/{abs}:{line}:{col}",
            requires=(),
            description="Open the function's defining file at the position.",
        ),
    )


def build_default_registry(
    config: LodestoneConfig | None = None,
    *,
    include_user_targets: bool = True,
) -> AnchorTargetRegistry:
    """Return a registry populated with built-in targets and any
    ``custom_targets`` carried by *config*.  Pass
    ``include_user_targets=False`` from tests that want a clean baseline.
    """
    cfg = config or LodestoneConfig()
    registry = AnchorTargetRegistry(cfg)
    for t in _editor_targets():
        registry.register(t)
    for t in _remote_targets():
        registry.register(t)
    for t in _tracker_targets():
        registry.register(t)
    for t in _function_targets():
        registry.register(t)
    if include_user_targets:
        for t in cfg.custom_targets:
            registry.register(t, overwrite=True)
    return registry


def default_target_ids() -> tuple[str, ...]:
    """Sorted tuple of every built-in target_id (handy for tests/docs)."""
    return tuple(
        sorted(
            {
                t.target_id
                for t in (
                    *_editor_targets(),
                    *_remote_targets(),
                    *_tracker_targets(),
                    *_function_targets(),
                )
            }
        )
    )


__all__ = [
    "GITCODE_HOST",
    "GITEE_HOST",
    "GITHUB_HOST",
    "LINEAR_HOST",
    "build_default_registry",
    "default_target_ids",
]

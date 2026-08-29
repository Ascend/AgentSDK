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

"""Bootstrap the default :class:`TemplateRegistry` from discovery paths.

P85-B: the public entry point for wiring template discovery into the
process-wide registry at CLI bootstrap time. After this module runs,
``template: <id>`` references on :class:`~src.agent.AgentDefinition`
resolve against a populated registry instead of always raising
:class:`TemplateNotFoundError`.

Precedence (highest priority last, last-wins via ``overwrite=True``):

* **built-in** (P85-E: the 5 canonical templates registered first so
  any user / project / managed entry with the same id can shadow them)
* **user** (``$CLAWCODEX_CONFIG_DIR/templates/`` or
  ``~/.clawcodex/templates/``)
* **project** (``.clawcodex/templates/`` walked up to ``.git`` or fs root)
* **managed** (``$CLAWCODEX_MANAGED_CONFIG_DIR/templates/`` or
  ``/etc/clawcodex/templates/``)

This matches the agent-discovery convention in
:mod:`src.utils.markdown_config_loader` and the explicit user
decision documented in the P85-B plan.

The bootstrap is **safe to call multiple times**: the underlying
:meth:`TemplateRegistry.register` deduplicates, and ``overwrite=True``
across sources means a re-run with different cwd still reflects the
latest state.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .built_in import SOURCE_BUILT_IN, register_built_in_templates
from .discovery import (
    get_managed_templates_dir,
    get_project_templates_dirs,
    get_user_templates_dir,
)
from .exceptions import TemplateAlreadyExistsError
from .models import Template
from .registry import TemplateRegistry, get_default_template_registry

logger = logging.getLogger(__name__)

# Source tags — exported so callers / tests can reference the same
# constants. Matches the convention used in
# src.utils.markdown_config_loader (SOURCE_USER / SOURCE_PROJECT /
# SOURCE_MANAGED) for consistency.
SOURCE_USER = "user"
SOURCE_PROJECT = "project"
SOURCE_MANAGED = "managed"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _register_orchestrator_templates(
    registry: TemplateRegistry,
    *,
    overwrite: bool,
) -> int:
    root = _repo_root()
    templates_dir = root / "extensions" / "orchestrator" / "templates"
    specs = (
        (
            "orchestrator-workflow",
            "Orchestrator Workflow",
            "workflow",
            "workflow.md",
            "workflow.template.md",
        ),
        (
            "orchestrator-workflow-local",
            "Orchestrator Local Workflow",
            "workflow",
            "workflow-local.md",
            "workflow-local.template.md",
        ),
        (
            "orchestrator-workflow-yaml",
            "Orchestrator YAML Workflow",
            "workflow",
            "workflow.yaml",
            "workflow.yaml.template",
        ),
        (
            "orchestrator-issue-card",
            "Orchestrator Issue Card",
            "issue",
            ".clawcodex_local_issues/{{ identifier }}.md",
            "issue-card.template.md",
        ),
    )
    added = 0
    for template_id, title, kind, output_path, filename in specs:
        ref = templates_dir / filename
        if not ref.is_file():
            continue
        metadata: dict[str, object] = {
            "kind": kind,
            "category": "orchestrator",
            "tags": ["orchestrator", kind],
            "schema_version": "1",
            "output_path_template": output_path,
        }
        if template_id == "orchestrator-issue-card":
            metadata["variables"] = [
                {
                    "name": "identifier",
                    "description": "Issue identifier used in the generated filename.",
                    "required": True,
                    "pattern": r"^[A-Za-z0-9._-]+$",
                }
            ]
        template = Template(
            id=template_id,
            title=title,
            description=f"Built-in orchestrator {kind} template.",
            fields={"content_template_ref": str(ref)},
            metadata=metadata,
            source=SOURCE_BUILT_IN,
        )
        before = len(registry)
        try:
            registry.register(template, overwrite=overwrite)
        except TemplateAlreadyExistsError:
            continue
        if len(registry) > before:
            added += 1
    return added


def _load_source(
    registry: TemplateRegistry,
    search_dir: Path | None,
    *,
    source: str,
    overwrite: bool,
) -> int:
    """Discover a single source dir into ``registry``. Returns # registered.

    Skips silently when ``search_dir`` is ``None`` or the directory
    does not exist on disk — the absence of a config dir is a normal
    state on a fresh install and must NOT raise.
    """
    if search_dir is None or not search_dir.is_dir():
        logger.debug("template source %s: dir missing, skipping (%s)", source, search_dir)
        return 0
    # Sub-registry per source so we can tag `source` on every discovered
    # Template without coupling discover() to the global default.
    sub = TemplateRegistry(search_dir=search_dir)
    added = sub.discover(source=source, overwrite=True)
    for tpl in sub.list_templates():
        try:
            registry.register(tpl, overwrite=overwrite)
        except TemplateAlreadyExistsError:
            # overwrite=False keeps an earlier copy (whichever one got
            # there first); higher-priority sources cannot displace it.
            logger.debug("template source %s: skipping existing id %r", source, tpl.id)
    logger.debug("template source %s: registered %d from %s", source, added, search_dir)
    return added


def bootstrap_default_templates(
    cwd: Path | None = None,
    *,
    overwrite: bool = True,
) -> int:
    """Populate the default :class:`TemplateRegistry` from standard paths.

    Walks built-in (P85-E) → user → project(s) → managed in
    **low → high** priority order and registers every
    ``*.yml`` / ``*.yaml`` / ``*.json`` file found under each
    ``templates/`` directory.

    Args:
        cwd: Project root to walk up from. ``None`` → :func:`os.getcwd`.
            Tests pass a tmp dir to keep the walker isolated from the
            real working directory.
        overwrite: When ``True`` (default), higher-priority sources
            overwrite lower-priority templates of the same id. Pass
            ``False`` to keep the first-registered copy (useful when
            injecting test fixtures that should not collide with
            on-disk templates).

    Returns:
        Total templates present in the default registry after the
        call. Overwrites are NOT counted as new registrations.

    Notes:
        * Idempotent — calling twice with the same inputs yields the
          same registry contents.
        * Does NOT load ``~/.clawcodex/templates.json`` — that is an
          opt-in single-file state surface via :func:`load_registry`.
        * Does NOT raise when any single source is missing — a fresh
          install with no ``.clawcodex`` directories is a no-op
          (aside from the always-on built-in catalogue).
    """
    registry = get_default_template_registry()

    # P85-E: register the built-in catalogue first. Subsequent sources
    # (user / project / managed) write with ``overwrite=True`` so any
    # user-named template shadowing a built-in wins automatically —
    # no flag is needed to override the defaults.
    register_built_in_templates(registry, overwrite=overwrite)
    _register_orchestrator_templates(registry, overwrite=overwrite)

    user_dir = get_user_templates_dir()
    project_dirs = get_project_templates_dirs(cwd)
    managed_dir = get_managed_templates_dir()

    _load_source(registry, user_dir, source=SOURCE_USER, overwrite=overwrite)
    # Project can yield multiple dirs (one per ancestor); each is its
    # own source-label iteration. The walker returned them
    # outermost-first, so iteration with overwrite=True naturally puts
    # the innermost (most specific) last.
    for project_dir in project_dirs:
        _load_source(registry, project_dir, source=SOURCE_PROJECT, overwrite=overwrite)
    _load_source(registry, managed_dir, source=SOURCE_MANAGED, overwrite=overwrite)

    return len(registry)


__all__ = [
    "SOURCE_BUILT_IN",
    "SOURCE_MANAGED",
    "SOURCE_PROJECT",
    "SOURCE_USER",
    "bootstrap_default_templates",
]

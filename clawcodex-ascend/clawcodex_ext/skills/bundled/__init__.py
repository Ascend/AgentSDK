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

"""Bundled-skill catalogue + init orchestrator.

Mirrors the TS pattern in ``typescript/src/skills/bundled/index.ts``:
each individual skill module exposes a ``register_*_skill()`` function
that calls ``register_bundled_skill(BundledSkillDefinition(...))``;
``init_bundled_skills()`` calls them in order at startup.

``init_bundled_skills`` is idempotent — calling it twice does not
re-register skills (the second call is a no-op). The bundled-skill
registry consults a sentinel set so a fresh ``clear_bundled_skills()``
forces re-init on the next call.
"""

from __future__ import annotations

import logging
import threading

from .batch import register_batch_skill
from .debug import register_debug_skill
from .loop import register_loop_skill
from .orchestrator import register_orchestrator_skill
from .remember import register_remember_skill
from .simplify import register_simplify_skill
from .spec_audit import register_spec_audit_skill
from .stuck import register_stuck_skill
from .update_config import register_update_config_skill
from .verify import register_verify_skill
from .verify_content import register_verify_content_skill

logger = logging.getLogger(__name__)


# Tracks whether ``init_bundled_skills`` has already populated the
# registry. Reset by ``clear_bundled_skills`` (via the hook below) so
# tests that wipe state can re-init cleanly.
_INITIALIZED: bool = False
_INIT_LOCK = threading.RLock()


def init_bundled_skills() -> bool:
    """Register every always-on bundled skill exactly once.

    Calls each ``register_*_skill()`` function in a fixed order. Skills
    with feature gates check ``is_enabled`` lazily at lookup time
    (matches TS) — they're registered unconditionally so they show up
    in the catalogue when the gate flips.

    Idempotent: subsequent calls are no-ops. Use the
    ``clear_bundled_skills()`` hook in ``src.skills.bundled_skills`` to
    reset state in tests.
    """
    global _INITIALIZED
    from ..bundled_skills import _registry_lock

    registrars = (
        register_simplify_skill,
        register_debug_skill,
        register_loop_skill,
        register_batch_skill,
        register_stuck_skill,
        register_verify_content_skill,
        register_verify_skill,
        register_update_config_skill,
        register_remember_skill,
        register_spec_audit_skill,
        register_orchestrator_skill,
    )
    with _INIT_LOCK, _registry_lock:
        if _INITIALIZED:
            return True

        all_ok = True
        for registrar in registrars:
            try:
                outcome = registrar()
            except Exception:
                all_ok = False
                logger.exception("failed to register bundled skill via %s", registrar.__name__)
            else:
                if outcome is False:
                    all_ok = False
                    logger.warning(
                        "bundled skill registrar %s rejected its definition",
                        registrar.__name__,
                    )

        _INITIALIZED = all_ok
        if all_ok:
            logger.debug("bundled skills initialized")
        return all_ok


def reset_bundled_skills_init_flag() -> None:
    """Drop the idempotency flag so the next ``init_bundled_skills``
    call re-runs. Wired to ``clear_bundled_skills`` so test fixtures
    that reset the registry also reset the flag.
    """
    global _INITIALIZED
    _INITIALIZED = False


__all__ = [
    "init_bundled_skills",
    "reset_bundled_skills_init_flag",
    "register_orchestrator_skill",
    "register_simplify_skill",
    "register_spec_audit_skill",
    "register_debug_skill",
    "register_loop_skill",
    "register_batch_skill",
    "register_remember_skill",
    "register_stuck_skill",
    "register_update_config_skill",
    "register_verify_skill",
    "register_verify_content_skill",
]

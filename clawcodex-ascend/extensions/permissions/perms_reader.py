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
"""ClawCodex permissions extensions — structured permissions read path.

Extracted from ``src/permissions/modes.py`` so the upstream mode
resolution stays free of structured-config aggregation concerns.

Architecture::

    src/permissions/modes.py              ← upstream (calls hooks below)
        |
        ↓ import
    extensions/permissions/perms_reader.py ← this module (structured read path)

Two public helpers:

* ``settings_perms_structured_is_explicit()`` — detects whether a
  ``PermissionsConfig`` object carries user-set (non-default) values.
* ``settings_perms()`` — aggregates legacy ``extra["permissions"]`` and
  structured ``PermissionsConfig.to_dict()`` into a single dict, with
  correct precedence.
"""

from __future__ import annotations

import logging

from typing import Any

logger = logging.getLogger(__name__)


def settings_perms_structured_is_explicit(perms_obj: Any) -> bool:
    """True when :class:`PermissionsConfig` carries any user-set value.

    The structured field is always populated (defaults are non-None), so
    we can't use "field is None" to detect "user set it".
    Instead we look for any value that diverges from the dataclass defaults
    -- a non-empty rules bucket, a behavior key not in the default
    3-behavior skeleton, a non-empty additional_directories, a non-empty
    additional, a default_mode, or allow_bypass_permissions_mode is True.

    Note: explicitly setting ``allow_bypass_permissions_mode = False`` is
    indistinguishable from leaving it at the default, so a structured
    ``False`` cannot override a legacy ``extra["permissions"]`` baseline
    that still carries ``allowBypassPermissionsMode: True``. To *disable*
    bypass, remove the ``permissions`` block entirely (not set it to
    ``False``). A full fix requires tracking field presence at
    deserialization time (upstream ``PermissionsConfig``).
    """
    if perms_obj is None:
        return False
    if getattr(perms_obj, "allow_bypass_permissions_mode", None) is True:
        return True
    if getattr(perms_obj, "default_mode", None):
        return True
    # ``rules`` is initialized to a 3-behavior skeleton
    # ``{"allow": [], "deny": [], "ask": []}``. Treat that as default.
    default_rule_keys = {"allow", "deny", "ask"}
    rules = getattr(perms_obj, "rules", None)
    if not isinstance(rules, dict):
        rules = {}
    if any(rules.get(b) for b in default_rule_keys):
        return True
    if set(rules.keys()) - default_rule_keys:
        return True
    if getattr(perms_obj, "additional_directories", None):
        return True
    return bool(getattr(perms_obj, "additional", None))


def settings_perms(settings: Any) -> dict[str, Any]:
    """Aggregate all readable ``permissions`` sub-keys from a settings object.

    Replaces the previous ``settings.extra["permissions"]`` read path,
    which was the only working fallback under the legacy
    ``list[PermissionRule]`` schema but became a dead-end once
    ``permissions`` was promoted to a structured
    :class:`PermissionsConfig` field.

    Semantic:

    * Legacy ``settings.extra["permissions"]`` is the *baseline* (covers
      binaries that wrote the dict into ``extra``).
    * Structured :class:`PermissionsConfig` *overrides* the baseline only
      when it carries an explicit non-default value
      (see :func:`settings_perms_structured_is_explicit`). The structured
      ``to_dict()`` keys then win over the legacy keys for the same field.
    * ``settings.permissions.additional`` is merged last and always wins
      (forward-compat bag for unknown sub-keys written by newer / custom
      config sources).

    Returns an empty dict when ``settings`` is ``None`` or has no usable
    permissions block — callers should treat empty as "no override". To
    distinguish "not configured" from "configured but read failed", pair
    with :func:`settings_perms_structured_is_explicit`.
    """
    bag: dict[str, Any] = {}
    if settings is None:
        return bag

    # 1. Legacy baseline (predates the structured config field).
    legacy = getattr(settings, "extra", None)
    if isinstance(legacy, dict):
        legacy_perms = legacy.get("permissions")
        if isinstance(legacy_perms, dict):
            bag.update(legacy_perms)

    perms_obj = getattr(settings, "permissions", None)
    structured_is_explicit = settings_perms_structured_is_explicit(perms_obj)

    # 2. Structured fields override legacy when explicit.
    if structured_is_explicit:
        to_dict = getattr(perms_obj, "to_dict", None)
        if not callable(to_dict):
            logger.warning(
                "Structured permissions config is explicit but %r has no callable "
                "to_dict(); skipping structured override",
                type(perms_obj).__name__,
            )
        else:
            try:
                rendered = to_dict()
                if not isinstance(rendered, dict):
                    logger.warning(
                        "Structured permissions config is explicit but to_dict() returned "
                        "%s instead of dict; skipping structured override",
                        type(rendered).__name__,
                    )
                else:
                    bag.update(rendered)
            except Exception:  # noqa: BLE001 — user-supplied object, don't guess failure modes
                logger.warning(
                    "Structured permissions config is explicit but to_dict() raised "
                    "an error; skipping structured override",
                    exc_info=True,
                )

    # 3. Forward-compat bag always wins for unknown sub-keys.
    if perms_obj is not None:
        additional = getattr(perms_obj, "additional", None)
        if isinstance(additional, dict):
            bag.update(additional)

    return bag

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

"""Runtime glue that mounts the proactive tick emitter onto a session context.

The mirror image of ``clawcodex_ext/cron_system/runtime.py``:

* :func:`attach_proactive_runtime` wires a single :class:`TickEmitter` to a
  context exactly the way :func:`attach_cron_runtime` wires a
  ``CronScheduler``: it reuses the context's outbox, records itself as
  ``ctx.proactive_emitter``, and (optionally) starts the 30 s tick daemon.
  Once attached, the context owns the emitter — the ``/proactive`` slash
  command reuses the same instance so on/off controls exactly one
  scheduler.
* :func:`drain_proactive_prompts` pops due ``proactive_prompt`` outbox
  entries the way :class:`CronDispatchBridge.drain` pops cron entries, so
  frontends can pick tick / sleep-wake prompts up at their existing
  outbox-drain barriers.

All of this is gated on the KAIROS / PROACTIVE feature flags (both default
off). With the features disabled the attach is a no-op and the producers
never run, so existing behaviour is unchanged.
"""

from __future__ import annotations

from typing import Any, Callable

from clawcodex_ext.query.outbox_types import ProactivePromptEvent  # pylint: disable=no-name-in-module
from clawcodex_ext.services.proactive.controller import get_default_controller
from clawcodex_ext.services.proactive.tick_emitter import TickEmitter

__all__ = [
    "attach_proactive_runtime",
    "drain_proactive_prompts",
    "is_proactive_feature_enabled",
]


def is_proactive_feature_enabled() -> bool:
    """Return whether KAIROS or PROACTIVE is enabled in the feature registry.

    Mirrors the predicate historically defined in
    ``command_system/proactive_command.py`` so the runtime attach / drain
    consumers and the ``/proactive`` command share one gate. Unknown or
    unregistered names resolve to ``False`` and a broken registry is treated
    as disabled — this never raises into startup paths.
    """
    try:
        from clawcodex_ext.feature_gate import get_registry  # pylint: disable=no-name-in-module

        reg = get_registry()
        return reg.is_enabled("PROACTIVE") or reg.is_enabled("KAIROS")
    except Exception:
        return False


def attach_proactive_runtime(
    ctx: Any,
    *,
    autostart: bool = False,
    should_skip: Callable[[], bool] | None = None,
) -> TickEmitter | None:
    """Wire a proactive tick emitter to a session context.

    Returns ``None`` (a no-op) when the KAIROS / PROACTIVE features are
    disabled. ``should_skip`` mirrors cron's ``is_loading`` busy gate: while
    it returns True the emitter drops ticks instead of delivering them, so
    an in-flight agent turn never accumulates a burst of autonomous wake-up
    prompts.

    Idempotent: when the context already carries a ``proactive_emitter`` the
    existing instance is returned unchanged (a context mounts the emitter at
    most once, and ``/proactive`` commands reuse the mounted instance).
    """
    if not is_proactive_feature_enabled():
        return None
    emitter = getattr(ctx, "proactive_emitter", None)
    if emitter is not None:
        return emitter
    outbox = getattr(ctx, "outbox", None)
    if outbox is None:
        outbox = []
        setattr(ctx, "outbox", outbox)
    emitter = TickEmitter(
        controller=get_default_controller(),
        outbox=outbox,
        should_skip=should_skip,
    )
    setattr(ctx, "proactive_emitter", emitter)
    if autostart:
        emitter.start()
    return emitter


def _entry_type(entry: Any) -> str:
    """Return the ``type`` discriminator of an outbox entry.

    Accepts typed :class:`ProactivePromptEvent` and legacy dict-style
    entries. This is a local mirror of ``cron_system.dispatch._entry_type``
    kept here so the proactive drain does not depend on the cron subsystem.
    """
    if isinstance(entry, ProactivePromptEvent):
        return "proactive_prompt"
    if isinstance(entry, dict):
        return entry.get("type", "")
    getter = getattr(entry, "get", None)
    if callable(getter):
        try:
            return getter("type", "")
        except Exception:
            return ""
    return ""


def _entry_field(entry: Any, key: str, default: Any = None) -> Any:
    """Read a field from a typed or dict-style outbox entry.

    Local mirror of ``cron_system.dispatch._entry_field`` (see
    :func:`_entry_type` for why it lives here).
    """
    if isinstance(entry, dict):
        return entry.get(key, default)
    if hasattr(entry, key):
        return getattr(entry, key)
    getter = getattr(entry, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except Exception:
            return default
    return default


def drain_proactive_prompts(outbox: list[Any]) -> list[str]:
    """Pop due ``proactive_prompt`` entries from the outbox.

    Accepts typed :class:`ProactivePromptEvent` and dict-style
    ``{"type": "proactive_prompt", ...}`` entries; every other entry (cron
    events, unknown dicts) is left in place for its own consumer. Entries
    with a blank prompt are left in place too, mirroring the cron bridge's
    drain contract. Returns the raw prompt texts in arrival order.
    """
    prompts: list[str] = []
    remaining: list[Any] = []
    for entry in outbox:
        if _entry_type(entry) == "proactive_prompt":
            prompt = (_entry_field(entry, "prompt", "") or "").strip()
            if prompt:
                prompts.append(prompt)
                continue
            remaining.append(entry)
        else:
            remaining.append(entry)
    outbox[:] = remaining
    return prompts

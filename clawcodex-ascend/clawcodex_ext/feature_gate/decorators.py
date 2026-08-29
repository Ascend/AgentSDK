#!/usr/bin/env python3
# -*- coding: utf-8 -*-


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

"""Decorators for feature-gated code.

Provides ``@feature_gated`` (function/method level) and
``feature_gated_class`` (class level) decorators that conditionally
activate or deactivate targets based on the feature registry state.
"""

from __future__ import annotations

import functools
import importlib
from typing import Any, Callable, TypeVar

T = TypeVar("T", bound=Callable[..., Any])
C = TypeVar("C", bound="type[Any]")


def _get_registry():
    """Return the feature registry after package initialization completes."""
    feature_gate = importlib.import_module("clawcodex_ext.feature_gate")
    return feature_gate.get_registry()


def feature_gated(
    feature_name: str,
    fallback: T | None = None,
) -> Callable[[T], T]:
    """Conditionally enable a function or method based on a feature flag.

    When the named feature is **disabled**, the decorated object is
    replaced by *fallback* (which defaults to a no-op stub).

    Example::

        @feature_gated("experimental_chat")
        def experimental_chat():
            ...

        @feature_gated("experimental_chat", fallback=_legacy_chat)
        def experimental_chat():
            ...

    Note: The guard is evaluated at decoration time (import time).  If
    you need runtime re-evaluation, use ``registry.is_enabled()``
    directly inside the function body.
    """

    def decorator(obj: T) -> T:
        reg = _get_registry()
        if not reg.is_enabled(feature_name):
            if fallback is not None:
                return fallback  # type: ignore[return-value]

            # Default: return a no-op stub so calls don't crash.
            @functools.wraps(obj)  # type: ignore[arg-type]
            def _noop(*_args: Any, **_kwargs: Any) -> None:
                pass

            return _noop  # type: ignore[return-value]
        return obj

    return decorator


def feature_gated_class(
    name: str,
    fallback_cls: type | None = None,
) -> Callable[[C], C]:
    """Conditionally register a class based on a feature flag.

    When the named feature is **enabled**, validates dependencies and
    mutual-exclusion constraints before returning the class.

    When **disabled**, returns *fallback_cls* (or the original class
    unchanged if no fallback is provided).

    Example::

        @feature_gated_class("agentic_mode")
        class AgenticModeAgent:
            ...

    Raises:
        RuntimeError: If the feature is enabled but dependency or mutex
            constraints are violated.
    """

    def wrapper(cls: C) -> C:
        reg = _get_registry()
        if reg.is_enabled(name):
            # Validate constraints when enabling
            missing = reg.check_deps(name)
            if missing:
                raise RuntimeError(f"Feature '{name}' requires but is missing: {missing}")
            conflicts = reg.check_mutex(name)
            if conflicts:
                raise RuntimeError(f"Feature '{name}' conflicts with: {conflicts}")
            return cls
        # Feature disabled -- use fallback or return the class unchanged.
        if fallback_cls is not None:
            return fallback_cls  # type: ignore[return-value]
        return cls

    return wrapper


def guarded_call(feature_name: str, func: T, *args: Any, **kwargs: Any) -> T | None:
    """Call *func* only if *feature_name* is enabled.

    This is a convenience helper for inline guards inside function
    bodies where a full decorator is inconvenient.

    Returns:
        The return value of *func* if enabled, or ``None`` if disabled.
    """
    if _get_registry().is_enabled(feature_name):
        return func(*args, **kwargs)
    return None


def guarded_is_enabled(feature_name: str) -> bool:
    """Quick check whether a feature is enabled.

    Useful for inline conditionals inside code that shouldn't be
    decorated.
    """
    return _get_registry().is_enabled(feature_name)

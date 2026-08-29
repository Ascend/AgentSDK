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

# pylint: disable=unnecessary-ellipsis

"""Registry and lazy-loading infrastructure for native-style modules.

The package provides pure-Python equivalents of native audio capture, image
comparison, URL-scheme registration, and modifier-key detection. Submodules
are imported only when ``load`` is called, so optional dependencies remain
optional. Implementations register themselves with
``NativeModuleRegistry.register`` and may provide a pure-Python ``fallback``.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, ClassVar, Protocol, runtime_checkable

__all__ = [
    "NativeModule",
    "NativeModuleRegistry",
    "NativeModuleError",
    "load",
    "load_or_fallback",
    "available_names",
]

_logger = logging.getLogger("clawcodex_ext.native")


# ---------------------------------------------------------------------------
# Protocol shared across module boundaries
# ---------------------------------------------------------------------------


@runtime_checkable
class NativeModule(Protocol):
    """Structural protocol implemented by every native-style module.

    Attributes:
        name: Stable registry identifier, such as ``"audio_capture"``.
    """

    name: str

    def is_available(self) -> bool:
        """Return whether optional dependencies and runtime support exist."""
        ...

    def get_version(self) -> str:
        """Return the implementation version, or ``"unavailable"``."""
        ...


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class NativeModuleError(RuntimeError):
    """Raised when a native-style module cannot be loaded or invoked."""


class NativeModuleRegistry:
    """Registry for lazily loaded native-style module classes.

    Classes register through ``@NativeModuleRegistry.register("name")``.
    Only class objects are stored, so each ``load`` call creates an independent
    instance.
    """

    # Store class references directly so nested ``__qualname__`` segments do
    # not require fragile reflective lookup.
    _registry: ClassVar[dict[str, type]] = {}

    @classmethod
    def register(cls, name: str) -> Any:
        """Return a class decorator that registers a module implementation.

        Example::

            @NativeModuleRegistry.register("audio_capture")
            class AudioCaptureModule:
                name = "audio_capture"
                ...
        """

        def _decorator(mod_cls: type) -> type:
            cls._registry[name] = mod_cls
            _logger.debug(
                "registered native module %r → %s.%s",
                name,
                mod_cls.__module__,
                mod_cls.__qualname__,
            )
            return mod_cls

        return _decorator

    @classmethod
    def names(cls) -> list[str]:
        return sorted(cls._registry)

    @classmethod
    def is_registered(cls, name: str) -> bool:
        return name in cls._registry

    @classmethod
    def get_class(cls, name: str) -> type:
        """Return a registered class or raise ``NativeModuleError``."""
        if name not in cls._registry:
            raise NativeModuleError(f"unknown native module: {name!r}")
        return cls._registry[name]

    @classmethod
    def _instantiate(cls, name: str) -> NativeModule:
        return cls.get_class(name)()


# ---------------------------------------------------------------------------
# Built-in module registration without eager imports
# ---------------------------------------------------------------------------


def _register_builtin_modules() -> None:
    """Register lazy placeholders for built-in modules.

    This keeps heavy optional dependencies out of package import. The first
    ``load`` or ``load_or_fallback`` imports the target submodule, whose
    registration decorator replaces the placeholder with the real class.
    """
    _builtin_paths = {
        "audio_capture": ("clawcodex_ext.native.audio", "AudioCaptureModule"),
        "image_processor": ("clawcodex_ext.native.image", "ImageProcessorModule"),
        "url_handler": ("clawcodex_ext.native.url_handler", "UrlHandlerModule"),
        "modifiers": ("clawcodex_ext.native.modifiers", "ModifiersModule"),
    }

    class _LazyPlaceholder:
        """Marker base class for entries that still need resolution."""

        def __init__(self, _name: str = "") -> None:
            raise RuntimeError("placeholder not directly instantiable")

    for name, (mod_name, cls_name) in _builtin_paths.items():
        if name not in NativeModuleRegistry._registry:
            placeholder = type(
                f"_Lazy_{name}",
                (_LazyPlaceholder,),
                {"name": name, "_lazy_target": (mod_name, cls_name)},
            )
            NativeModuleRegistry._registry[name] = placeholder


_register_builtin_modules()


# ---------------------------------------------------------------------------
# Public loading API
# ---------------------------------------------------------------------------


def _resolve_real_class(name: str) -> type:
    """Resolve a lazy placeholder by importing and re-reading the registry."""
    cls = NativeModuleRegistry.get_class(name)
    target = getattr(cls, "_lazy_target", None)
    if target is not None:
        mod_name, cls_name = target
        importlib.import_module(mod_name)  # Trigger the submodule's registration.
        cls = NativeModuleRegistry.get_class(name)
    return cls


def load(name: str) -> NativeModule | None:
    """Lazily load an available module instance by name.

    Return ``None`` for unknown, unavailable, or import-failing modules.
    Other exceptions propagate to the caller.
    """
    if not NativeModuleRegistry.is_registered(name):
        return None
    try:
        real_cls = _resolve_real_class(name)
    except ImportError:
        _logger.debug("native module %r import failed — unavailable", name)
        return None
    try:
        instance = real_cls()
    except ImportError:
        _logger.debug("native module %r import failed — unavailable", name)
        return None
    try:
        if instance.is_available():
            return instance
    except ImportError:
        _logger.debug("native module %r is_available() import failed", name)
        return None
    _logger.debug("native module %r not available on this host", name)
    return None


def load_or_fallback(name: str) -> NativeModule:
    """Load a module or return its pure-Python fallback.

    Raise ``NativeModuleError`` when the module is unknown or has no fallback.
    """
    instance = load(name)
    if instance is not None:
        return instance
    if not NativeModuleRegistry.is_registered(name):
        raise NativeModuleError(f"unknown native module: {name!r}")
    try:
        cls_obj = _resolve_real_class(name)
    except ImportError:
        cls_obj = NativeModuleRegistry.get_class(name)
    fallback_factory = getattr(cls_obj, "fallback", None)
    if callable(fallback_factory):
        return fallback_factory()  # type: ignore[return-value]
    raise NativeModuleError(f"native module {name!r} unavailable and has no fallback")


def available_names() -> list[str]:
    """Return every registered module name, whether available or not."""
    return NativeModuleRegistry.names()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright (c) 2026 Clawd Codex Team
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

# pylint: disable=unnecessary-ellipsis

"""F-81: Native 原生模块系统 — 统一注册表与懒加载基础设施.

对标 CCB Rust/NAPI 原生模块（audio-capture-napi / color-diff-napi /
image-processor-napi / url-handler-napi / modifiers-napi），用纯 Python
等价实现音频捕获、图像差异对比、URL Scheme 注册、修饰键检测等能力。

设计要点
--------

* **懒加载** —— 子模块（``audio`` / ``image`` / ``url_handler`` /
  ``modifiers``）不在 ``__init__`` 中导入，仅在 ``load()`` 调用时按需
  实例化。可选依赖（``pyaudio`` / ``Pillow`` / ``numpy`` / ``pynput``）
  缺失时 ``is_available()`` 返回 ``False``，``load()`` 返回 ``None``。
* **降级 (F-81.6)** —— 每个模块实现一个 ``fallback`` 类方法，在可选依赖
  缺失时返回纯 Python 兜底实例，调用方可通过 ``load_or_fallback(name)``
  拿到一个总是可用的对象（功能受限但不抛 ``ImportError``）。
* **注册模式 (Golden Rule #5)** —— 模块通过 ``@NativeModuleRegistry.register``
  装饰器自注册，调用方按名字查找，无需修改本文件即可新增模块。
* **不侵入 ``src/``** —— 整套子系统位于 ``clawcodex_ext/native/``，符合
  Golden Rule #1/#6（增强上游行为 → Layer 1）。

详见 ``docs/feature_plan/06-ccb-benchmark/f-81-native-modules.md``。
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
# Protocol — 层间契约（Golden Rule #4）
# ---------------------------------------------------------------------------


@runtime_checkable
class NativeModule(Protocol):
    """所有原生模块必须满足的结构化协议.

    Attributes:
        name: 模块稳定标识（用于注册表查找，例如 ``"audio_capture"``）.
    """

    name: str

    def is_available(self) -> bool:
        """返回 ``True`` 当且仅当可选依赖已安装且运行时可用."""
        ...

    def get_version(self) -> str:
        """返回底层实现的版本字符串（缺失依赖时返回 ``"unavailable"``）."""
        ...


# ---------------------------------------------------------------------------
# Registry — 注册模式 (Golden Rule #5)
# ---------------------------------------------------------------------------


class NativeModuleError(RuntimeError):
    """原生模块加载或调用失败时抛出."""


class NativeModuleRegistry:
    """统一的原生模块注册表，懒加载 + 降级检查.

    模块类通过 ``@NativeModuleRegistry.register("name")`` 自注册；调用方
    通过 :meth:`load` 按需实例化。注册表只保存类对象，**不**保存实例，
    因此多次 ``load`` 调用得到的是独立实例（线程安全前提：模块自身
    需保证实例化无副作用）。
    """

    # name → 模块类对象（注册装饰器直接持有类引用，避免 qualname 反射
    # 查找的复杂性——嵌套类 ``__qualname__`` 含 ``<locals>`` 等非标识符段，
    # 用 ``getattr`` 逐段取会失败。直接存类对象最简洁稳健）。
    _registry: ClassVar[dict[str, type]] = {}

    @classmethod
    def register(cls, name: str) -> Any:
        """类装饰器：把 ``NativeModule`` 实现类登记到注册表.

        用法::

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
        """返回已注册的类对象（未注册抛 :class:`NativeModuleError`）."""
        if name not in cls._registry:
            raise NativeModuleError(f"unknown native module: {name!r}")
        return cls._registry[name]

    @classmethod
    def _instantiate(cls, name: str) -> NativeModule:
        return cls.get_class(name)()


# ---------------------------------------------------------------------------
# 默认注册 — 内置四个子模块（仅记录路径，不触发 import）
# ---------------------------------------------------------------------------


def _register_builtin_modules() -> None:
    """登记 F-81 内置模块（懒加载占位）.

    为保持懒加载语义（不在 ``__init__`` 导入时拉入 ``pyaudio`` / ``Pillow``
    等重型可选依赖），这里登记一个 *延迟绑定* 的占位类：当且仅当某模块
    名被 :func:`load` / :func:`load_or_fallback` 触发时，才真正 import 对应
    子模块并替换注册表项为真实类对象.

    各子模块在自身文件被首次 import 时会通过 ``@NativeModuleRegistry.register``
    二次登记——这会用真实类对象覆盖占位项，无副作用.
    """
    _builtin_paths = {
        "audio_capture": ("clawcodex_ext.native.audio", "AudioCaptureModule"),
        "image_processor": ("clawcodex_ext.native.image", "ImageProcessorModule"),
        "url_handler": ("clawcodex_ext.native.url_handler", "UrlHandlerModule"),
        "modifiers": ("clawcodex_ext.native.modifiers", "ModifiersModule"),
    }

    class _LazyPlaceholder:
        """占位类：首次实例化时触发子模块 import 并替换注册表项."""

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
# 公共加载 API
# ---------------------------------------------------------------------------


def _resolve_real_class(name: str) -> type:
    """若注册表项是懒占位，触发子模块 import 并替换为真实类对象.

    子模块被 import 后，其 ``@NativeModuleRegistry.register`` 装饰器
    会把真实类写入注册表（覆盖占位）；本函数随后返回最新注册表项.
    """
    cls = NativeModuleRegistry.get_class(name)
    target = getattr(cls, "_lazy_target", None)
    if target is not None:
        mod_name, cls_name = target
        importlib.import_module(mod_name)  # 触发子模块自注册
        cls = NativeModuleRegistry.get_class(name)
    return cls


def load(name: str) -> NativeModule | None:
    """按名字懒加载并返回一个 *可用* 的原生模块实例.

    Returns:
        - 模块未注册 → ``None``
        - 模块 ``is_available()`` 为 ``False`` → ``None``
        - 实例化期间抛 ``ImportError`` → ``None``（吞掉并记日志）
        - 其它异常向上冒泡（调用方应处理 :class:`NativeModuleError`）

    语义对齐 F-81 §1.4 ``NativeModuleRegistry.load``.
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
    """加载原生模块；若不可用则返回其纯 Python fallback 实例 (F-81.6).

    与 :func:`load` 的区别：永远返回一个非 ``None`` 对象。若底层实现类
    提供了 ``fallback()`` 类方法则使用之；否则抛 :class:`NativeModuleError`
    （调用方可据此判断"完全不支持"）。
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
    """返回已注册的全部模块名（不论是否当前可用）."""
    return NativeModuleRegistry.names()

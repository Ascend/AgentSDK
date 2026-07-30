"""F-81.1: 注册表与懒加载基础设施单元测试."""

from __future__ import annotations

import pytest
from clawcodex_ext.native import (
    NativeModule,
    NativeModuleError,
    NativeModuleRegistry,
    available_names,
    load,
    load_or_fallback,
)


def test_builtin_modules_registered():
    """四个内置模块应在 _register_builtin_modules 后全部出现在注册表."""
    names = available_names()
    assert "audio_capture" in names
    assert "image_processor" in names
    assert "url_handler" in names
    assert "modifiers" in names


def test_unknown_module_load_returns_none():
    assert load("does_not_exist") is None


def test_load_or_fallback_unknown_raises():
    with pytest.raises(NativeModuleError):
        load_or_fallback("does_not_exist")


def test_registry_is_registered():
    assert NativeModuleRegistry.is_registered("url_handler")
    assert not NativeModuleRegistry.is_registered("nope")


def test_registry_register_decorator():
    """自定义模块通过装饰器自注册."""

    @NativeModuleRegistry.register("test_dummy_mod")
    class _Dummy:
        name = "test_dummy_mod"

        def is_available(self) -> bool:
            return True

        def get_version(self) -> str:
            return "test-1.0"

    assert NativeModuleRegistry.is_registered("test_dummy_mod")
    inst = load("test_dummy_mod")
    assert inst is not None
    assert inst.get_version() == "test-1.0"
    # 满足 NativeModule Protocol（runtime_checkable）
    assert isinstance(inst, NativeModule)


def test_load_unavailable_returns_none(monkeypatch):
    """is_available() 返回 False 时 load() 返回 None."""

    @NativeModuleRegistry.register("test_unavail_mod")
    class _Unavail:
        name = "test_unavail_mod"

        def is_available(self) -> bool:
            return False

        def get_version(self) -> str:
            return "x"

    assert load("test_unavail_mod") is None


def test_load_swallows_import_error(monkeypatch):
    """实例化期间抛 ImportError 时 load() 返回 None 而非冒泡."""

    @NativeModuleRegistry.register("test_imperr_mod")
    class _ImpErr:
        name = "test_imperr_mod"

        def __init__(self) -> None:
            raise ImportError("simulated missing dep")

        def is_available(self) -> bool:
            return True

        def get_version(self) -> str:
            return "x"

    assert load("test_imperr_mod") is None


def test_load_or_fallback_uses_fallback_when_unavailable():
    """load_or_fallback 在主实现不可用且提供 fallback() 时返回 fallback 实例."""

    @NativeModuleRegistry.register("test_fb_mod")
    class _WithFallback:
        name = "test_fb_mod"

        def is_available(self) -> bool:
            return False

        def get_version(self) -> str:
            return "main"

        @classmethod
        def fallback(cls):
            class _Fb:
                name = "test_fb_mod"

                def is_available(self) -> bool:
                    return False

                def get_version(self) -> str:
                    return "fb-1.0"

            return _Fb()

    inst = load_or_fallback("test_fb_mod")
    assert inst is not None
    assert inst.get_version() == "fb-1.0"


def test_load_or_fallback_no_fallback_raises():
    @NativeModuleRegistry.register("test_nofb_mod")
    class _NoFallback:
        name = "test_nofb_mod"

        def is_available(self) -> bool:
            return False

        def get_version(self) -> str:
            return "main"

    with pytest.raises(NativeModuleError):
        load_or_fallback("test_nofb_mod")

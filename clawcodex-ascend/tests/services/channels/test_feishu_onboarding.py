#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright (c) 2026 Clawd Codex Team
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

"""Feishu QR scan-to-create onboarding tests."""
# pylint: disable=no-name-in-module

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
from typing import Any


def test_feishu_qr_register_uses_sdk_registration_without_bot_probe() -> None:
    from clawcodex_ext.services.channels.feishu_onboarding import qr_register

    calls: dict[str, Any] = {}
    rendered: list[str] = []

    def _register_app(**kwargs: Any) -> dict[str, Any]:
        calls.update(kwargs)
        kwargs["on_qr_code"](
            {
                "url": "https://accounts.feishu.cn/verify?from=sdk&tp=sdk&source=python-sdk%2Fclawcodex",
                "expire_in": 60,
            }
        )
        return {
            "client_id": "cli_app",
            "client_secret": "secret",
            "user_info": {"open_id": "ou_operator", "tenant_brand": "lark"},
        }

    result = qr_register(
        initial_domain="feishu",
        timeout_seconds=30,
        register_app=_register_app,
        render_qr=lambda url: rendered.append(url) or True,
    )

    assert calls["source"] == "clawcodex"
    assert calls["domain"] == "https://accounts.feishu.cn"
    assert calls["lark_domain"] == "https://accounts.larksuite.com"
    assert calls["cancel_event"].is_set() is False
    assert rendered == ["https://accounts.feishu.cn/verify?from=sdk&tp=sdk&source=python-sdk%2Fclawcodex"]
    assert result == {
        "app_id": "cli_app",
        "app_secret": "secret",
        "domain": "lark",
        "open_id": "ou_operator",
    }


def test_feishu_qr_register_returns_none_on_sdk_denied_scan() -> None:
    from clawcodex_ext.services.channels.feishu_onboarding import qr_register

    def _register_app(**_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("access_denied: user denied")

    result = qr_register(
        register_app=_register_app,
        render_qr=lambda _url: True,
    )

    assert result is None


def test_feishu_qr_register_rejects_credentials_without_scanner_open_id() -> None:
    from clawcodex_ext.services.channels.feishu_onboarding import qr_register

    result = qr_register(
        register_app=lambda **_kwargs: {
            "client_id": "cli_app",
            "client_secret": "secret",
        },
        render_qr=lambda _url: True,
    )

    assert result is None


def test_sdk_register_app_avoids_lark_oapi_root_import(monkeypatch, tmp_path) -> None:
    import clawcodex_ext.services.channels.feishu_onboarding as onboarding

    package = tmp_path / "lark_oapi"
    registration = package / "scene" / "registration"
    registration.mkdir(parents=True)
    (package / "__init__.py").write_text(
        'raise AssertionError("root package should not be imported")\n',
        encoding="utf-8",
    )
    (package / "scene" / "__init__.py").write_text("", encoding="utf-8")
    (registration / "errors.py").write_text(
        "class RegisterAppError(Exception):\n    pass\n",
        encoding="utf-8",
    )
    (registration / "__init__.py").write_text(
        "from .errors import RegisterAppError\n\n"
        "def register_app(**kwargs):\n"
        '    kwargs["on_qr_code"]({"url": "https://qr.example"})\n'
        '    return {"client_id": "cli_fast", "client_secret": "secret"}\n',
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    spec = importlib.machinery.ModuleSpec("lark_oapi", loader=None, is_package=True)
    spec.origin = str(package / "__init__.py")
    spec.submodule_search_locations = [str(package)]
    original_find_spec = importlib.util.find_spec
    previous_modules = {name: module for name, module in sys.modules.items() if name.startswith("lark_oapi")}
    for name in list(previous_modules):
        monkeypatch.delitem(sys.modules, name, raising=False)

    def _find_spec(name: str, package_name: str | None = None) -> Any:
        if name == "lark_oapi":
            return spec
        return original_find_spec(name, package_name)

    monkeypatch.setattr(importlib.util, "find_spec", _find_spec)
    qr: list[dict[str, Any]] = []

    try:
        result = onboarding._sdk_register_app(on_qr_code=qr.append)
    finally:
        for name in [name for name in sys.modules if name.startswith("lark_oapi")]:
            sys.modules.pop(name, None)
        sys.modules.update(previous_modules)

    assert qr == [{"url": "https://qr.example"}]
    assert result == {"client_id": "cli_fast", "client_secret": "secret"}


def test_sdk_register_app_retries_transient_ssl_error_inside_registration(monkeypatch, tmp_path) -> None:
    import clawcodex_ext.services.channels.feishu_onboarding as onboarding

    package = tmp_path / "lark_oapi"
    registration = package / "scene" / "registration"
    registration.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "scene" / "__init__.py").write_text("", encoding="utf-8")
    (registration / "errors.py").write_text(
        "class RegisterAppError(Exception):\n    pass\n",
        encoding="utf-8",
    )
    (registration / "__init__.py").write_text(
        "class SSLError(Exception):\n    pass\n"
        "class Exceptions:\n"
        "    SSLError = SSLError\n"
        "    ConnectionError = SSLError\n"
        "    Timeout = SSLError\n"
        "class Requests:\n"
        "    exceptions = Exceptions\n"
        "    def __init__(self):\n"
        "        self.calls = 0\n"
        "    def post(self, *args, **kwargs):\n"
        "        self.calls += 1\n"
        "        if self.calls == 1:\n"
        '            raise self.exceptions.SSLError("eof")\n'
        '        return {"ok": True, "timeout": kwargs.get("timeout")}\n'
        "requests = Requests()\n\n"
        "def register_app(**kwargs):\n"
        '    kwargs["on_qr_code"]({"url": "https://qr.example"})\n'
        '    response = requests.post("https://accounts.feishu.cn/oauth/v1/app/registration")\n'
        '    return {"client_id": "cli_fast", "client_secret": "secret", "response": response}\n',
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    spec = importlib.machinery.ModuleSpec("lark_oapi", loader=None, is_package=True)
    spec.origin = str(package / "__init__.py")
    spec.submodule_search_locations = [str(package)]
    original_find_spec = importlib.util.find_spec
    previous_modules = {name: module for name, module in sys.modules.items() if name.startswith("lark_oapi")}
    for name in list(previous_modules):
        monkeypatch.delitem(sys.modules, name, raising=False)

    def _find_spec(name: str, package_name: str | None = None) -> Any:
        if name == "lark_oapi":
            return spec
        return original_find_spec(name, package_name)

    monkeypatch.setattr(importlib.util, "find_spec", _find_spec)

    try:
        result = onboarding._sdk_register_app(on_qr_code=lambda _info: None)
        registration_module = sys.modules["lark_oapi.scene.registration"]
    finally:
        for name in [name for name in sys.modules if name.startswith("lark_oapi")]:
            sys.modules.pop(name, None)
        sys.modules.update(previous_modules)

    assert result["response"] == {"ok": True, "timeout": 30}
    assert registration_module.requests.calls == 2


def test_feishu_qr_register_uses_sdk_status_for_domain_switch() -> None:
    from clawcodex_ext.services.channels.feishu_onboarding import qr_register

    def _register_app(**kwargs: Any) -> dict[str, Any]:
        kwargs["on_status_change"]({"status": "domain_switched"})
        return {
            "client_id": "cli_lark",
            "client_secret": "secret",
            "user_info": {"open_id": "ou_operator"},
        }

    result = qr_register(
        register_app=_register_app,
        render_qr=lambda _url: True,
    )

    assert result["domain"] == "lark"

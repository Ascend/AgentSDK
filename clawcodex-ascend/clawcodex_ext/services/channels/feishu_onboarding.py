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

"""Feishu / Lark QR scan-to-create onboarding."""

from __future__ import annotations

import importlib.machinery
import importlib.util
import logging
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

logger = logging.getLogger(__name__)

_ACCOUNTS_BASE_URLS = {
    "feishu": "https://accounts.feishu.cn",
    "lark": "https://accounts.larksuite.com",
}
_REGISTRATION_MODULE = "lark_oapi.scene.registration"
_REGISTRATION_REQUEST_TIMEOUT_SECONDS = 30
_REGISTRATION_POST_ATTEMPTS = 3
_REGISTRATION_RETRY_DELAY_SECONDS = 0.5

RegisterAppFn = Callable[..., dict[str, Any]]
RenderQrFn = Callable[[str], bool]


def qr_register(
    *,
    initial_domain: str = "feishu",
    timeout_seconds: int = 600,
    register_app: RegisterAppFn | None = None,
    render_qr: RenderQrFn | None = None,
) -> dict[str, Any] | None:
    """Run SDK-backed Feishu/Lark QR scan-to-create registration.

    Returns app credentials on success, or ``None`` for expected user/network
    failures such as denied scan, expired token, timeout, or registration errors.
    """
    domain = _normalize_domain(initial_domain)
    current_domain = {"value": domain}
    register = register_app or _sdk_register_app
    render = render_qr or _render_qr
    cancel_event = threading.Event()
    timer = _start_cancel_timer(cancel_event, timeout_seconds)

    def _on_qr_code(info: dict[str, Any]) -> None:
        qr_url = str(_mapping(info).get("url") or "")
        if not qr_url:
            return
        if not render(qr_url):
            print(f"扫码链接：{qr_url}")
            print(
                "提示：安装 qrcode 可在终端直接显示二维码："
                "`uv sync --locked --extra feishu`；开发环境 `--extra dev` 已包含。"
            )

    def _on_status_change(info: dict[str, Any]) -> None:
        status = str(_mapping(info).get("status") or "")
        if status == "domain_switched":
            current_domain["value"] = "lark"
        if status:
            logger.debug("feishu onboarding status: %s", status)

    try:
        registration = register(
            on_qr_code=_on_qr_code,
            on_status_change=_on_status_change,
            source="clawcodex",
            cancel_event=cancel_event,
            domain=_accounts_base_url(domain),
            lark_domain=_accounts_base_url("lark"),
        )
        result = _registration_result(registration, current_domain["value"])
        print("扫码完成，已获取 Feishu 应用凭证。")
        return result
    except Exception as exc:  # noqa: BLE001
        if _is_expected_registration_error(exc):
            logger.warning("feishu onboarding failed: %s", exc)
            return None
        raise
    finally:
        if timer is not None:
            timer.cancel()


def _sdk_register_app(**kwargs: Any) -> dict[str, Any]:
    registration = _load_registration_module()
    register_app = getattr(registration, "register_app")

    return _call_register_app_with_resilient_post(registration, register_app, kwargs)


def _call_register_app_with_resilient_post(
    registration: ModuleType,
    register_app: RegisterAppFn,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    requests_module = getattr(registration, "requests", None)
    post = getattr(requests_module, "post", None)
    if requests_module is None or not callable(post):
        return register_app(**kwargs)

    transient_errors = _registration_transient_errors(requests_module)

    def _post_with_retry(*args: Any, **post_kwargs: Any) -> Any:
        post_kwargs.setdefault("timeout", _REGISTRATION_REQUEST_TIMEOUT_SECONDS)
        if not transient_errors:
            return post(*args, **post_kwargs)
        for attempt in range(1, _REGISTRATION_POST_ATTEMPTS + 1):
            try:
                return post(*args, **post_kwargs)
            except transient_errors as exc:  # pylint: disable=catching-non-exception
                if attempt >= _REGISTRATION_POST_ATTEMPTS:
                    raise
                logger.warning(
                    "feishu onboarding registration POST failed transiently (%s/%s), retrying: %s",
                    attempt,
                    _REGISTRATION_POST_ATTEMPTS,
                    exc,
                )
                time.sleep(_REGISTRATION_RETRY_DELAY_SECONDS * attempt)
        raise RuntimeError("unreachable registration retry state")

    requests_module.post = _post_with_retry
    try:
        return register_app(**kwargs)
    finally:
        requests_module.post = post


def _registration_transient_errors(requests_module: Any) -> tuple[type[BaseException], ...]:
    exceptions = getattr(requests_module, "exceptions", None)
    candidates = (
        getattr(exceptions, "SSLError", None),
        getattr(exceptions, "ConnectionError", None),
        getattr(exceptions, "Timeout", None),
    )
    return tuple(error for error in candidates if isinstance(error, type))


def _load_registration_module() -> ModuleType:
    module = sys.modules.get(_REGISTRATION_MODULE)
    if module is not None and hasattr(module, "register_app"):
        return module

    package_dir = _find_lark_oapi_package_dir()
    _ensure_package_module("lark_oapi", [str(package_dir)], package_dir / "__init__.py")
    scene_dir = package_dir / "scene"
    _ensure_package_module("lark_oapi.scene", [str(scene_dir)], scene_dir / "__init__.py")
    registration_dir = scene_dir / "registration"
    errors_name = f"{_REGISTRATION_MODULE}.errors"
    if errors_name not in sys.modules:
        _load_module(errors_name, registration_dir / "errors.py")
    return _load_module(
        _REGISTRATION_MODULE,
        registration_dir / "__init__.py",
        submodule_search_locations=[str(registration_dir)],
    )


def _find_lark_oapi_package_dir() -> Path:
    spec = importlib.util.find_spec("lark_oapi")
    locations = list(spec.submodule_search_locations or []) if spec is not None else []
    if not locations:
        raise ImportError("lark_oapi package is not installed")
    return Path(locations[0])


def _ensure_package_module(name: str, paths: list[str], origin: Path) -> None:
    if name in sys.modules:
        return
    module = ModuleType(name)
    spec = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
    spec.origin = str(origin)
    spec.submodule_search_locations = paths
    module.__file__ = str(origin)
    module.__path__ = paths
    module.__package__ = name
    module.__spec__ = spec
    sys.modules[name] = module


def _load_module(
    name: str,
    path: Path,
    *,
    submodule_search_locations: list[str] | None = None,
) -> ModuleType:
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(
        name,
        path,
        submodule_search_locations=submodule_search_locations,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _registration_result(registration: dict[str, Any], domain: str) -> dict[str, Any]:
    payload = _mapping(registration)
    user_info = _mapping(payload.get("user_info"))
    result_domain = _normalize_domain(domain)
    tenant_brand = str(user_info.get("tenant_brand") or "").lower()
    if tenant_brand == "lark":
        result_domain = "lark"
    app_id = str(payload.get("client_id") or payload.get("app_id") or "")
    app_secret = str(payload.get("client_secret") or payload.get("app_secret") or "")
    if not app_id or not app_secret:
        raise ValueError("Feishu registration did not return app credentials")
    open_id = str(user_info.get("open_id") or "").strip()
    if not open_id:
        # The pinned SDK requests ``request_user_info=open_id`` during the
        # registration begin step.  Treat a response without it as an
        # incomplete registration: credentials alone cannot support the
        # promised first proactive delivery before any inbound message.
        raise ValueError("Feishu registration did not return scanning user open_id")
    return {
        "app_id": app_id,
        "app_secret": app_secret,
        "domain": result_domain,
        "open_id": open_id,
    }


def _start_cancel_timer(cancel_event: threading.Event, timeout_seconds: int) -> threading.Timer | None:
    if timeout_seconds <= 0:
        cancel_event.set()
        return None
    timer = threading.Timer(timeout_seconds, cancel_event.set)
    timer.daemon = True
    timer.start()
    return timer


def _render_qr(url: str) -> bool:
    try:
        import qrcode
    except (ImportError, TypeError):
        return False
    try:
        qr = qrcode.QRCode()
        qr.add_data(url)
        qr.make(fit=True)
        qr.print_ascii(invert=True)
        print(f"\n扫码后继续，或打开链接：{url}")
        return True
    except Exception:  # noqa: BLE001
        logger.debug("failed to render feishu onboarding QR", exc_info=True)
        return False


def _accounts_base_url(domain: str) -> str:
    return _ACCOUNTS_BASE_URLS.get(_normalize_domain(domain), _ACCOUNTS_BASE_URLS["feishu"])


def _normalize_domain(domain: str) -> str:
    normalized = (domain or "feishu").strip().lower()
    return normalized if normalized in {"feishu", "lark"} else "feishu"


def _is_expected_registration_error(exc: BaseException) -> bool:
    if isinstance(exc, (RuntimeError, OSError, TimeoutError, ValueError)):
        return True
    module = sys.modules.get(_REGISTRATION_MODULE)
    error_cls = getattr(module, "RegisterAppError", None)
    return isinstance(exc, error_cls) if isinstance(error_cls, type) else False


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


__all__ = ["qr_register"]

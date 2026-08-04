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

"""F-65 P65-A — Langfuse SDK client loader.

The ``langfuse`` Python SDK is an **optional** dependency. This
module wraps the import + global singleton pattern so the rest of
the codebase can call :func:`get_langfuse_client` without try /
except boilerplate, and so tests can monkeypatch the singleton via
:func:`reset_langfuse_client`.

Behaviour matrix
----------------
``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY`` set + SDK importable
    → returns a real ``langfuse.Langfuse`` client.
Either env-var missing
    → returns ``None`` and logs a single warning. The sink then
      degrades to no-op.
SDK not importable
    → returns ``None`` silently (warning at first call only). This
      is the case in dev environments / minimal installs where the
      optional dep was not pulled in.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


# --- Config ------------------------------------------------------------------


@dataclass(frozen=True)
class LangfuseConfig:
    """Resolved Langfuse configuration.

    All fields are optional; the sink degrades to no-op when the
    credentials are missing. ``host`` defaults to Langfuse Cloud
    when not supplied.
    """

    public_key: str | None
    secret_key: str | None
    host: str = "https://cloud.langfuse.com"

    @property
    def is_configured(self) -> bool:
        return bool(self.public_key) and bool(self.secret_key)

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> LangfuseConfig:
        """Read the three env vars. Honors an injected mapping for tests."""
        src = env if env is not None else os.environ
        return cls(
            public_key=src.get("LANGFUSE_PUBLIC_KEY") or None,
            secret_key=src.get("LANGFUSE_SECRET_KEY") or None,
            host=src.get("LANGFUSE_HOST") or "https://cloud.langfuse.com",
        )


# --- Client singleton --------------------------------------------------------


_client_lock = threading.RLock()
_client: Any | None = None
_client_config: LangfuseConfig | None = None
_warned_missing_dep = False
_warned_missing_creds = False


def _try_import_sdk() -> Any:
    """Return the ``langfuse`` module or ``None`` if not installed."""
    global _warned_missing_dep
    try:
        import langfuse  # type: ignore[import-not-found]
    except ImportError:
        if not _warned_missing_dep:
            logger.info(
                "langfuse SDK not installed; LangfuseSink will degrade to no-op. "
                "Install with `pip install langfuse` to enable observability."
            )
            _warned_missing_dep = True
        return None
    return langfuse


def init_langfuse(
    public_key: str | None = None,
    secret_key: str | None = None,
    host: str | None = None,
    *,
    env: dict[str, str] | None = None,
) -> Any:
    """Initialise the global Langfuse client and return it (or ``None``).

    ``public_key`` / ``secret_key`` / ``host`` win over env-var lookup
    when supplied. Pass ``env=...`` to inject a custom mapping (tests).

    The function is idempotent: a second call returns the existing
    singleton unless its config differs from the request, in which
    case a warning is logged and the existing client is returned
    anyway. Use :func:`reset_langfuse_client` to force a rebuild.
    """
    global _client, _client_config
    cfg = LangfuseConfig.from_env(env=env)
    # Override with explicit args.
    cfg = LangfuseConfig(
        public_key=public_key if public_key is not None else cfg.public_key,
        secret_key=secret_key if secret_key is not None else cfg.secret_key,
        host=host if host is not None else cfg.host,
    )

    with _client_lock:
        if _client is not None and _client_config == cfg:
            return _client
        if _client is not None and _client_config != cfg:
            logger.warning(
                "init_langfuse called with new config; returning existing client. "
                "Call reset_langfuse_client() to force a rebuild."
            )
            return _client

        if not cfg.is_configured:
            global _warned_missing_creds
            if not _warned_missing_creds:
                logger.info("LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set; LangfuseSink will degrade to no-op.")
                _warned_missing_creds = True
            _client = None
            _client_config = cfg
            return None

        sdk = _try_import_sdk()
        if sdk is None:
            _client = None
            _client_config = cfg
            return None

        _client = sdk.Langfuse(
            public_key=cfg.public_key,
            secret_key=cfg.secret_key,
            host=cfg.host,
        )
        _client_config = cfg
        return _client


def get_langfuse_client() -> Any:
    """Return the active Langfuse client, initialising on first call."""
    return _client if _client is not None else init_langfuse()


def is_langfuse_available() -> bool:
    """True iff the SDK is importable AND credentials are configured."""
    return get_langfuse_client() is not None


def reset_langfuse_client() -> None:
    """Drop the cached client + config. Test-only escape hatch."""
    global _client, _client_config
    with _client_lock:
        _client = None
        _client_config = None

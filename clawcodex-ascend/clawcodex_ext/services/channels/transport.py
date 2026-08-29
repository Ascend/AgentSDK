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

"""HTTP transport and webhook URL safety helpers for Channels.

The transport is intentionally minimal and dependency-free so the Channels
package can be used in restricted environments and in unit tests without
pulling in ``httpx`` / ``aiohttp``. The default implementation runs
``urllib.request.urlopen`` in a thread so the asyncio event loop is never
blocked. Tests can inject a fake transport that records calls.
"""

from __future__ import annotations

import asyncio
import ipaddress
import json
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .exceptions import InvalidWebhookURLError, TransportError


DEFAULT_TIMEOUT_SECONDS = 10.0

_ALLOWED_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def validate_webhook_url(
    url: str,
    *,
    allow_http: bool = False,
    allow_loopback: bool = False,
    allow_private: bool = False,
    resolve_host: bool = True,
) -> str:
    """Validate a webhook URL.

    Rules:
      * Scheme must be ``https`` unless ``allow_http`` is set.
      * Hostname must not be empty.
      * If the hostname resolves to a private / loopback / link-local IP
        (and ``allow_loopback`` is not set), the URL is rejected to prevent
        SSRF to internal services.
      * A literal IP address is also checked against the same ranges.
      * Loopback hosts (``localhost``, ``127.0.0.1``, ``::1``) are only
        allowed when ``allow_loopback`` is set.

    Returns the URL unchanged on success. Raises :class:`InvalidWebhookURLError`
    on any violation.
    """
    if not isinstance(url, str) or not url:
        raise InvalidWebhookURLError("webhook url must be a non-empty string")
    parsed = urllib.parse.urlparse(url)
    scheme = (parsed.scheme or "").lower()
    if scheme == "https":
        pass
    elif scheme == "http" and allow_http:
        pass
    else:
        raise InvalidWebhookURLError(f"webhook url must use https (got scheme {scheme!r})")
    if not parsed.hostname:
        raise InvalidWebhookURLError("webhook url must include a hostname")

    host = parsed.hostname
    if host.lower() in _ALLOWED_LOOPBACK_HOSTS and not allow_loopback:
        raise InvalidWebhookURLError(f"webhook url loopback host {host!r} is not allowed by default")

    # When the host is itself a literal IP address, validate it directly so
    # callers can't smuggle private/loopback addresses past the check by
    # setting ``resolve_host=False``. If parsing fails, we treat it as a
    # hostname and rely on the resolver path below.
    try:
        literal_ip = ipaddress.ip_address(host)
    except ValueError:
        literal_ip = None
    if literal_ip is not None:
        if literal_ip.is_loopback and not allow_loopback:
            raise InvalidWebhookURLError(f"webhook url host {host!r} is a loopback address")
        if literal_ip.is_private and not literal_ip.is_loopback and not allow_private:
            raise InvalidWebhookURLError(f"webhook url host {host!r} is a private address")
        if literal_ip.is_link_local:
            raise InvalidWebhookURLError(f"webhook url host {host!r} is a link-local address")
        if literal_ip.is_multicast or literal_ip.is_reserved or literal_ip.is_unspecified:
            raise InvalidWebhookURLError(f"webhook url host {host!r} is a reserved address")

    if resolve_host:
        try:
            infos = socket.getaddrinfo(host, parsed.port or 443, type=socket.SOCK_STREAM)
        except socket.gaierror as exc:
            raise InvalidWebhookURLError(f"webhook url host {host!r} could not be resolved: {exc}") from exc
        for info in infos:
            sockaddr = info[4]
            ip_str = sockaddr[0]
            try:
                ip = ipaddress.ip_address(ip_str)
            except ValueError:
                continue
            if ip.is_loopback and not allow_loopback:
                raise InvalidWebhookURLError(f"webhook url resolves to loopback {ip_str}")
            if ip.is_private and not ip.is_loopback and not allow_private:
                raise InvalidWebhookURLError(f"webhook url resolves to private address {ip_str}")
            if ip.is_link_local:
                raise InvalidWebhookURLError(f"webhook url resolves to link-local address {ip_str}")
            if ip.is_multicast or ip.is_reserved or ip.is_unspecified:
                raise InvalidWebhookURLError(f"webhook url resolves to reserved address {ip_str}")
    return url


_TOKEN_RE = re.compile(r"/(?P<token>[A-Za-z0-9_\-]{6,})/?$")


def redact_webhook_url(url: str) -> str:
    """Return a redacted copy of a webhook URL safe for logging.

    The last path segment is replaced with ``***`` when it looks like a
    secret token. Query parameters are always stripped. The scheme,
    hostname, and path prefix are preserved.
    """
    if not isinstance(url, str) or not url:
        return ""
    parsed = urllib.parse.urlparse(url)
    parts = parsed.path.rsplit("/", 1)
    if len(parts) == 2 and _TOKEN_RE.search(parsed.path):
        redacted_path = parts[0] + "/***"
    else:
        redacted_path = parsed.path
    host = parsed.hostname or ""
    if ":" in host:
        host = f"[{host}]"
    netloc = host
    if parsed.port is not None:
        netloc += f":{parsed.port}"
    return urllib.parse.urlunparse(
        (
            parsed.scheme,
            netloc,
            redacted_path,
            "",
            "",
            "",
        )
    )


@dataclass
class TransportResponse:
    status: int
    body: bytes = b""
    headers: Mapping[str, str] = field(default_factory=dict)


class ChannelTransport(ABC):
    """Async HTTP transport used by channel implementations."""

    async def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> TransportResponse:
        raise TransportError("transport GET is not implemented")

    @abstractmethod
    async def post(
        self,
        url: str,
        body: bytes,
        *,
        headers: Mapping[str, str] | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> TransportResponse: ...


class UrllibChannelTransport(ChannelTransport):
    """Default transport backed by :mod:`urllib.request` running in a thread."""

    def __init__(self, *, default_timeout: float = DEFAULT_TIMEOUT_SECONDS) -> None:
        self._default_timeout = default_timeout

    async def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> TransportResponse:
        if timeout is None:
            timeout = self._default_timeout

        return await asyncio.to_thread(
            self._send_request,
            url,
            method="GET",
            body=None,
            headers=headers,
            timeout=timeout,
        )

    async def post(
        self,
        url: str,
        body: bytes,
        *,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> TransportResponse:
        if timeout is None:
            timeout = self._default_timeout

        return await asyncio.to_thread(
            self._send_request,
            url,
            method="POST",
            body=body,
            headers=headers,
            timeout=timeout,
        )

    @staticmethod
    def _send_request(
        url: str,
        *,
        method: str,
        body: bytes | None,
        headers: Mapping[str, str] | None,
        timeout: float,
    ) -> TransportResponse:
        request = urllib.request.Request(
            url=url,
            data=body,
            method=method,
            headers=dict(headers or {}),
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as resp:
                return TransportResponse(
                    status=resp.status,
                    body=resp.read(),
                    headers=dict(resp.headers.items()),
                )
        except urllib.error.HTTPError as exc:
            return TransportResponse(
                status=exc.code,
                body=exc.read() if hasattr(exc, "read") else b"",
                headers=dict(exc.headers.items()) if exc.headers else {},
            )
        except urllib.error.URLError as exc:
            raise TransportError(f"transport error: {exc.reason}") from exc
        except (TimeoutError, socket.timeout) as exc:
            raise TransportError(f"transport timeout: {exc}") from exc
        except OSError as exc:
            raise TransportError(f"transport os error: {exc}") from exc


def encode_json_body(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def decode_json_body(
    body: bytes | str,
    *,
    default: Any = None,
    raise_on_error: bool = False,
) -> Any:
    """Decode JSON from bytes or string.

    Args:
        body: JSON data as bytes or string.
        default: Value to return on decode error (used when raise_on_error=False).
        raise_on_error: If True, raise TransportError on decode error.

    Returns:
        Parsed JSON data, or default value on error.

    Raises:
        TransportError: If decode fails and raise_on_error=True.
    """
    try:
        if isinstance(body, bytes):
            return json.loads(body.decode("utf-8"))
        return json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        if raise_on_error:
            raise TransportError(f"JSON decode failed: {exc}") from exc
        return default


def default_headers(content_type: str = "application/json") -> dict[str, str]:
    return {"Content-Type": content_type, "User-Agent": "clawcodex-channels/0.1"}


__all__ = [
    "DEFAULT_TIMEOUT_SECONDS",
    "ChannelTransport",
    "TransportResponse",
    "UrllibChannelTransport",
    "decode_json_body",
    "default_headers",
    "encode_json_body",
    "redact_webhook_url",
    "validate_webhook_url",
]

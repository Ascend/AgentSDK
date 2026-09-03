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

"""Token storage for OpenAI Codex ChatGPT OAuth.

At-rest encryption via AES-256-GCM, keyed from machine identity, with
in-memory zeroization on logout.
"""

from __future__ import annotations

import base64
import gc
import hashlib
import json
import os
import socket
import subprocess  # nosec: only called for macOS machine-id, not user input
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CODEX_PROVIDER_ID = "openai-codex"
AUTH_FILE = Path.home() / ".clawcodex" / "auth.json"
CODEX_CLI_AUTH_FILE = Path.home() / ".codex" / "auth.json"

# Encrypted-file magic header (5 bytes + newline = 6 bytes).
AUTH_MAGIC = b"CLXC1\n"

# Fixed domain salt for key derivation — changing this invalidates all
# existing encrypted files (users must re-authenticate).
_KEY_SALT = b"clawcodex-auth-v1"

# AES-256-GCM parameter sizes (in bytes).
_NONCE_SIZE = 12
_TAG_SIZE = 16

# Size of the scrub buffer used for zeroization (4 KiB typical page).
_SCRUB_CHUNK = 4096

# Module-level mutable buffer for intermediate plaintext — overwritten
# with zeros after every read so a core-dump / memory-scan after parse
# won't find the raw decrypted bytes.  Reused across calls to avoid
# allocation churn.
_plaintext_buf: bytearray | None = None

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CodexOAuthTokens:
    access_token: str
    refresh_token: str
    expires_at: float | None = None
    token_type: str = "Bearer"
    scope: str | None = None

    @property
    def is_expired(self) -> bool:
        return self.expires_at is not None and self.expires_at <= time.time()

    def is_expiring(self, skew_seconds: int = 120) -> bool:
        return self.expires_at is not None and self.expires_at <= time.time() + skew_seconds


@dataclass
class CodexAuthRecord:
    tokens: CodexOAuthTokens
    auth_mode: str = "chatgpt"
    last_refresh: float | None = None
    source: str = "clawcodex"


# ---------------------------------------------------------------------------
# Key derivation — machine-bound so the file is protected at rest but
# auto-decryptable on the same host.
# ---------------------------------------------------------------------------


def _derive_key() -> bytes:
    """Derive an AES-256 key from the local machine identity.

    Sources (first available wins):
      1. ``/etc/machine-id`` (Linux)
      2. ``ioreg IOPlatformUUID`` (macOS)
      3. ``socket.gethostname() + $HOME`` (cross-platform fallback)

    The result is SHA-256(input + fixed domain salt), yielding 32 raw bytes
    suitable for AES-256.
    """
    raw = _get_machine_id_bytes()
    return hashlib.sha256(raw + _KEY_SALT).digest()


def _get_machine_id_bytes() -> bytes:
    """Return a stable machine-identity byte string."""
    # 1. Linux: /etc/machine-id (128-bit hex, no trailing newline).
    mid = Path("/etc/machine-id")
    if mid.is_file():
        try:
            return mid.read_bytes().strip()
        except OSError:
            pass  # This optional system source is unavailable; try the next source.

    # 2. macOS: IOPlatformUUID from I/O Registry.
    try:
        result = subprocess.run(  # nosec B607
            ["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"],
            capture_output=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            import re

            m = re.search(rb'"IOPlatformUUID" = "([^"]+)"', result.stdout)
            if m:
                return m.group(1)
    except Exception:  # nosec B110
        pass  # This machine identifier source is unavailable; try the next platform source.

    # 3. Fallback: hostname + home directory.
    host = socket.gethostname().encode("utf-8", errors="replace")
    home = str(Path.home()).encode("utf-8", errors="replace")
    return host + b"|" + home


# ---------------------------------------------------------------------------
# AES-256-GCM encrypt / decrypt
# ---------------------------------------------------------------------------


def _encrypt(data: dict[str, Any]) -> bytes:
    """Serialize *data* as JSON and encrypt with AES-256-GCM.

    Returns
    -------
        ``AUTH_MAGIC`` + 12-byte nonce + ciphertext (includes 16-byte tag)
    """
    key = _derive_key()
    aesgcm = AESGCM(key)
    nonce = os.urandom(_NONCE_SIZE)
    plaintext = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
    # AESGCM.encrypt returns ciphertext + tag (nonce is NOT included).
    ct_with_tag = aesgcm.encrypt(nonce, plaintext, None)
    return AUTH_MAGIC + nonce + ct_with_tag


def _decrypt(raw: bytes) -> dict[str, Any] | None:
    """Decrypt an AES-256-GCM payload written by ``_encrypt``.

    Returns ``None`` when *raw* is not in encrypted format (legacy plain
    JSON), or when decryption fails (wrong key, corruption).
    """
    if not raw.startswith(AUTH_MAGIC):
        return None  # Legacy plain JSON — caller falls back.

    key = _derive_key()
    aesgcm = AESGCM(key)
    payload = raw[len(AUTH_MAGIC) :]
    nonce = payload[:_NONCE_SIZE]
    ct_with_tag = payload[_NONCE_SIZE:]
    try:
        plaintext = aesgcm.decrypt(nonce, ct_with_tag, None)
    except Exception:
        # Wrong key (machine-id changed) or corruption — treat as empty.
        return None
    return json.loads(plaintext)


# ---------------------------------------------------------------------------
# Zeroization helpers
# ---------------------------------------------------------------------------


def _zero_bytearray(buf: bytearray | None) -> None:
    """Overwrite *buf* with zeros and release it."""
    if buf is None:
        return
    view = memoryview(buf)
    n = len(buf)
    written = 0
    while written < n:
        chunk = min(_SCRUB_CHUNK, n - written)
        view[written : written + chunk] = b"\x00" * chunk
        written += chunk
    view.release()


def _scrub_path(path: Path, passes: int = 3) -> None:
    """Overwrite *path* with random data *passes* times, then unlink.

    Only works on POSIX; on Windows the file is simply unlinked.
    """
    if not path.exists():
        return
    if os.name == "nt":
        path.unlink(missing_ok=True)
        return
    size = path.stat().st_size
    for _ in range(passes):
        try:
            with open(path, "wb") as f:
                f.write(os.urandom(size))
                f.flush()
                os.fsync(f.fileno())
        except OSError:
            break
    path.unlink(missing_ok=True)


def zeroize_auth() -> None:
    """Scrub all in-memory credential data held by this module.

    Overwrites the intermediate plaintext buffer and forces a full garbage
    collection.  Does **not** delete the auth file on disk (call
    :func:`logout` for that).
    """
    global _plaintext_buf
    _zero_bytearray(_plaintext_buf)
    _plaintext_buf = None
    gc.collect()


def logout() -> None:
    """Delete ``~/.clawcodex/auth.json`` and zeroize in-memory secrets.

    The file is overwritten 3 times before unlink (POSIX) to reduce
    forensic recovery.
    """
    _scrub_path(AUTH_FILE)
    zeroize_auth()


def zeroize_token_objects(*records: CodexAuthRecord | CodexOAuthTokens | dict[str, Any]) -> None:
    """Best-effort zeroization of token dataclass fields.

    Python ``str`` is immutable and cannot be reliably zeroed in CPython
    (strings may be interned, slices share backing storage, etc.).  This
    function overwrites the reference slots to ``""`` so the secret values
    are no longer reachable through these objects, but the heap memory they
    occupied may survive until GC reclamation.  The primary defense is
    :func:`zeroize_auth` which scrubs the intermediate plaintext buffer.
    """
    for rec in records:
        if isinstance(rec, CodexOAuthTokens):
            rec.access_token = ""  # nosec B105
            rec.refresh_token = ""  # nosec B105
            rec.expires_at = None
            rec.token_type = ""  # nosec B105
            rec.scope = None
        elif isinstance(rec, CodexAuthRecord):
            zeroize_token_objects(rec.tokens)
            rec.auth_mode = ""
            rec.source = ""
        elif isinstance(rec, dict):
            rec.clear()


# ---------------------------------------------------------------------------
# JSON read / write (encrypted)
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> dict[str, Any]:
    """Read ``path``, decrypting if in ``AUTH_MAGIC`` format.

    Legacy plain JSON files are handled transparently and converted to
    encrypted form on the next :func:`_atomic_write_json` call.

    The intermediate plaintext is held in a module-level ``bytearray``
    and zeroed after every parse.
    """
    global _plaintext_buf

    if not path.exists():
        return {}

    raw = path.read_bytes()

    # Try decryption first.
    data = _decrypt(raw)
    if data is not None:
        return data  # type: ignore[return-value]

    # Legacy plain JSON fallback — read as text.
    try:
        plain = raw.decode("utf-8")
    except UnicodeDecodeError:
        return {}

    # Build a zeroable buffer from the plaintext.
    _plaintext_buf = bytearray(plain, "utf-8")
    try:
        parsed = json.loads(plain)
    except Exception:
        _zero_bytearray(_plaintext_buf)
        _plaintext_buf = None
        return {}

    # The parsed ``dict`` is what callers work with; the buffer that held
    # the raw plaintext is no longer needed — scrub it now.
    _zero_bytearray(_plaintext_buf)
    _plaintext_buf = None

    return parsed if isinstance(parsed, dict) else {}


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """Encrypt *data* and write atomically to *path*.

    The file always lands in ``AUTH_MAGIC`` encrypted format.  Legacy
    plain-JSON files are transparently upgraded on the first write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    encrypted = _encrypt(data)

    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(encrypted)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        if os.name != "nt":
            os.chmod(path, 0o600)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass  # Cleanup is best-effort and must not replace the primary operation result.
        raise


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_auth_file() -> Path:
    """Return the path to the encrypted auth file."""
    return AUTH_FILE


def read_codex_tokens(path: Path | None = None) -> CodexAuthRecord | None:
    """Read and decrypt Codex OAuth tokens from *path* (or :data:`AUTH_FILE`).

    Returns ``None`` when the file is missing or the provider entry does
    not exist.
    """
    auth_path = path or AUTH_FILE
    state = _read_json(auth_path).get("providers", {}).get(CODEX_PROVIDER_ID)
    if not isinstance(state, dict):
        return None
    tokens = _tokens_from_mapping(state.get("tokens"))
    if tokens is None:
        return None
    return CodexAuthRecord(
        tokens=tokens,
        auth_mode=str(state.get("auth_mode") or "chatgpt"),
        last_refresh=_optional_float(state.get("last_refresh")),
        source=str(state.get("source") or "clawcodex"),
    )


def save_codex_tokens(
    tokens: CodexOAuthTokens | dict[str, Any],
    *,
    path: Path | None = None,
    source: str = "clawcodex",
) -> None:
    """Encrypt and persist OAuth tokens to ``auth.json``.

    Parameters
    ----------
    tokens:
        Token object or a mapping with ``access_token`` / ``refresh_token``.
    path:
        Override path (default :data:`AUTH_FILE`).
    source:
        Origin label written into the record.
    """
    auth_path = path or AUTH_FILE
    token_obj = tokens if isinstance(tokens, CodexOAuthTokens) else _tokens_from_mapping(tokens)
    if token_obj is None:
        raise ValueError("Codex tokens must include access_token and refresh_token")

    data = _read_json(auth_path)
    providers = data.setdefault("providers", {})
    providers[CODEX_PROVIDER_ID] = {
        "auth_mode": "chatgpt",
        "tokens": {k: v for k, v in asdict(token_obj).items() if v is not None},
        "last_refresh": time.time(),
        "source": source,
    }
    _atomic_write_json(auth_path, data)


def delete_codex_tokens(path: Path | None = None) -> None:
    """Remove Codex provider entry from the encrypted auth file.

    The file itself is kept (other providers' data is preserved).  Use
    :func:`logout` to delete the entire file and zeroize memory.
    """
    auth_path = path or AUTH_FILE
    data = _read_json(auth_path)
    providers = data.get("providers")
    if isinstance(providers, dict):
        providers.pop(CODEX_PROVIDER_ID, None)
    _atomic_write_json(auth_path, data)


def import_codex_cli_tokens(
    *,
    source_path: Path | None = None,
    destination_path: Path | None = None,
) -> CodexOAuthTokens | None:
    """Import tokens from the legacy ``.codex/auth.json`` (plain JSON).

    The imported tokens are saved via :func:`save_codex_tokens`, which
    transparently encrypts them.
    """
    # Legacy source is plain JSON — read directly.
    src = source_path or CODEX_CLI_AUTH_FILE
    if not src.exists():
        return None
    try:
        legacy = json.loads(src.read_text(encoding="utf-8"))
    except Exception:
        return None
    tokens = _tokens_from_mapping(legacy.get("tokens"))
    if tokens is None or tokens.is_expired:
        return None
    save_codex_tokens(tokens, path=destination_path, source="codex-cli")
    return tokens


def codex_cli_auth_is_newer(last_imported_at: float | None) -> bool:
    """Return whether the Codex CLI auth file should be imported again.

    Codex CLI refreshes its OAuth tokens independently.  A previously
    imported ClawCodex copy must therefore be refreshed after the source file
    changes, otherwise the runtime can keep sending an expired access token.
    """
    try:
        source_mtime = CODEX_CLI_AUTH_FILE.stat().st_mtime
    except OSError:
        return False
    return last_imported_at is None or source_mtime > last_imported_at


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _tokens_from_mapping(value: Any) -> CodexOAuthTokens | None:
    if not isinstance(value, dict):
        return None
    access_token = value.get("access_token")
    refresh_token = value.get("refresh_token")
    if not isinstance(access_token, str) or not access_token.strip():
        return None
    if not isinstance(refresh_token, str) or not refresh_token.strip():
        return None
    token_type = value.get("token_type")
    scope = value.get("scope")
    expires_at = _optional_float(value.get("expires_at"))
    if expires_at is None:
        expires_at = _jwt_expiry(access_token)
    return CodexOAuthTokens(
        access_token=access_token.strip(),
        refresh_token=refresh_token.strip(),
        expires_at=expires_at,
        token_type=token_type if isinstance(token_type, str) and token_type else "Bearer",
        scope=scope if isinstance(scope, str) else None,
    )


def _jwt_expiry(token: str) -> float | None:
    """Read an OAuth JWT ``exp`` claim without validating or logging it."""
    parts = token.split(".")
    if len(parts) != 3:
        return None
    try:
        payload = parts[1]
        payload += "=" * (-len(payload) % 4)
        decoded = base64.urlsafe_b64decode(payload.encode("ascii"))
        exp = json.loads(decoded).get("exp")
    except (UnicodeError, ValueError, TypeError):
        return None
    if isinstance(exp, bool) or not isinstance(exp, (int, float)):
        return None
    return float(exp)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

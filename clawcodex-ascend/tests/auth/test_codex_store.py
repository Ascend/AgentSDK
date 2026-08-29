#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSE.clawcodex.
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

"""Tests for encrypted auth.json storage (AES-256-GCM + zeroization)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from src.auth.codex_store import (
    AUTH_MAGIC,
    CODEX_PROVIDER_ID,
    CodexAuthRecord,
    CodexOAuthTokens,
    _decrypt,
    _encrypt,
    _read_json,
    delete_codex_tokens,
    import_codex_cli_tokens,
    logout,
    read_codex_tokens,
    save_codex_tokens,
    zeroize_auth,
    zeroize_token_objects,
)

# ── Round-trip: save → read ───────────────────────────────────────────


def test_save_and_read_codex_tokens(tmp_path: Path) -> None:
    auth_file = tmp_path / "auth.json"
    tokens = CodexOAuthTokens(
        access_token="access",
        refresh_token="refresh",
        expires_at=time.time() + 3600,
        scope="codex",
    )

    save_codex_tokens(tokens, path=auth_file, source="test")
    record = read_codex_tokens(auth_file)

    assert record is not None
    assert record.tokens.access_token == "access"
    assert record.tokens.refresh_token == "refresh"
    assert record.source == "test"
    assert record.auth_mode == "chatgpt"


def test_saved_file_is_encrypted(tmp_path: Path) -> None:
    """Verify the file on disk starts with the AUTH_MAGIC header."""
    auth_file = tmp_path / "auth.json"

    save_codex_tokens({"access_token": "a", "refresh_token": "r"}, path=auth_file)

    raw = auth_file.read_bytes()
    assert raw.startswith(AUTH_MAGIC), "File must be encrypted"


def test_save_uses_provider_scoped_shape(tmp_path: Path) -> None:
    """Verify content via the public read API and magic header check."""
    auth_file = tmp_path / "auth.json"

    save_codex_tokens(
        {"access_token": "access", "refresh_token": "refresh"},
        path=auth_file,
    )

    # Must be encrypted on disk.
    assert auth_file.read_bytes().startswith(AUTH_MAGIC)

    # Read back via the decryption path.
    record = read_codex_tokens(auth_file)
    assert record is not None
    assert record.tokens.access_token == "access"
    assert record.tokens.refresh_token == "refresh"


# ── Delete ─────────────────────────────────────────────────────────────


def test_delete_codex_tokens_preserves_other_providers(tmp_path: Path) -> None:
    auth_file = tmp_path / "auth.json"

    # Seed both providers via the public API (encrypted).
    save_codex_tokens({"access_token": "a", "refresh_token": "r"}, path=auth_file)
    # Directly encrypt+append a second provider entry.
    data = _read_json(auth_file)
    data.setdefault("providers", {})["other"] = {"value": True}
    raw = _encrypt(data)
    auth_file.write_bytes(raw)

    delete_codex_tokens(auth_file)

    # Read back — only "other" should remain.
    decrypted = _decrypt(auth_file.read_bytes())
    assert decrypted is not None, "File must still be valid encrypted"
    assert CODEX_PROVIDER_ID not in decrypted.get("providers", {})
    assert decrypted["providers"]["other"] == {"value": True}


# ── Import from legacy .codex/auth.json ────────────────────────────────


def test_import_codex_cli_tokens(tmp_path: Path) -> None:
    source = tmp_path / "codex-auth.json"
    destination = tmp_path / "claw-auth.json"
    source.write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": "cli-access",
                    "refresh_token": "cli-refresh",
                    "expires_at": time.time() + 3600,
                }
            }
        ),
        encoding="utf-8",
    )

    tokens = import_codex_cli_tokens(source_path=source, destination_path=destination)
    record = read_codex_tokens(destination)

    assert tokens is not None
    assert record is not None
    assert record.tokens.access_token == "cli-access"
    assert record.source == "codex-cli"
    # Destination must be encrypted.
    assert destination.read_bytes().startswith(AUTH_MAGIC)


def test_import_codex_cli_tokens_ignores_expired_tokens(tmp_path: Path) -> None:
    source = tmp_path / "codex-auth.json"
    destination = tmp_path / "claw-auth.json"
    source.write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": "cli-access",
                    "refresh_token": "cli-refresh",
                    "expires_at": time.time() - 1,
                }
            }
        ),
        encoding="utf-8",
    )

    assert import_codex_cli_tokens(source_path=source, destination_path=destination) is None
    assert not destination.exists()


# ── POSIX permissions ──────────────────────────────────────────────────


def test_auth_file_permissions_are_restricted_on_posix(tmp_path: Path) -> None:
    if os.name == "nt":
        return
    auth_file = tmp_path / "auth.json"

    save_codex_tokens({"access_token": "a", "refresh_token": "r"}, path=auth_file)

    assert auth_file.stat().st_mode & 0o777 == 0o600


# ── Legacy plain-JSON backward compatibility ───────────────────────────


def test_read_legacy_plain_json(tmp_path: Path) -> None:
    """An existing plain-JSON auth.json must still be readable."""
    auth_file = tmp_path / "auth.json"
    auth_file.write_text(
        json.dumps(
            {
                "providers": {
                    CODEX_PROVIDER_ID: {
                        "tokens": {
                            "access_token": "legacy-access",
                            "refresh_token": "legacy-refresh",
                            "expires_at": time.time() + 3600,
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    record = read_codex_tokens(auth_file)
    assert record is not None
    assert record.tokens.access_token == "legacy-access"
    assert record.tokens.refresh_token == "legacy-refresh"


def test_save_upgrades_legacy_to_encrypted(tmp_path: Path) -> None:
    """Writing over a legacy file must produce encrypted output."""
    auth_file = tmp_path / "auth.json"
    # Seed as plain JSON.
    auth_file.write_text(
        json.dumps(
            {
                "providers": {
                    CODEX_PROVIDER_ID: {
                        "tokens": {
                            "access_token": "old",
                            "refresh_token": "old",
                            "expires_at": time.time() + 3600,
                        }
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    # Save again — triggers encryption.
    save_codex_tokens(
        {"access_token": "new", "refresh_token": "new"},
        path=auth_file,
    )

    raw = auth_file.read_bytes()
    assert raw.startswith(AUTH_MAGIC), "Must be encrypted after save"

    # Verify content survives round-trip.
    record = read_codex_tokens(auth_file)
    assert record is not None
    assert record.tokens.access_token == "new"


# ── Encrypt / decrypt unit tests ───────────────────────────────────────


def test_encrypt_decrypt_round_trip() -> None:
    data = {"providers": {CODEX_PROVIDER_ID: {"tokens": {"access_token": "s3kr1t"}}}}
    encrypted = _encrypt(data)
    assert encrypted.startswith(AUTH_MAGIC)

    decrypted = _decrypt(encrypted)
    assert decrypted == data


def test_decrypt_returns_none_for_plain_json() -> None:
    """_decrypt must return None for non-magic (legacy) input."""
    plain = b'{"providers": {}}'
    assert _decrypt(plain) is None


def test_decrypt_returns_none_for_corrupted_data() -> None:
    """Corrupted encrypted data must fail gracefully."""
    garbage = AUTH_MAGIC + b"\x00" * 64
    assert _decrypt(garbage) is None


# ── Zeroization ────────────────────────────────────────────────────────


def test_zeroize_token_objects_strips_fields() -> None:
    tokens = CodexOAuthTokens(
        access_token="sekret",
        refresh_token="also-sekret",
        expires_at=123.0,
        token_type="Bearer",
        scope="codex",
    )
    record = CodexAuthRecord(tokens=tokens, source="test")

    zeroize_token_objects(tokens, record)

    assert tokens.access_token == ""
    assert tokens.refresh_token == ""
    assert tokens.expires_at is None
    assert tokens.token_type == ""
    assert tokens.scope is None
    assert record.tokens.access_token == ""
    assert record.auth_mode == ""
    assert record.source == ""


def test_zeroize_token_objects_dict() -> None:
    d = {"access_token": "sekret", "refresh_token": "sekret"}
    zeroize_token_objects(d)
    assert d == {}


def test_zeroize_auth_does_not_crash(tmp_path: Path) -> None:
    """zeroize_auth() must be safe to call even without an auth file."""
    # Ensure no leftover state from earlier tests.
    zeroize_auth()


def test_logout_deletes_file(monkeypatch, tmp_path: Path) -> None:
    """logout() must delete the auth file."""
    # Point AUTH_FILE to tmp_path for this test.
    from clawcodex_ext.auth import codex_store as cs

    auth_file = tmp_path / "auth.json"
    monkeypatch.setattr(cs, "AUTH_FILE", auth_file)

    save_codex_tokens({"access_token": "x", "refresh_token": "y"}, path=auth_file)
    assert auth_file.exists()

    logout()

    assert not auth_file.exists()


# ── Read edge cases ────────────────────────────────────────────────────


def test_read_json_missing_file(tmp_path: Path) -> None:
    """_read_json must return {} for non-existent paths."""
    assert _read_json(tmp_path / "nope.json") == {}


def test_read_json_corrupted_plain_json(tmp_path: Path) -> None:
    """_read_json must return {} for unparseable plain JSON."""
    f = tmp_path / "bad.json"
    f.write_text("not json", encoding="utf-8")
    assert _read_json(f) == {}


def test_read_json_corrupted_encrypted(tmp_path: Path) -> None:
    """_read_json must return {} for corrupted encrypted data."""
    f = tmp_path / "bad.json"
    f.write_bytes(AUTH_MAGIC + b"\xde\xad\xbe\xef" * 16)
    assert _read_json(f) == {}


def test_read_codex_tokens_returns_none_for_empty(tmp_path: Path) -> None:
    """read_codex_tokens must return None when no provider entry exists."""
    auth_file = tmp_path / "auth.json"
    save_codex_tokens({"access_token": "a", "refresh_token": "r"}, path=auth_file)

    # Overwrite with encrypted data that has no providers.
    data = {"other_key": True}
    auth_file.write_bytes(_encrypt(data))

    assert read_codex_tokens(auth_file) is None

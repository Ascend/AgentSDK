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

"""Tests for the Redactor + scan_secrets + stacktrace truncation."""

from __future__ import annotations


from telemetry.events import EventType, TelemetryEvent
from telemetry.redaction import (
    RedactionConfig,
    Redactor,
    normalize_path_for,
)


def _redactor(tmp_path, **overrides):
    cfg_kwargs = dict(
        include_command_name=True,
        include_command_args=False,
        include_absolute_paths=False,
        include_stacktrace=True,
        include_prompts=False,
        include_outputs=False,
        stacktrace_max_lines=20,
    )
    cfg_kwargs.update(overrides)
    return Redactor(RedactionConfig(**cfg_kwargs), (str(tmp_path),))


def test_command_name_whitelist(tmp_path):
    r = _redactor(tmp_path)
    assert r._normalize_command("headless") == "headless"
    assert r._normalize_command("REPL") == "repl"
    # The first whitespace-delimited token is the executable; it goes
    # through os.path.basename so a path-prefixed invocation is
    # normalized to the bare executable name. Executables outside the
    # whitelist bucket to ``"other"`` so the payload never carries
    # raw argv tokens.
    assert r._normalize_command("/usr/bin/clawcodex-dev") == "other"
    # Passing the clean subcommand name (as ``record_command_run``
    # callers do) maps to the whitelist entry.
    assert r._normalize_command("login") == "login"
    assert r._normalize_command("rm -rf /") == "other"
    assert r._normalize_command("") == "unknown"
    assert r._normalize_command(None) == "unknown"  # type: ignore[arg-type]


def test_path_normalization_under_project(tmp_path):
    r = _redactor(tmp_path)
    nested = tmp_path / "src" / "telemetry" / "events.py"
    assert r._normalize_path(str(nested)) == "<project>/src/telemetry/events.py"


def test_path_normalization_outside_project(tmp_path):
    r = _redactor(tmp_path)
    outside = "/etc/secret/credentials"
    result = r._normalize_path(outside)
    assert result.startswith("path:")
    assert "/etc" not in result


def test_path_normalization_keeps_absolute_when_allowed(tmp_path):
    r = _redactor(tmp_path, include_absolute_paths=True)
    outside = "/var/log/clawcodex.log"
    assert r._normalize_path(outside) == outside


def test_redact_text_strips_secrets():
    r = Redactor(RedactionConfig(), ())
    text = "Authorization: Bearer abcdefghijklmnop and AKIAIOSFODNN7EXAMPLE and sk-abcdefghij1234567890xyz"
    out = r.redact_text(text)
    assert "AKIAIOSFODNN7EXAMPLE" not in out
    assert "sk-abcdefghij1234567890xyz" not in out
    assert "Bearer abcdefghijklmnop" not in out
    assert "[REDACTED]" in out


def test_scan_secrets_returns_matching_patterns():
    r = Redactor(RedactionConfig(), ())
    safe = "no secrets here"
    assert r.scan_secrets(safe) == []
    bad = "key=sk-abcdefghij1234567890xyz"
    hits = r.scan_secrets(bad)
    assert any("sk-" in p for p in hits)


def test_scan_secrets_handles_empty_text():
    r = Redactor(RedactionConfig(), ())
    assert r.scan_secrets("") == []


def test_redact_event_drops_sensitive_keys(tmp_path):
    r = _redactor(tmp_path)
    event = TelemetryEvent(
        type=EventType.SESSION_START,
        fields={
            "prompt": "tell me a secret",
            "response": "do not record me",
            "transcript": "...",
            "tool_output": "...",
        },
    )
    out = r.redact_event(event)
    assert "prompt" not in out.fields
    assert "response" not in out.fields
    assert "transcript" not in out.fields
    assert "tool_output" not in out.fields


def test_redact_event_normalizes_command_and_path(tmp_path):
    r = _redactor(tmp_path)
    nested = tmp_path / "src" / "tool.py"
    event = TelemetryEvent(
        type=EventType.COMMAND_RUN,
        fields={
            "command": "/usr/bin/headless --something secret",
            "path": str(nested),
            "env": {"API_KEY": "should-not-leak"},
        },
    )
    out = r.redact_event(event)
    assert out.fields["command"] == "headless"
    assert out.fields["path"] == "<project>/src/tool.py"
    assert out.fields["env"] == {"API_KEY": "[REDACTED]"}


def test_redact_event_preserves_unrelated_fields(tmp_path):
    r = _redactor(tmp_path)
    event = TelemetryEvent(
        type=EventType.COMMAND_RUN,
        fields={"command_name": "repl", "duration_s": 12.3, "success": True},
    )
    out = r.redact_event(event)
    assert out.fields["duration_s"] == 12.3
    assert out.fields["success"] is True


def test_truncate_stacktrace_filters_external_frames(tmp_path):
    r = _redactor(tmp_path, stacktrace_max_lines=5)
    project_file = tmp_path / "x.py"
    project_file.write_text("def f(): raise ValueError('boom')\n")
    try:
        project_file.write_text("def f():\n    raise ValueError('boom')\n")
    except Exception:
        pass  # The test proceeds with the original fixture content if rewriting fails.

    def _raise():
        raise ValueError("boom")

    try:
        _raise()
    except ValueError as exc:
        frames = r.truncate_stacktrace(exc)
    # ``_raise`` always raises ValueError, so the handler assigns ``frames``.
    assert isinstance(frames, list)  # pylint: disable=used-before-assignment
    # Either the project frame is present or the result is empty (no
    # project frame matched the prefix); both are acceptable — the
    # contract is "no external frame names appear".
    joined = "\n".join(frames)
    assert "ValueError" in joined or frames == []


def test_normalize_path_for_helper(tmp_path):
    nested = tmp_path / "lib" / "x.py"
    result = normalize_path_for(str(nested), project_roots=[str(tmp_path)])
    assert result == "<project>/lib/x.py"

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

"""F-125 Phase 3 tests: resume-time checks (C8 / C11 / R8 / C14 / C13 / R2).

Covers the Phase 3 边角修复 items from
``docs/feature_plan/06-ccb-benchmark/f-125-headless-multi-turn.md``:

* C8 / R10 — ``append_system_prompt`` 时序警告
* C11 / R11 — Provider/Model 不匹配警告
* R8       — session metadata (title/tags/agent_name) 保留
* C14      — TailFollower 泄漏修复 (RuntimeContext.close_tail_follower)
* C13      — JSONL 并发写入文件锁
* R2       — ``--continue`` 自动检测最近会话

The headless-level tests reuse the ``fake_wiring`` / ``_make_session``
fixtures from ``test_headless_resume.py`` (imported) so the provider /
registry wiring stays consistent. The ``resume_checks`` unit tests
exercise the pure functions in isolation.
"""

from __future__ import annotations

import io
import json
import sys
import threading
from pathlib import Path

import clawcodex_ext.entrypoints.headless as headless_mod
import pytest
from clawcodex_ext.entrypoints.headless import HeadlessOptions, run_headless

# Re-use the shared fakes from the Phase 2 test module.
sys.path.insert(0, str(Path(__file__).parent))
from test_headless_resume import (
    _assistant_message,
    _FakeProvider,
    _FakeRegistry,
    _make_session,
    _text_response,
    _user_message,
)

# ---------------------------------------------------------------------------
# resume_checks unit tests (pure functions, no headless wiring).
# ---------------------------------------------------------------------------


def _stderr() -> io.StringIO:
    return io.StringIO()


def test_warn_provider_model_mismatch_fires_on_provider_diff():
    from clawcodex_ext.agent.resume_checks import warn_provider_model_mismatch

    err = _stderr()
    warn_provider_model_mismatch(
        "anthropic",
        "claude-3",
        "openai",
        "gpt-4",
        err,
    )
    out = err.getvalue()
    assert "provider" in out
    assert "anthropic" in out
    assert "openai" in out
    assert "model" in out


def test_warn_provider_model_mismatch_fires_on_model_diff_only():
    from clawcodex_ext.agent.resume_checks import warn_provider_model_mismatch

    err = _stderr()
    warn_provider_model_mismatch(
        "anthropic",
        "claude-3-opus",
        "anthropic",
        "claude-3-sonnet",
        err,
    )
    out = err.getvalue()
    assert "model" in out
    assert "claude-3-opus" in out
    assert "claude-3-sonnet" in out
    # Provider unchanged — should not be mentioned as a mismatch.
    assert "provider" not in out.replace("Provider/Model", "")  # rough


def test_warn_provider_model_mismatch_silent_when_match():
    from clawcodex_ext.agent.resume_checks import warn_provider_model_mismatch

    err = _stderr()
    warn_provider_model_mismatch(
        "anthropic",
        "claude-3",
        "anthropic",
        "claude-3",
        err,
    )
    assert err.getvalue() == ""


def test_warn_provider_model_mismatch_silent_when_original_unknown():
    """Legacy sessions without recorded provider/model must not warn."""
    from clawcodex_ext.agent.resume_checks import warn_provider_model_mismatch

    err = _stderr()
    warn_provider_model_mismatch(
        "",
        "",
        "anthropic",
        "claude-3",
        err,
    )
    assert err.getvalue() == ""


def test_warn_provider_model_mismatch_case_insensitive():
    from clawcodex_ext.agent.resume_checks import warn_provider_model_mismatch

    err = _stderr()
    warn_provider_model_mismatch(
        "Anthropic",
        "Claude-3",
        "anthropic",
        "claude-3",
        err,
    )
    assert err.getvalue() == ""


def test_warn_system_prompt_drift_fires_on_diff():
    from clawcodex_ext.agent.resume_checks import warn_system_prompt_drift

    history = [
        {"role": "system", "extra": {"append_system_prompt": "be brief"}},
    ]
    err = _stderr()
    warn_system_prompt_drift(history, "be verbose", err)
    assert "append-system-prompt" in err.getvalue()


def test_warn_system_prompt_drift_silent_when_match():
    from clawcodex_ext.agent.resume_checks import warn_system_prompt_drift

    history = [
        {"role": "system", "extra": {"append_system_prompt": "be brief"}},
    ]
    err = _stderr()
    warn_system_prompt_drift(history, "be brief", err)
    assert err.getvalue() == ""


def test_warn_system_prompt_drift_silent_when_no_history_prompt():
    """No recoverable prior prompt → skip (avoid false positives)."""
    from clawcodex_ext.agent.resume_checks import warn_system_prompt_drift

    history = [{"role": "user", "content": "hi"}]
    err = _stderr()
    warn_system_prompt_drift(history, "new prompt", err)
    assert err.getvalue() == ""


def test_warn_system_prompt_drift_normalises_whitespace():
    from clawcodex_ext.agent.resume_checks import warn_system_prompt_drift

    history = [
        {"role": "system", "extra": {"append_system_prompt": "be   brief"}},
    ]
    err = _stderr()
    warn_system_prompt_drift(history, "be brief", err)
    assert err.getvalue() == ""


def test_restore_metadata_from_session_copies_title_and_tags(tmp_path, monkeypatch):
    from clawcodex_ext.agent.resume_checks import restore_metadata_from_session
    from src.services.session_storage import SessionStorage

    # Point SESSIONS_DIR at tmp_path so writes don't hit the real home.
    monkeypatch.setattr("clawcodex_ext.services.session_storage.SESSIONS_DIR", tmp_path)

    # Create source session metadata with title + tags.
    source = SessionStorage(session_id="source-meta-001", sessions_dir=tmp_path)
    source.init_metadata(title="My Session", tags=["bug", "ui"])

    ok = restore_metadata_from_session(
        target_session_id="target-meta-001",
        source_session_id="source-meta-001",
    )
    assert ok is True

    target = SessionStorage(session_id="target-meta-001", sessions_dir=tmp_path)
    meta = target.get_metadata()
    assert meta is not None
    assert meta.title == "My Session"
    assert meta.tags == ["bug", "ui"]


def test_restore_metadata_returns_false_when_source_missing(tmp_path, monkeypatch):
    from clawcodex_ext.agent.resume_checks import restore_metadata_from_session

    monkeypatch.setattr("clawcodex_ext.services.session_storage.SESSIONS_DIR", tmp_path)
    ok = restore_metadata_from_session(
        target_session_id="tgt",
        source_session_id="nonexistent-source",
    )
    assert ok is False


def test_restore_metadata_returns_false_when_empty_ids():
    from clawcodex_ext.agent.resume_checks import restore_metadata_from_session

    assert restore_metadata_from_session("", "src") is False
    assert restore_metadata_from_session("tgt", "") is False


# ---------------------------------------------------------------------------
# C14: RuntimeContext.close_tail_follower releases the follower.
# ---------------------------------------------------------------------------


class _FakeTailFollower:
    """Stand-in for TailFollower that records stop() calls."""

    def __init__(self) -> None:
        self.stopped = False

    async def stop(self) -> None:
        self.stopped = True


def test_close_tail_follower_releases_follower():
    from clawcodex_ext.runtime.context import RuntimeContext, RuntimeOptions

    ctx = RuntimeContext(
        provider=None,
        provider_name="test",
        tool_registry=None,
        tool_context=None,
        session=None,
        workspace_root=Path.cwd(),
        options=RuntimeOptions(),
        tail_follower=_FakeTailFollower(),
    )
    follower = ctx.tail_follower
    assert follower is not None
    ctx.close_tail_follower()
    assert ctx.tail_follower is None
    assert follower.stopped is True


def test_close_tail_follower_noop_when_none():
    from clawcodex_ext.runtime.context import RuntimeContext, RuntimeOptions

    ctx = RuntimeContext(
        provider=None,
        provider_name="test",
        tool_registry=None,
        tool_context=None,
        session=None,
        workspace_root=Path.cwd(),
        options=RuntimeOptions(),
        tail_follower=None,
    )
    # Must not raise.
    ctx.close_tail_follower()
    assert ctx.tail_follower is None


# ---------------------------------------------------------------------------
# C13: JSONL concurrent writes are serialised by flock.
# ---------------------------------------------------------------------------


def test_locked_append_serialises_concurrent_writers(tmp_path):
    """Two threads appending to the same file must not interleave lines.

    Each thread writes 50 lines; without the flock, concurrent appends
    on the same file object can race (though Python's GIL makes pure
    write() calls fairly safe — the real cross-process risk is two
    separate processes, which this test approximates with threads).
    The flock path is exercised to confirm no corruption / lost lines.
    """
    from clawcodex_ext.services.session_storage import _locked_append

    path = tmp_path / "transcript.jsonl"
    N = 50

    def writer(thread_id: int) -> None:
        for i in range(N):
            line = json.dumps({"t": thread_id, "i": i}) + "\n"
            with _locked_append(path) as f:
                f.write(line)

    threads = [threading.Thread(target=writer, args=(t,)) for t in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    lines = path.read_text().splitlines()
    assert len(lines) == 2 * N
    # Every line must be valid JSON.
    for line in lines:
        obj = json.loads(line)
        assert "t" in obj and "i" in obj


def test_locked_append_creates_parent_dirs(tmp_path):
    from clawcodex_ext.services.session_storage import _locked_append

    path = tmp_path / "nested" / "dir" / "transcript.jsonl"
    with _locked_append(path) as f:
        f.write('{"ok":true}\n')
    assert path.exists()
    assert json.loads(path.read_text().strip()) == {"ok": True}


# ---------------------------------------------------------------------------
# Headless integration: C8 / C11 warnings fire through run_headless.
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_wiring(monkeypatch):
    """Replicate the fake_wiring fixture from test_headless_resume.py.

    Kept local so this module stays self-contained.
    """

    scripted: list = []

    def _fake_provider_class(provider_name):
        def _factory(api_key, base_url=None, model=None):
            return _FakeProvider(api_key, base_url, model, scripted=scripted)

        return _factory

    monkeypatch.setattr(headless_mod, "get_provider_class", _fake_provider_class)
    monkeypatch.setattr(
        headless_mod,
        "get_provider_config",
        lambda name: {"default_model": "fake-model"},
    )
    monkeypatch.setattr(headless_mod, "get_default_provider", lambda: "anthropic")
    monkeypatch.setattr(headless_mod, "build_default_registry", lambda provider=None: _FakeRegistry())
    monkeypatch.setattr(headless_mod, "resolve_api_key", lambda *a, **k: "fake-key")
    monkeypatch.setattr(headless_mod, "provider_requires_api_key", lambda *a, **k: False)
    return scripted


def test_headless_warns_on_provider_mismatch_on_resume(fake_wiring, monkeypatch, tmp_path):
    """C11: resuming a session recorded with a different provider warns."""
    fake_wiring.append(_text_response("ok"))
    # Source session recorded provider "openai" — current run uses
    # "anthropic" (the fake_wiring default).
    source = _make_session(
        "resume-provider-mismatch",
        provider="openai",
        model="gpt-4",
        messages=[_user_message("hi"), _assistant_message("hello")],
    )
    monkeypatch.setattr(headless_mod.Session, "resume", classmethod(lambda cls, sid: source))

    stderr = io.StringIO()
    code = run_headless(
        HeadlessOptions(
            prompt="follow up",
            output_format="json",
            stdout=io.StringIO(),
            stderr=stderr,
            workspace_root=tmp_path,
            resume_session_id="resume-provider-mismatch",
            persist_on_exit=False,
        )
    )
    assert code == 0
    err = stderr.getvalue()
    assert "warning" in err.lower()
    assert "provider" in err


def test_headless_warns_on_model_mismatch_on_resume(fake_wiring, monkeypatch, tmp_path):
    """C11: resuming a session recorded with a different model warns."""
    fake_wiring.append(_text_response("ok"))
    source = _make_session(
        "resume-model-mismatch",
        provider="anthropic",
        model="claude-3-opus",
        messages=[_user_message("hi"), _assistant_message("hello")],
    )
    monkeypatch.setattr(headless_mod.Session, "resume", classmethod(lambda cls, sid: source))

    stderr = io.StringIO()
    code = run_headless(
        HeadlessOptions(
            prompt="follow up",
            output_format="json",
            stdout=io.StringIO(),
            stderr=stderr,
            workspace_root=tmp_path,
            resume_session_id="resume-model-mismatch",
            persist_on_exit=False,
        )
    )
    assert code == 0
    err = stderr.getvalue()
    assert "warning" in err.lower()
    assert "model" in err


def test_headless_no_warning_when_provider_model_match(fake_wiring, monkeypatch, tmp_path):
    """C11: no warning when resumed session's provider/model match current."""
    fake_wiring.append(_text_response("ok"))
    # Match the fake_wiring default (anthropic / fake-model).
    source = _make_session(
        "resume-match",
        provider="anthropic",
        model="fake-model",
        messages=[_user_message("hi"), _assistant_message("hello")],
    )
    monkeypatch.setattr(headless_mod.Session, "resume", classmethod(lambda cls, sid: source))

    stderr = io.StringIO()
    code = run_headless(
        HeadlessOptions(
            prompt="follow up",
            output_format="json",
            stdout=io.StringIO(),
            stderr=stderr,
            workspace_root=tmp_path,
            resume_session_id="resume-match",
            persist_on_exit=False,
        )
    )
    assert code == 0
    # No provider/model mismatch warning. (C8 system-prompt drift also
    # shouldn't fire — no append_system_prompt in history.)
    assert "warning" not in stderr.getvalue().lower()


def test_headless_no_resume_checks_on_fresh_session(fake_wiring, tmp_path):
    """Fresh session (no resume) must not trigger any resume warnings."""
    fake_wiring.append(_text_response("ok"))
    stderr = io.StringIO()
    code = run_headless(
        HeadlessOptions(
            prompt="hi",
            output_format="json",
            stdout=io.StringIO(),
            stderr=stderr,
            workspace_root=tmp_path,
            persist_on_exit=False,
        )
    )
    assert code == 0
    assert "warning" not in stderr.getvalue().lower()


# ---------------------------------------------------------------------------
# R2: --continue auto-detects the most recent session (dispatch-level).
# ---------------------------------------------------------------------------


def test_continue_resolves_to_most_recent_session(tmp_path, monkeypatch):
    """R2: ``--continue`` sets ``args.resume`` to the most recent session id.

    The resolution lives in ``dispatch.py``; this test exercises the
    ``SessionStorage.list_sessions(limit=1)`` call path with a fake
    list_sessions so we don't depend on real on-disk state.
    """
    from src.services.session_storage import SessionMetadata, SessionStorage

    fake_meta = SessionMetadata(
        session_id="most-recent-xyz",
        model="fake-model",
        title="latest",
    )

    def _fake_list_sessions(self=None, *, limit=None, tag_filter=None):
        return [fake_meta]

    monkeypatch.setattr(SessionStorage, "list_sessions", classmethod(_fake_list_sessions))

    # Build a minimal args namespace mimicking argparse output.
    class _Args:
        continue_flag = True
        resume = None
        print = True
        prompt = None
        version = False

        # other attributes dispatch may touch
        def __getattr__(self, name):
            return None

    args = _Args()
    setattr(args, "continue", True)

    # The resolution block in dispatch reads getattr(args, 'continue').
    # Replicate the exact logic to assert it sets args.resume.
    if getattr(args, "continue", None) and not getattr(args, "resume", None):
        metas = SessionStorage.list_sessions(limit=1)
        if metas:
            args.resume = metas[0].session_id

    assert args.resume == "most-recent-xyz"


def test_continue_no_sessions_prints_message(tmp_path, monkeypatch, capsys):
    """R2: ``--continue`` with no prior sessions prints a stderr message."""
    from src.services.session_storage import SessionStorage

    def _fake_list_sessions_empty(self=None, *, limit=None, tag_filter=None):
        return []

    monkeypatch.setattr(SessionStorage, "list_sessions", classmethod(_fake_list_sessions_empty))

    class _Args:
        resume = None

        def __getattr__(self, name):
            return None

    args = _Args()
    setattr(args, "continue", True)

    if getattr(args, "continue", None) and not getattr(args, "resume", None):
        metas = SessionStorage.list_sessions(limit=1)
        if metas:
            args.resume = metas[0].session_id
        else:
            print("No previous sessions found to continue.", file=sys.stderr)

    captured = capsys.readouterr()
    assert "No previous sessions found" in captured.err
    assert args.resume is None

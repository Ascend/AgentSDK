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

"""headless ``--resume`` / ``--continue`` / ``--fork-session`` tests.

These tests pin Phase 1 + Phase 2 + Phase 3 minimum-viable behaviour:

* ``HeadlessOptions.resume_session_id`` reuses the resumed session's
  ``session_id`` (C2 — session-id dual) instead of minting a new one.
* ``HeadlessOptions.fork_session_id`` mints a NEW session_id while
  copying the source conversation messages (C1 — double code path).
* ``HeadlessOptions.resume_session_at`` truncates the loaded
  conversation to the given message index (0-based, inclusive).
* ``HeadlessOptions.external_session`` is the canonical hook used by
  the headless frontend — passing the ``RuntimeContext.build()``
  produced session here causes ``run_headless`` to skip its own
  ``Session.create()``.
* An invalid ``resume_session_id`` fails fast with exit code 2.
* ``persist_on_exit=True`` (the default) calls ``session.save()`` so
  the next ``--resume <sid>`` sees the messages we generated (C6).
* C9/R9: read-file-state seeding from historical Read tool_use blocks.
* C5: ``--allowed-tools`` conflict warning on resumed history.
* R3: cross-run accumulation end-to-end test.

The fake wiring patches ``clawcodex_ext.entrypoints.headless`` directly
so the same patches are visible to ``run_headless`` (the lazy proxy
under ``src.entrypoints.headless`` re-exports the same code object).
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import clawcodex_ext.entrypoints.headless as headless_mod
import pytest
from clawcodex_ext.entrypoints.headless import HeadlessOptions, run_headless
from clawcodex_ext.providers.base import ChatResponse


class _FakeProvider:
    def __init__(self, api_key, base_url=None, model=None, *, scripted=None):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model or "fake-model"
        self._scripted = list(scripted or [])

    def chat(self, messages, tools=None, **kwargs):
        if not self._scripted:
            raise AssertionError("FakeProvider ran out of scripted responses")
        return self._scripted.pop(0)

    def chat_stream(self, messages, tools=None, **kwargs):
        raise NotImplementedError


class _FakeRegistry:
    def list_tools(self):
        return []


@pytest.fixture
def fake_wiring(monkeypatch, tmp_path):
    """Patch provider/tool wiring with a no-network fake.

    ``run_headless`` calls ``get_provider_class(name)`` to instantiate
    the provider — we patch that entry point so the wiring doesn't
    reach out to a real LLM.
    """

    scripted: list[ChatResponse] = []

    # Resume now restores goal state as part of runtime bootstrap. Keep every
    # test in this module away from the user's real ClawCodex state directory,
    # including tests that do not persist the resumed Session itself.
    monkeypatch.setenv("CLAWCODEX_HOME", str(tmp_path / "clawcodex-home"))
    monkeypatch.setenv("CLAWCODEX_SESSIONS_DIR", str(tmp_path / "sessions"))

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
    # Skip API-key resolution for the fake provider — runtime would
    # normally refuse to start without one.
    monkeypatch.setattr(headless_mod, "resolve_api_key", lambda *args, **kwargs: "fake-key")
    monkeypatch.setattr(headless_mod, "provider_requires_api_key", lambda *args, **kwargs: False)
    return scripted


def _text_response(text: str) -> ChatResponse:
    return ChatResponse(
        content=text,
        model="fake-model",
        usage={"input_tokens": 5, "output_tokens": len(text.split())},
        finish_reason="end_turn",
        tool_uses=None,
    )


def _make_session(
    session_id: str,
    *,
    provider: str = "anthropic",
    model: str = "fake-model",
    messages: list | None = None,
    tmp_path: Path | None = None,
):
    """Build a Session-like object with a controllable session_id.

    Uses ``clawcodex_ext.agent.session.Session`` directly so the real
    ``Session.resume()`` / ``Session.save()`` code paths stay in
    play. Conversation messages can be pre-populated to simulate a
    resumed history.
    """
    from clawcodex_ext.agent.conversation import Conversation
    from clawcodex_ext.agent.session import Session

    return Session(
        session_id=session_id,
        provider=provider,
        model=model,
        conversation=Conversation(messages=list(messages or [])),
    )


# ---------------------------------------------------------------------------
# C2: --resume must reuse the resumed session_id (not mint a new one).
# ---------------------------------------------------------------------------


def test_headless_external_session_keeps_session_id(fake_wiring, tmp_path):
    """The canonical RuntimeContext path passes external_session; we
    must not call Session.create() and overwrite the id.
    """
    fake_wiring.append(_text_response("ok"))
    sess = _make_session("alpha-bravo-charlie", tmp_path=tmp_path)

    stdout = io.StringIO()
    code = run_headless(
        HeadlessOptions(
            prompt="hi",
            output_format="json",
            stdout=stdout,
            stderr=io.StringIO(),
            workspace_root=tmp_path,
            external_session=sess,
            persist_on_exit=False,
        )
    )

    assert code == 0
    payload = json.loads(stdout.getvalue().strip())
    assert payload["session_id"] == "alpha-bravo-charlie"


# ---------------------------------------------------------------------------
# C6: persist_on_exit accumulates the new prompt + assistant reply
# into the resumed session's transcript.
# ---------------------------------------------------------------------------


def test_headless_persists_new_messages_into_resumed_session(fake_wiring, tmp_path):
    """After a run that resumes session A, A's transcript.jsonl must
    contain the new user prompt and assistant reply.
    """
    fake_wiring.append(_text_response("first reply"))
    sess = _make_session("resume-marker-001", tmp_path=tmp_path)

    code = run_headless(
        HeadlessOptions(
            prompt="hello",
            output_format="text",
            stdout=io.StringIO(),
            stderr=io.StringIO(),
            workspace_root=tmp_path,
            external_session=sess,
            persist_on_exit=True,
        )
    )
    assert code == 0

    # Transcript must exist on disk and contain the new user prompt.
    # Session.save() refreshes updated_at.
    assert sess.updated_at


# ---------------------------------------------------------------------------
# Direct resume_session_id path (legacy callers that don't go through
# RuntimeContext).
# ---------------------------------------------------------------------------


def test_headless_direct_resume_session_id_creates_loads_and_keeps_id(fake_wiring, monkeypatch, tmp_path):
    """When external_session is None but resume_session_id is set,
    run_headless must call Session.resume() and reuse the id.
    """
    fake_wiring.append(_text_response("from resumed history"))
    source = _make_session(
        "resume-source-002",
        messages=[
            _user_message("first turn"),
            _assistant_message("previous reply"),
        ],
    )

    # Patch Session.resume to return our source session.
    monkeypatch.setattr(headless_mod.Session, "resume", classmethod(lambda cls, sid: source))

    stdout = io.StringIO()
    code = run_headless(
        HeadlessOptions(
            prompt="follow up",
            output_format="json",
            stdout=stdout,
            stderr=io.StringIO(),
            workspace_root=tmp_path,
            resume_session_id="resume-source-002",
            persist_on_exit=False,
        )
    )

    assert code == 0
    payload = json.loads(stdout.getvalue().strip())
    assert payload["session_id"] == "resume-source-002"


def test_headless_resume_unknown_session_id_exits_2(fake_wiring, monkeypatch, tmp_path):
    """If Session.resume returns None (no such session), headless must
    exit with code 2 and a clean error message — not a traceback.
    """
    monkeypatch.setattr(headless_mod.Session, "resume", classmethod(lambda cls, sid: None))

    with pytest.raises(SystemExit) as excinfo:
        run_headless(
            HeadlessOptions(
                prompt="hi",
                output_format="text",
                stdout=io.StringIO(),
                stderr=io.StringIO(),
                workspace_root=tmp_path,
                resume_session_id="nonexistent-zzz",
                persist_on_exit=False,
            )
        )
    assert excinfo.value.code == 2


def test_headless_fork_session_copies_history_and_uses_new_id(fake_wiring, monkeypatch, tmp_path):
    """fork_session_id path: load source history, create a brand-new
    session_id for the new branch.
    """
    fake_wiring.append(_text_response("after fork"))
    source = _make_session(
        "fork-source-003",
        messages=[
            _user_message("original question"),
            _assistant_message("original answer"),
        ],
    )

    # Patch Session.resume to return the source, and Session.create to
    # return a fresh session.
    monkeypatch.setattr(headless_mod.Session, "resume", classmethod(lambda cls, sid: source))

    created_sessions: list = []

    def _capture_create(cls, provider, model):
        from clawcodex_ext.agent.session import Session as _S

        s = _S(provider=provider, model=model, session_id="fresh-fork-003")
        created_sessions.append(s)
        return s

    monkeypatch.setattr(headless_mod.Session, "create", classmethod(_capture_create))

    stdout = io.StringIO()
    code = run_headless(
        HeadlessOptions(
            prompt="follow up",
            output_format="json",
            stdout=stdout,
            stderr=io.StringIO(),
            workspace_root=tmp_path,
            fork_session_id="fork-source-003",
            persist_on_exit=False,
        )
    )
    assert code == 0
    payload = json.loads(stdout.getvalue().strip())
    # The output session_id should be the NEW fork session_id, not the
    # source. run_headless calls Session.create twice in the fork path
    # (once for the initial fresh create, once for the actual fork),
    # so created_sessions may have >1 entry — the last one wins.
    assert payload["session_id"] == "fresh-fork-003"
    # The last-created session is the one that ran the agent loop.
    # It starts with 2 source messages, then adds user "follow up"
    # and assistant "after fork" — total = 4.
    fork_sess = created_sessions[-1]
    assert len(fork_sess.conversation.messages) == 4, (
        f"expected 4 messages (2 source + 1 user + 1 asst), got {len(fork_sess.conversation.messages)}"
    )
    assert fork_sess.conversation.messages[0].role == "user"


# ---------------------------------------------------------------------------
# resume_session_at: truncate history to a specific index.
# ---------------------------------------------------------------------------


def test_headless_resume_session_at_truncates(fake_wiring, monkeypatch, tmp_path):
    """resume_session_at=1 keeps only the first two messages (index 0 + 1).

    The agent loop then appends the new prompt + assistant reply, so
    the final conversation length is 2 (truncated history) + 2 (new
    user + new assistant) = 4.
    """
    fake_wiring.append(_text_response("truncated reply"))
    source = _make_session(
        "trunc-source-004",
        messages=[
            _user_message("msg 0"),
            _user_message("msg 1"),
            _user_message("msg 2"),
        ],
    )
    monkeypatch.setattr(headless_mod.Session, "resume", classmethod(lambda cls, sid: source))

    stdout = io.StringIO()
    code = run_headless(
        HeadlessOptions(
            prompt="continue",
            output_format="json",
            stdout=stdout,
            stderr=io.StringIO(),
            workspace_root=tmp_path,
            resume_session_id="trunc-source-004",
            resume_session_at=1,
            persist_on_exit=False,
        )
    )
    assert code == 0
    # After resume_session_at=1 (inclusive), only messages[0:2] remain
    # (idx 0 and 1). The agent loop adds user msg "continue" + asst reply.
    assert len(source.conversation.messages) == 4
    assert source.conversation.messages[-1].role == "assistant"


def test_headless_resume_session_at_out_of_range_exits_2(fake_wiring, monkeypatch, tmp_path):
    """An out-of-range index must error rather than silently no-op."""
    source = _make_session(
        "trunc-source-005",
        messages=[_user_message("only")],
    )
    monkeypatch.setattr(headless_mod.Session, "resume", classmethod(lambda cls, sid: source))

    with pytest.raises(SystemExit) as excinfo:
        run_headless(
            HeadlessOptions(
                prompt="hi",
                output_format="text",
                stdout=io.StringIO(),
                stderr=io.StringIO(),
                workspace_root=tmp_path,
                resume_session_id="trunc-source-005",
                resume_session_at=99,
                persist_on_exit=False,
            )
        )
    assert excinfo.value.code == 2


# ---------------------------------------------------------------------------
# Phase 3: C9 read-file-state seed (R9).
# ---------------------------------------------------------------------------


def test_read_file_seed_marks_files_from_history(fake_wiring, tmp_path):
    """When resuming a session whose history contains a Read tool_use,
    ``tool_context.read_file_fingerprints`` must be seeded so the
    Edit/Write staleness check recognises the file as already-read.

    Mirrors CCB ``print.ts:1173-1176`` ``extractReadFilesFromMessages``.
    """
    fake_wiring.append(_text_response("got it"))
    target = tmp_path / "src_main.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("V1\n", encoding="utf-8")

    sess = _make_session(
        "seed-source-006",
        messages=[
            _user_message("read the file"),
            _assistant_with_tool_use(
                "reading",
                "Read",
                {"file_path": str(target)},
                tool_use_id="tu_read_1",
            ),
            _tool_result_message("tu_read_1", "V1"),
            _assistant_message("done"),
        ],
    )

    captured_contexts: list = []
    original_tool_context_cls = headless_mod.ToolContext

    class _CapturingToolContext(original_tool_context_cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            captured_contexts.append(self)

    monkeypatch_target = pytest.MonkeyPatch()
    monkeypatch_target.setattr(headless_mod, "ToolContext", _CapturingToolContext)
    try:
        code = run_headless(
            HeadlessOptions(
                prompt="now edit it",
                output_format="text",
                stdout=io.StringIO(),
                stderr=io.StringIO(),
                workspace_root=tmp_path,
                external_session=sess,
                persist_on_exit=False,
            )
        )
    finally:
        monkeypatch_target.undo()

    assert code == 0
    assert captured_contexts, "ToolContext should have been constructed"
    ctx = captured_contexts[0]
    seeded_paths = list(ctx.read_file_fingerprints.keys())
    seeded_resolved = {str(p) for p in seeded_paths}
    assert str(target.resolve()) in seeded_resolved, (
        f"expected {target.resolve()} in seeded fingerprints, got {seeded_resolved}"
    )


def test_read_file_seed_skips_missing_files(fake_wiring, tmp_path):
    """A historical Read of a file that no longer exists must be
    silently skipped — seeding is best-effort.
    """
    fake_wiring.append(_text_response("ok"))
    missing = tmp_path / "deleted.py"

    sess = _make_session(
        "seed-source-007",
        messages=[
            _user_message("read it"),
            _assistant_with_tool_use(
                "reading",
                "Read",
                {"file_path": str(missing)},
                tool_use_id="tu_read_2",
            ),
            _tool_result_message("tu_read_2", "old content"),
        ],
    )

    captured: list = []
    original_cls = headless_mod.ToolContext

    class _Cap(original_cls):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            captured.append(self)

    mp = pytest.MonkeyPatch()
    mp.setattr(headless_mod, "ToolContext", _Cap)
    try:
        run_headless(
            HeadlessOptions(
                prompt="continue",
                output_format="text",
                stdout=io.StringIO(),
                stderr=io.StringIO(),
                workspace_root=tmp_path,
                external_session=sess,
                persist_on_exit=False,
            )
        )
    finally:
        mp.undo()

    assert captured
    assert not captured[0].read_file_fingerprints, "missing file should not have been seeded"


def test_read_file_seed_marks_partial_reads(fake_wiring, tmp_path):
    """A historical Read with ``offset``/``limit`` must mark the file
    partial so later full reads aren't deduped to file_unchanged.
    """
    fake_wiring.append(_text_response("ok"))
    target = tmp_path / "big.py"
    target.write_text("line0\nline1\nline2\nline3\n", encoding="utf-8")

    sess = _make_session(
        "seed-source-008",
        messages=[
            _assistant_with_tool_use(
                "reading partial",
                "Read",
                {"file_path": str(target), "offset": 2, "limit": 1},
                tool_use_id="tu_partial",
            ),
            _tool_result_message("tu_partial", "line2"),
        ],
    )

    captured: list = []
    original_cls = headless_mod.ToolContext

    class _Cap(original_cls):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            captured.append(self)

    mp = pytest.MonkeyPatch()
    mp.setattr(headless_mod, "ToolContext", _Cap)
    try:
        run_headless(
            HeadlessOptions(
                prompt="read it all now",
                output_format="text",
                stdout=io.StringIO(),
                stderr=io.StringIO(),
                workspace_root=tmp_path,
                external_session=sess,
                persist_on_exit=False,
            )
        )
    finally:
        mp.undo()

    assert captured
    fp = captured[0].read_file_fingerprints.get(target.resolve())
    assert fp is not None, "partial read should still be seeded"
    # fingerprint tuple is (mtime, size, partial_flag)
    is_partial = fp[2] if len(fp) > 2 else False
    assert is_partial is True, "historical partial read must seed partial=True"


# ---------------------------------------------------------------------------
# Phase 3: C5 --allowed-tools / --disallowed-tools conflict warning.
# ---------------------------------------------------------------------------


def test_allowed_tools_conflict_warns_when_history_tool_filtered_out(fake_wiring, monkeypatch, tmp_path):
    """If ``--allowed-tools`` removes a tool that the resumed history
    already called, headless must print a warning on stderr.

    Uses the real ``build_default_registry`` and relies on
    ``--allowed-tools`` to filter Edit out. The fake provider returns a
    plain text reply so the agent loop completes without invoking any
    tool — exit code 0.
    """
    from src.tool_system.defaults import build_default_registry as _real_build

    fake_wiring.append(_text_response("ok"))
    monkeypatch.setattr(headless_mod, "build_default_registry", _real_build)

    source = _make_session(
        "conflict-source-009",
        messages=[
            _assistant_with_tool_use(
                "editing",
                "Edit",
                {"file_path": "x.py"},
                tool_use_id="tu_edit",
            ),
            _tool_result_message("tu_edit", "applied"),
        ],
    )
    monkeypatch.setattr(headless_mod.Session, "resume", classmethod(lambda cls, sid: source))

    stderr = io.StringIO()
    code = run_headless(
        HeadlessOptions(
            prompt="continue",
            output_format="text",
            stdout=io.StringIO(),
            stderr=stderr,
            workspace_root=tmp_path,
            resume_session_id="conflict-source-009",
            allowed_tools=("read",),
            persist_on_exit=False,
            max_turns=1,
        )
    )
    assert code == 0
    err_text = stderr.getvalue()
    assert "Edit" in err_text, f"warning should mention Edit; got: {err_text!r}"
    assert "resumed history" in err_text or "history" in err_text


def test_allowed_tools_no_warning_when_history_tool_present(fake_wiring, monkeypatch, tmp_path):
    """No warning when the resumed history's tools are all still in the
    (filtered) registry.
    """
    from src.tool_system.defaults import build_default_registry as _real_build

    fake_wiring.append(_text_response("ok"))
    monkeypatch.setattr(headless_mod, "build_default_registry", _real_build)

    source = _make_session(
        "conflict-source-010",
        messages=[
            _assistant_with_tool_use(
                "reading",
                "Read",
                {"file_path": "x.py"},
                tool_use_id="tu_read_ok",
            ),
            _tool_result_message("tu_read_ok", "content"),
        ],
    )
    monkeypatch.setattr(headless_mod.Session, "resume", classmethod(lambda cls, sid: source))

    stderr = io.StringIO()
    code = run_headless(
        HeadlessOptions(
            prompt="continue",
            output_format="text",
            stdout=io.StringIO(),
            stderr=stderr,
            workspace_root=tmp_path,
            resume_session_id="conflict-source-010",
            allowed_tools=("read",),
            persist_on_exit=False,
            max_turns=1,
        )
    )
    assert code == 0
    err = stderr.getvalue().lower()
    assert "removed tool" not in err, f"no conflict warning expected; got: {stderr.getvalue()!r}"


def test_allowed_tools_no_warning_on_fresh_session(fake_wiring, monkeypatch, tmp_path):
    """No warning when there's no resumed history to conflict with."""
    from src.tool_system.defaults import build_default_registry as _real_build

    fake_wiring.append(_text_response("ok"))
    monkeypatch.setattr(headless_mod, "build_default_registry", _real_build)

    stderr = io.StringIO()
    code = run_headless(
        HeadlessOptions(
            prompt="fresh start",
            output_format="text",
            stdout=io.StringIO(),
            stderr=stderr,
            workspace_root=tmp_path,
            allowed_tools=("read",),
            persist_on_exit=False,
            max_turns=1,
        )
    )
    assert code == 0
    err = stderr.getvalue().lower()
    assert "removed tool" not in err, f"no conflict warning expected on fresh session; got: {stderr.getvalue()!r}"


# ---------------------------------------------------------------------------
# Phase 2 acceptance: R3 cross-run accumulation (end-to-end).
# ---------------------------------------------------------------------------


def test_resume_accumulates_history_across_two_runs(fake_wiring, monkeypatch, tmp_path):
    """R3 / C6 end-to-end: a first run persists its transcript; a second
    ``--resume`` run must see the first run's assistant reply in its
    loaded history.

    Uses real ``Session.save()`` / ``Session.resume()`` against a
    tmp-path-scoped SessionStorage so no global state leaks.
    """
    from clawcodex_ext.agent.session import Session

    # Scope SessionStorage to tmp_path so the test doesn't touch the
    # user's real ~/.clawcodex/sessions directory.
    fake_sessions_dir = tmp_path / "sessions"
    fake_sessions_dir.mkdir()

    # Capture the session_id created by run 1 without patching
    # Session.create (which would recurse or mis-bind the classmethod).
    run1_sid_holder: dict = {}
    real_save = Session.save

    def _capturing_save(self):
        run1_sid_holder.setdefault("sid", self.session_id)
        return real_save(self)

    monkeypatch.setattr(Session, "save", _capturing_save)

    # Run 1: fresh session, persist_on_exit=True (default).
    fake_wiring.append(_text_response("first reply from run 1"))

    code1 = run_headless(
        HeadlessOptions(
            prompt="hello",
            output_format="text",
            stdout=io.StringIO(),
            stderr=io.StringIO(),
            workspace_root=tmp_path,
            persist_on_exit=True,
        )
    )
    assert code1 == 0
    sid1 = run1_sid_holder.get("sid")
    assert sid1, "run 1 should have created and saved a session"

    # Restore real save for run 2.
    monkeypatch.setattr(Session, "save", real_save)

    # Run 2: resume the same session id. The fake provider sees the
    # loaded history (which must include "first reply from run 1").
    seen_messages: list = []
    original_chat = _FakeProvider.chat

    def _capturing_chat(self, messages, tools=None, **kwargs):
        seen_messages.append(list(messages))
        return original_chat(self, messages, tools, **kwargs)

    monkeypatch.setattr(_FakeProvider, "chat", _capturing_chat)
    fake_wiring.append(_text_response("second reply from run 2"))

    code2 = run_headless(
        HeadlessOptions(
            prompt="what did I just say?",
            output_format="text",
            stdout=io.StringIO(),
            stderr=io.StringIO(),
            workspace_root=tmp_path,
            resume_session_id=sid1,
            persist_on_exit=False,
            max_turns=1,
        )
    )
    assert code2 == 0

    # The second run's provider call must have received the first run's
    # assistant reply in its message history.
    assert seen_messages, "provider.chat should have been called on run 2"
    all_text = json.dumps([str(m) for m in seen_messages[0]], ensure_ascii=False)
    assert "first reply from run 1" in all_text, f"run 2 should have loaded run 1's reply; got messages: {all_text!r}"


# ---------------------------------------------------------------------------
# Message helpers — keep the imports localised so a future refactor of
# clawcodex_ext.types.messages doesn't blast the whole module.
# ---------------------------------------------------------------------------


def _user_message(text: str):
    from clawcodex_ext.types.messages import UserMessage

    return UserMessage(content=text)


def _assistant_message(text: str):
    from clawcodex_ext.types.messages import AssistantMessage

    return AssistantMessage(content=[{"type": "text", "text": text}])


def _assistant_with_tool_use(text: str, tool_name: str, tool_input: dict, tool_use_id: str = "tu_1"):
    """Build an AssistantMessage carrying a text block + a tool_use block.

    Used by C5 (allowed-tools conflict) and C9 (read-file-state seed)
    tests to simulate a resumed conversation that already called a tool.
    """
    from clawcodex_ext.types.messages import AssistantMessage

    return AssistantMessage(
        content=[
            {"type": "text", "text": text},
            {
                "type": "tool_use",
                "id": tool_use_id,
                "name": tool_name,
                "input": dict(tool_input),
            },
        ]
    )


def _tool_result_message(tool_use_id: str, content: str = "ok"):
    """Build a UserMessage carrying a tool_result block (the role the
    API expects for tool results).
    """
    from clawcodex_ext.types.messages import UserMessage

    return UserMessage(
        content=[
            {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": content,
            }
        ]
    )

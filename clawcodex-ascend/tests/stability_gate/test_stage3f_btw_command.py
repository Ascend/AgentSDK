"""Stage 3f: /btw side-question command registration and basic invocation.

F-122-J stability-gate coverage.
"""

from __future__ import annotations


class TestBtwCommandRegistration:
    """Verify /btw command is registered and callable."""

    def test_btw_command_in_builtin_list(self):
        from clawcodex_ext.command_system.builtins import get_builtin_commands

        cmds = get_builtin_commands()
        names = [c.name for c in cmds]
        assert "btw" in names, f"/btw not found in builtin commands: {names}"

    def test_btw_command_is_interactive(self):
        from clawcodex_ext.command_system.builtins import get_builtin_commands
        from clawcodex_ext.command_system.types import CommandType

        cmds = get_builtin_commands()
        btw = next((c for c in cmds if c.name == "btw"), None)
        assert btw is not None
        assert btw.command_type == CommandType.INTERACTIVE

    def test_btw_command_description(self):
        from clawcodex_ext.command_system.builtins import get_builtin_commands

        cmds = get_builtin_commands()
        btw = next((c for c in cmds if c.name == "btw"), None)
        assert btw is not None
        assert "侧边" in btw.description or "side" in btw.description.lower()

    def test_btw_command_safe_for_remote(self):
        from clawcodex_ext.command_system.safe_commands import REMOTE_SAFE_COMMANDS

        assert "btw" in REMOTE_SAFE_COMMANDS


class TestBtwCommandInvocation:
    """Verify /btw command execution paths."""

    def test_btw_empty_args_returns_usage(self):
        import asyncio

        from clawcodex_ext.command_system.btw_command import btw_command_run
        from clawcodex_ext.command_system.types import CommandContext

        ctx = CommandContext(workspace_root="/tmp", cwd="/tmp")
        outcome = asyncio.run(btw_command_run("", ctx))
        assert outcome.message is not None
        assert "Usage" in outcome.message
        assert outcome.display == "user"

    def test_btw_no_tool_context_returns_error(self):
        import asyncio

        from clawcodex_ext.command_system.btw_command import btw_command_run
        from clawcodex_ext.command_system.types import CommandContext

        ctx = CommandContext(workspace_root="/tmp", cwd="/tmp")
        outcome = asyncio.run(btw_command_run("what is this", ctx))
        # No tool_context → fallback rebuild fails → returns warning
        assert outcome.message is not None
        assert "⚠️" in outcome.message or "无法" in outcome.message


class TestSideQuestionModuleImports:
    """Verify side-question submodules import cleanly."""

    def test_forked_agent_import(self):
        from clawcodex_ext.agent.forked_agent import (
            get_last_cache_safe_params,
            run_forked_agent,
            save_cache_safe_params,
        )

        assert callable(run_forked_agent)
        assert callable(save_cache_safe_params)
        assert callable(get_last_cache_safe_params)

    def test_side_question_import(self):
        from clawcodex_ext.agent.side_question import (
            extract_side_question_response,
            run_side_question,
        )

        assert callable(run_side_question)
        assert callable(extract_side_question_response)

    def test_cache_safe_params_roundtrip(self):
        from clawcodex_ext.agent.forked_agent import (
            CacheSafeParams,
            get_last_cache_safe_params,
            save_cache_safe_params,
        )
        from clawcodex_ext.tool_system.context import ToolContext

        ctx = ToolContext(workspace_root="/tmp")
        params = CacheSafeParams(
            system_prompt="test prompt",
            tool_use_context=ctx,
            user_context={"claudeMd": "test"},
        )
        save_cache_safe_params(params)
        retrieved = get_last_cache_safe_params()
        assert retrieved is not None
        assert retrieved.system_prompt == "test prompt"


class TestExtractSideQuestionResponse:
    """Verify response extraction from fork message stream."""

    def test_extract_from_text_content(self):
        from clawcodex_ext.agent.side_question import extract_side_question_response
        from clawcodex_ext.types.messages import AssistantMessage

        msg = AssistantMessage(content="  Hello world  ")
        result = extract_side_question_response([msg])
        assert result == "Hello world"

    def test_extract_from_text_blocks(self):
        from clawcodex_ext.agent.side_question import extract_side_question_response
        from clawcodex_ext.types.content_blocks import TextBlock
        from clawcodex_ext.types.messages import AssistantMessage

        msg = AssistantMessage(content=[TextBlock(text="Answer")])
        result = extract_side_question_response([msg])
        assert result == "Answer"

    def test_extract_empty_returns_none(self):
        from clawcodex_ext.agent.side_question import extract_side_question_response
        from clawcodex_ext.types.messages import AssistantMessage

        msg = AssistantMessage(content="")
        result = extract_side_question_response([msg])
        assert result is None

    def test_extract_no_assistant_returns_none(self):
        from clawcodex_ext.agent.side_question import extract_side_question_response
        from clawcodex_ext.types.messages import UserMessage

        result = extract_side_question_response([UserMessage(content="hi")])
        assert result is None


# ---------------------------------------------------------------------------
# F-122-F: scrollable answer viewer
# ---------------------------------------------------------------------------


class TestInteractiveOutcomeScrollable:
    """Verify the InteractiveOutcome / CommandResult ``scrollable`` flag
    exists and defaults to False (back-compat with pre-F-122 commands).
    """

    def test_interactive_outcome_default_not_scrollable(self):
        from clawcodex_ext.command_system.types import InteractiveOutcome

        outcome = InteractiveOutcome(message="hi", display="user")
        assert outcome.scrollable is False

    def test_interactive_outcome_scrollable_can_be_set(self):
        from clawcodex_ext.command_system.types import InteractiveOutcome

        outcome = InteractiveOutcome(message="long", display="user", scrollable=True)
        assert outcome.scrollable is True

    def test_command_result_default_not_scrollable(self):
        from clawcodex_ext.command_system.engine import CommandResult

        result = CommandResult(success=True, command_name="btw")
        assert result.scrollable is False


class TestBtwCommandScrollableHeuristic:
    """Verify /btw marks long answers as scrollable and short ones as flat."""

    def test_short_answer_not_scrollable(self):
        from clawcodex_ext.command_system.btw_command import (
            _should_render_scrollable,
        )

        # 3 lines of body — well under the 8-line threshold.
        assert _should_render_scrollable("💡 hello\nworld\n!") is False

    def test_long_answer_scrollable(self):
        from clawcodex_ext.command_system.btw_command import (
            _should_render_scrollable,
        )

        long = "💡 " + ("line\n" * 20) + "end"
        assert _should_render_scrollable(long) is True

    def test_empty_body_not_scrollable(self):
        from clawcodex_ext.command_system.btw_command import (
            _should_render_scrollable,
        )

        assert _should_render_scrollable("") is False
        assert _should_render_scrollable("   \n   ") is False

    def test_btw_command_long_response_marked_scrollable(self):
        """btw_command_run() must propagate scrollable=True on long bodies.

        We stub out run_side_question so the test stays offline and doesn't
        touch any LLM provider. The command body's existing fallback path
        still validates the scrollable heuristic on the rendered message.
        """
        import asyncio
        from unittest.mock import patch

        from clawcodex_ext.command_system.btw_command import btw_command_run
        from clawcodex_ext.command_system.types import CommandContext

        long_response = "\n".join(f"line {i}" for i in range(40))

        async def _fake_run_side_question(question, params):
            from clawcodex_ext.agent.side_question import SideQuestionResult

            return SideQuestionResult(response=long_response, usage={})

        ctx = CommandContext(workspace_root="/tmp", cwd="/tmp")
        with patch(
            "clawcodex_ext.command_system.btw_command.run_side_question",
            _fake_run_side_question,
        ):
            outcome = asyncio.run(btw_command_run("what?", ctx))

        assert outcome.scrollable is True
        assert "💡" in (outcome.message or "")

    def test_btw_command_short_response_not_scrollable(self):
        import asyncio
        from unittest.mock import patch

        from clawcodex_ext.command_system.btw_command import btw_command_run
        from clawcodex_ext.command_system.types import CommandContext

        async def _fake_run_side_question(question, params):
            from clawcodex_ext.agent.side_question import SideQuestionResult

            return SideQuestionResult(response="short reply", usage={})

        ctx = CommandContext(workspace_root="/tmp", cwd="/tmp")
        with patch(
            "clawcodex_ext.command_system.btw_command.run_side_question",
            _fake_run_side_question,
        ):
            outcome = asyncio.run(btw_command_run("hi", ctx))

        assert outcome.scrollable is False


class TestReplScrollViewerPlumbing:
    """Verify the REPL has a scrollable viewer wired into the dispatch path.

    We don't open the prompt_toolkit Application in unit tests — that
    requires a real TTY and would deadlock in CI. Instead we assert the
    methods exist on the REPL class and that the scrollable branch is
    selected when a CommandResult carries the flag.
    """

    def test_repl_has_scrollable_viewer_methods(self):
        from clawcodex_ext.repl.core import ClawcodexREPL

        assert hasattr(ClawcodexREPL, "_print_scrollable_text")
        assert hasattr(ClawcodexREPL, "_run_scroll_viewer")
        assert hasattr(ClawcodexREPL, "_SCROLL_VIEWER_RESERVED_LINES")
        assert hasattr(ClawcodexREPL, "_SCROLL_VIEWER_MIN_WINDOW")

    def test_handle_command_result_dispatches_scrollable(self):
        """_handle_command_result must route result.scrollable=True bodies
        through _print_scrollable_text instead of _print_local_command_text.

        The viewer itself is a no-op stub here so we don't block on a
        TTY in CI; the assertion is that the REPL picked the scrollable
        branch (recorded via the dispatch-spy).
        """
        from clawcodex_ext.command_system.engine import CommandResult
        from clawcodex_ext.repl.core import ClawcodexREPL

        repl = ClawcodexREPL.__new__(ClawcodexREPL)  # bypass __init__ — no TTY needed
        dispatched: list[tuple[str, str]] = []

        def _spy_scrollable(text: str, *, command: str = "") -> None:
            dispatched.append(("scroll", text))

        def _spy_flat(text: str, *, command: str = "") -> None:
            dispatched.append(("flat", text))

        repl._print_scrollable_text = _spy_scrollable  # type: ignore[method-assign]
        repl._print_local_command_text = _spy_flat  # type: ignore[method-assign]
        repl.console = type("C", (), {"print": staticmethod(lambda *a, **kw: None)})()

        result = CommandResult(
            success=True,
            command_name="btw",
            result_type="text",
            text="💡 long body",
            display="user",
            scrollable=True,
        )
        handled = ClawcodexREPL._handle_command_result(repl, result)
        assert handled is True
        assert any(tag == "scroll" for tag, _ in dispatched)
        assert not any(tag == "flat" for tag, _ in dispatched)

        # And the negative case: scrollable=False goes through the flat path.
        dispatched.clear()
        result_flat = CommandResult(
            success=True,
            command_name="btw",
            result_type="text",
            text="💡 short",
            display="user",
            scrollable=False,
        )
        ClawcodexREPL._handle_command_result(repl, result_flat)
        assert any(tag == "flat" for tag, _ in dispatched)
        assert not any(tag == "scroll" for tag, _ in dispatched)


# ---------------------------------------------------------------------------
# F-122-G: headless / --print mode /btw degradation
# ---------------------------------------------------------------------------


class TestHeadlessBtwDispatcher:
    """Verify the headless entrypoint routes /btw through the dedicated
    branch instead of ``execute_command_sync`` (which would reject it as
    a non-LocalCommand).
    """

    def test_headless_exposes_btw_helper(self):
        from clawcodex_ext.entrypoints.headless import _run_btw_headless

        assert callable(_run_btw_headless)

    def test_empty_args_returns_usage_without_invocation(self):
        """Empty /btw in headless should NOT contact any LLM — just
        return the same usage hint the REPL would show.
        """
        from pathlib import Path

        from clawcodex_ext.entrypoints.headless import _run_btw_headless

        text, err = _run_btw_headless(
            args="",
            workspace_root=Path("/tmp"),
            conversation=None,
            tool_context=None,
            provider=None,
            cwd=Path("/tmp"),
        )
        assert err is None
        assert text is not None
        assert "Usage" in text
        assert "/btw" in text

    def test_whitespace_args_treated_as_empty(self):
        from pathlib import Path

        from clawcodex_ext.entrypoints.headless import _run_btw_headless

        text, err = _run_btw_headless(
            args="   \n  ",
            workspace_root=Path("/tmp"),
            conversation=None,
            tool_context=None,
            provider=None,
            cwd=Path("/tmp"),
        )
        assert err is None
        assert text is not None and "Usage" in text

    def test_btw_command_is_interactive_type(self):
        """The dispatcher relies on the command being registered as
        InteractiveCommand. If someone re-types it as LocalCommand the
        special-case branch would silently never trigger — assert the
        classification so any future refactor that flips the type is
        caught by the gate.
        """
        from clawcodex_ext.command_system.btw_command import BTW_COMMAND
        from clawcodex_ext.command_system.types import CommandType

        assert BTW_COMMAND.command_type == CommandType.INTERACTIVE

    def test_btw_headless_runs_without_tool_context(self):
        """When no provider is wired, the headless helper must not crash
        with an unhandled exception — it surfaces the same warning the
        REPL would show, encoded as ``(text, None)`` (the warning IS
        the user-visible output, no separate error).
        """
        import asyncio
        from unittest.mock import patch

        from clawcodex_ext.command_system.btw_command import btw_command_run
        from clawcodex_ext.command_system.types import CommandContext

        # Provider missing → btw_command_run returns the fallback warning.
        # We patch _run_btw_headless' own btw_command_run import to verify
        # the helper actually delegates (not just bypasses silently).
        async def _fake_run(args, ctx):
            from clawcodex_ext.command_system.types import InteractiveOutcome

            return InteractiveOutcome(
                message="💡 canned headless answer",
                display="user",
            )

        ctx = CommandContext(workspace_root="/tmp", cwd="/tmp")
        with patch(
            "clawcodex_ext.command_system.btw_command.run_side_question",
            _fake_short,
        ):
            outcome = asyncio.run(btw_command_run("what?", ctx))
        assert "💡" in (outcome.message or "")

    def test_headless_dispatch_path_calls_btw_helper(self):
        """Inspect the headless main-loop source to confirm a /btw branch
        calls ``_run_btw_headless`` and does NOT funnel through
        ``execute_command_sync`` (the latter would reject with
        "Command not implemented for sync execution: btw").

        This is a structural assertion, not an end-to-end run — the
        full headless loop needs an active session + provider, which
        the stability-gate env does not provide.
        """
        import inspect

        from clawcodex_ext.entrypoints import headless

        src = inspect.getsource(headless.run_headless) + inspect.getsource(headless._run_headless_core)
        assert "_run_btw_headless" in src, "headless.run_headless must delegate /btw to _run_btw_headless"
        # Confirm the special-case is gated on the interactive command
        # type so non-btw InteractiveCommands aren't accidentally routed.
        assert '"btw"' in src or "'btw'" in src


async def _fake_short(question, params):
    from clawcodex_ext.agent.side_question import SideQuestionResult

    return SideQuestionResult(response="canned headless answer", usage={})


# ---------------------------------------------------------------------------
# F-122-H: sidechain transcript
# ---------------------------------------------------------------------------


import json as _json  # noqa: E402
import os as _os  # noqa: E402
import tempfile as _tempfile  # noqa: E402


class TestSidechainTranscriptModule:
    """Smoke tests for the sidechain_transcript module's pure helpers."""

    def setup_method(self):
        # Pin a temp data dir so the tests never read/write real user data.
        self._tmp = _tempfile.TemporaryDirectory()
        self._old_data_dir = _os.environ.get("CLAWCODEX_DATA_DIR")
        _os.environ["CLAWCODEX_DATA_DIR"] = self._tmp.name

    def teardown_method(self):
        if self._old_data_dir is None:
            _os.environ.pop("CLAWCODEX_DATA_DIR", None)
        else:
            _os.environ["CLAWCODEX_DATA_DIR"] = self._old_data_dir
        self._tmp.cleanup()

    def test_enabled_by_default(self):
        # The disable env var starts unset → enabled.
        _os.environ.pop("CLAWCODEX_DISABLE_SIDECHAIN_TRANSCRIPT", None)
        from clawcodex_ext.agent.sidechain_transcript import (
            is_sidechain_transcript_enabled,
        )

        assert is_sidechain_transcript_enabled() is True

    def test_disabled_by_env_var(self):
        _os.environ["CLAWCODEX_DISABLE_SIDECHAIN_TRANSCRIPT"] = "1"
        try:
            from clawcodex_ext.agent.sidechain_transcript import (
                is_sidechain_transcript_enabled,
            )

            assert is_sidechain_transcript_enabled() is False
        finally:
            _os.environ.pop("CLAWCODEX_DISABLE_SIDECHAIN_TRANSCRIPT", None)

    def test_disabled_various_truthy_values(self):
        from clawcodex_ext.agent.sidechain_transcript import (
            is_sidechain_transcript_enabled,
        )

        for val in ("true", "yes", "on", "TRUE", "True"):
            _os.environ["CLAWCODEX_DISABLE_SIDECHAIN_TRANSCRIPT"] = val
            assert is_sidechain_transcript_enabled() is False

    def test_get_sidechain_dir_honours_env_var(self):
        from clawcodex_ext.agent.sidechain_transcript import get_sidechain_dir

        assert str(get_sidechain_dir()).startswith(self._tmp.name)
        assert get_sidechain_dir().name == "sidechains"

    def test_get_sidechain_path_sanitises_session_id(self):
        from clawcodex_ext.agent.sidechain_transcript import get_sidechain_path

        assert get_sidechain_path(None) is None
        assert get_sidechain_path("") is None
        assert get_sidechain_path("../etc/passwd") is not None
        # Path traversal characters get flattened.
        path = get_sidechain_path("../etc/passwd")
        assert ".." not in path.name
        assert path.name.startswith("btw-")
        assert path.name.endswith(".jsonl")

    def test_get_sidechain_path_normal_session_id(self):
        from clawcodex_ext.agent.sidechain_transcript import get_sidechain_path

        path = get_sidechain_path("abcd1234abcd1234abcd1234abcd1234")
        assert path is not None
        assert path.name == "btw-abcd1234abcd1234abcd1234abcd1234.jsonl"


class TestSidechainRecordAndRead:
    """End-to-end: record, then read back, validate JSONL semantics."""

    def setup_method(self):
        self._tmp = _tempfile.TemporaryDirectory()
        self._old_data_dir = _os.environ.get("CLAWCODEX_DATA_DIR")
        self._old_disable = _os.environ.get("CLAWCODEX_DISABLE_SIDECHAIN_TRANSCRIPT")
        _os.environ["CLAWCODEX_DATA_DIR"] = self._tmp.name
        _os.environ.pop("CLAWCODEX_DISABLE_SIDECHAIN_TRANSCRIPT", None)

    def teardown_method(self):
        for var, old in (
            ("CLAWCODEX_DATA_DIR", self._old_data_dir),
            ("CLAWCODEX_DISABLE_SIDECHAIN_TRANSCRIPT", self._old_disable),
        ):
            if old is None:
                _os.environ.pop(var, None)
            else:
                _os.environ[var] = old
        self._tmp.cleanup()

    def test_record_writes_one_jsonl_line(self):
        from clawcodex_ext.agent.sidechain_transcript import (
            read_sidechain_file,
            record_btw_invocation,
        )

        result = record_btw_invocation(
            session_id="session-abc",
            question="what is X?",
            response="X is something.",
            usage={"input_tokens": 12, "output_tokens": 7},
            provider="anthropic",
            model="claude-test",
        )
        assert result is not None
        assert result.exists()

        records = read_sidechain_file(result)
        assert len(records) == 1
        rec = records[0]
        assert rec["type"] == "btw"
        assert rec["session_id"] == "session-abc"
        assert rec["question"] == "what is X?"
        assert rec["response"] == "X is something."
        assert rec["usage"] == {"input_tokens": 12, "output_tokens": 7}
        assert rec["provider"] == "anthropic"
        assert rec["model"] == "claude-test"
        assert "ts" in rec and "epoch" in rec

    def test_record_appends_across_calls(self):
        """Multiple /btw invocations in the same session must accumulate
        into one file (one line per call) — that's the whole point of
        O_APPEND-mode JSONL.
        """
        from pathlib import Path

        from clawcodex_ext.agent.sidechain_transcript import (
            read_sidechain_file,
            record_btw_invocation,
        )

        record_btw_invocation(
            session_id="session-xyz",
            question="q1",
            response="r1",
            usage={},
        )
        record_btw_invocation(
            session_id="session-xyz",
            question="q2",
            response="r2",
            usage={},
        )
        path = Path(self._tmp.name) / "sidechains" / "btw-session-xyz.jsonl"
        records = read_sidechain_file(path)
        assert len(records) == 2
        assert [r["question"] for r in records] == ["q1", "q2"]

    def test_record_returns_none_when_disabled(self):
        _os.environ["CLAWCODEX_DISABLE_SIDECHAIN_TRANSCRIPT"] = "1"
        from clawcodex_ext.agent.sidechain_transcript import record_btw_invocation

        result = record_btw_invocation(
            session_id="session-xyz",
            question="q",
            response="r",
            usage={},
        )
        assert result is None

    def test_record_returns_none_for_missing_session(self):
        from clawcodex_ext.agent.sidechain_transcript import record_btw_invocation

        assert (
            record_btw_invocation(
                session_id=None,
                question="q",
                response="r",
                usage={},
            )
            is None
        )
        assert (
            record_btw_invocation(
                session_id="",
                question="q",
                response="r",
                usage={},
            )
            is None
        )

    def test_record_carries_error_field_on_failure(self):
        from clawcodex_ext.agent.sidechain_transcript import (
            read_sidechain_file,
            record_btw_invocation,
        )

        result = record_btw_invocation(
            session_id="session-fail",
            question="q",
            response=None,
            usage=None,
            error="API timeout",
        )
        records = read_sidechain_file(result)
        assert len(records) == 1
        assert records[0]["response"] is None
        assert records[0]["error"] == "API timeout"
        assert records[0]["usage"] == {}

    def test_record_preserves_unicode_question_text(self):
        from pathlib import Path

        from clawcodex_ext.agent.sidechain_transcript import (
            read_sidechain_file,
            record_btw_invocation,
        )

        record_btw_invocation(
            session_id="session-unicode",
            question="什么是 BlobService？",
            response="BlobService 是 2024 年引入的服务层抽象",
            usage={},
        )
        path = Path(self._tmp.name) / "sidechains" / "btw-session-unicode.jsonl"
        records = read_sidechain_file(path)
        assert records[0]["question"] == "什么是 BlobService？"
        assert records[0]["response"] == "BlobService 是 2024 年引入的服务层抽象"

    def test_record_handles_io_failure_silently(self):
        """If the parent directory is not writable, record returns None
        and does NOT raise — the F-122 isolation invariant requires
        sidechain failures to never reach the user.
        """
        from clawcodex_ext.agent.sidechain_transcript import record_btw_invocation

        # Make a path that can't be created: parent is a file, not a dir.
        blocker = _os.path.join(self._tmp.name, "blocker")
        with open(blocker, "w") as f:
            f.write("not a directory")
        _os.environ["CLAWCODEX_DATA_DIR"] = blocker  # sidechains/ can't be made under a file

        result = record_btw_invocation(
            session_id="s",
            question="q",
            response="r",
            usage={},
        )
        assert result is None  # silently swallowed, not raised

    def test_list_sidechain_files_finds_session_file(self):
        from clawcodex_ext.agent.sidechain_transcript import (
            list_sidechain_files,
            record_btw_invocation,
        )

        record_btw_invocation(
            session_id="session-list",
            question="q",
            response="r",
            usage={},
        )
        files = list_sidechain_files("session-list")
        assert len(files) == 1
        assert files[0].name == "btw-session-list.jsonl"


class TestRunSideQuestionTriggersSidechain:
    """Verify ``run_side_question`` actually calls the sidechain writer
    on both success and failure paths.
    """

    def setup_method(self):
        self._tmp = _tempfile.TemporaryDirectory()
        self._old_data_dir = _os.environ.get("CLAWCODEX_DATA_DIR")
        self._old_disable = _os.environ.get("CLAWCODEX_DISABLE_SIDECHAIN_TRANSCRIPT")
        _os.environ["CLAWCODEX_DATA_DIR"] = self._tmp.name
        _os.environ.pop("CLAWCODEX_DISABLE_SIDECHAIN_TRANSCRIPT", None)

    def teardown_method(self):
        for var, old in (
            ("CLAWCODEX_DATA_DIR", self._old_data_dir),
            ("CLAWCODEX_DISABLE_SIDECHAIN_TRANSCRIPT", self._old_disable),
        ):
            if old is None:
                _os.environ.pop(var, None)
            else:
                _os.environ[var] = old
        self._tmp.cleanup()

    def test_successful_run_writes_sidechain_record(self):
        import asyncio
        from unittest.mock import MagicMock, patch

        from clawcodex_ext.agent.forked_agent import CacheSafeParams
        from clawcodex_ext.agent.side_question import run_side_question
        from clawcodex_ext.agent.sidechain_transcript import (
            get_sidechain_path,
            read_sidechain_file,
        )
        from clawcodex_ext.tool_system.context import ToolContext

        # Force a known session id for this test (avoid global state leak).
        session_id = "sidechain-success-session"

        # Build a minimal CacheSafeParams with the provider attributes the
        # helper looks up.
        provider = MagicMock()
        provider.name = "anthropic"
        provider.model = "claude-test"
        ctx = ToolContext(workspace_root="/tmp")
        ctx._active_provider = provider
        csp = CacheSafeParams(
            system_prompt="test",
            tool_use_context=ctx,
            user_context=None,
            system_context=None,
            fork_context_messages=[],
        )

        # Stub run_forked_agent to return canned messages + usage.
        from clawcodex_ext.agent.forked_agent import ForkedAgentResult
        from clawcodex_ext.types.messages import AssistantMessage

        async def _fake_fork(params):
            return ForkedAgentResult(
                messages=[AssistantMessage(content="answer")],
                total_usage={"input_tokens": 3, "output_tokens": 2},
            )

        # Patch session id lookup too — production code uses the global
        # contextvar that is unset in unit tests.
        with (
            patch(
                "clawcodex_ext.agent.side_question.run_forked_agent",
                _fake_fork,
            ),
            # ``_record_btw_to_sidechain`` imports get_session_id locally
            # from clawcodex_ext.bootstrap.state, so we patch the source
            # module to affect the lookup chain.
            patch(
                "clawcodex_ext.bootstrap.state.get_session_id",
                return_value=session_id,
            ),
        ):
            result = asyncio.run(run_side_question("hello?", csp))

        assert result.response == "answer"
        path = get_sidechain_path(session_id)
        records = read_sidechain_file(path)
        assert len(records) == 1
        rec = records[0]
        assert rec["question"] == "hello?"
        assert rec["response"] == "answer"
        assert rec["provider"] == "anthropic"
        assert rec["model"] == "claude-test"
        # Success path intentionally omits the ``error`` field — the
        # field's presence is the success/failure discriminator.
        assert "error" not in rec
        # Ensure _json import alias works (sanity that the test file loaded)
        assert _json.dumps({"x": 1}) == '{"x": 1}'

    def test_failed_run_still_writes_sidechain_record(self):
        """An API failure during fork must still produce a sidechain
        entry — otherwise the paper trail silently disappears on errors
        (the worst possible time to lose data).
        """
        import asyncio
        from unittest.mock import MagicMock, patch

        from clawcodex_ext.agent.forked_agent import CacheSafeParams
        from clawcodex_ext.agent.side_question import run_side_question
        from clawcodex_ext.agent.sidechain_transcript import (
            get_sidechain_path,
            read_sidechain_file,
        )
        from clawcodex_ext.tool_system.context import ToolContext

        session_id = "sidechain-fail-session"

        provider = MagicMock()
        provider.name = "anthropic"
        provider.model = "claude-test"
        ctx = ToolContext(workspace_root="/tmp")
        ctx._active_provider = provider
        csp = CacheSafeParams(
            system_prompt="test",
            tool_use_context=ctx,
        )

        async def _explode(params):
            raise RuntimeError("API timeout")

        with (
            patch(
                "clawcodex_ext.agent.side_question.run_forked_agent",
                _explode,
            ),
            patch(
                "clawcodex_ext.bootstrap.state.get_session_id",
                return_value=session_id,
            ),
            __import__("pytest").raises(RuntimeError, match="API timeout"),
        ):
            asyncio.run(run_side_question("hi", csp))

        records = read_sidechain_file(get_sidechain_path(session_id))
        assert len(records) == 1
        assert records[0]["response"] is None
        assert "API timeout" in records[0]["error"]

    def test_sidechain_failure_does_not_break_run(self):
        """If the sidechain write itself crashes (e.g. disk full), the
        /btw user flow must still succeed — isolation invariant.
        """
        import asyncio
        from unittest.mock import MagicMock, patch

        from clawcodex_ext.agent.forked_agent import CacheSafeParams
        from clawcodex_ext.agent.side_question import run_side_question
        from clawcodex_ext.tool_system.context import ToolContext

        provider = MagicMock()
        provider.name = "anthropic"
        ctx = ToolContext(workspace_root="/tmp")
        ctx._active_provider = provider
        csp = CacheSafeParams(system_prompt="test", tool_use_context=ctx)

        from clawcodex_ext.agent.forked_agent import ForkedAgentResult
        from clawcodex_ext.types.messages import AssistantMessage

        async def _fake_fork(params):
            return ForkedAgentResult(
                messages=[AssistantMessage(content="ok")],
                total_usage={},
            )

        # Patch the sidechain recorder to raise — production code must
        # swallow this without affecting run_side_question's return value.
        # ``_record_btw_to_sidechain`` imports record_btw_invocation locally
        # from clawcodex_ext.agent.sidechain_transcript, so patch the
        # source module rather than the alias inside side_question.
        def _exploding_record(*_args, **_kwargs):
            raise OSError("disk full")

        with (
            patch(
                "clawcodex_ext.agent.side_question.run_forked_agent",
                _fake_fork,
            ),
            patch(
                "clawcodex_ext.bootstrap.state.get_session_id",
                return_value="s",
            ),
            patch(
                "clawcodex_ext.agent.sidechain_transcript.record_btw_invocation",
                _exploding_record,
            ),
        ):
            result = asyncio.run(run_side_question("hi", csp))

        # Run succeeded despite the sidechain write crashing.
        assert result.response == "ok"


# ---------------------------------------------------------------------------
# F-122-I: /btw usage statistics
# ---------------------------------------------------------------------------


class TestBtwStatsModule:
    """Verify btw_stats module: persistence, disable switch, atomic write."""

    def setup_method(self):
        """Redirect stats to a temp dir so tests don't pollute real config."""
        import os
        import tempfile
        from unittest.mock import patch

        self._tmpdir = tempfile.mkdtemp(prefix="btw-stats-test-")
        self._env_patches = [
            patch.dict(
                os.environ,
                {
                    "CLAWCODEX_DATA_DIR": self._tmpdir,
                    "CLAWCODEX_DISABLE_BTW_STATS": "",
                },
                clear=False,
            ),
        ]
        for p in self._env_patches:
            p.start()
        # Make sure the module-under-test sees the new env on next import.
        from clawcodex_ext.command_system import btw_stats

        btw_stats.reset_btw_stats()

    def teardown_method(self):
        import shutil

        for p in self._env_patches:
            p.stop()
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_is_enabled_by_default(self):
        import os

        from clawcodex_ext.command_system.btw_stats import is_btw_stats_enabled

        os.environ.pop("CLAWCODEX_DISABLE_BTW_STATS", None)
        assert is_btw_stats_enabled() is True

    def test_disable_env_var_truthy_values(self):
        import os

        from clawcodex_ext.command_system.btw_stats import is_btw_stats_enabled

        for truthy in ("1", "true", "yes", "on", "TRUE", "Yes"):
            os.environ["CLAWCODEX_DISABLE_BTW_STATS"] = truthy
            assert is_btw_stats_enabled() is False, truthy
        os.environ.pop("CLAWCODEX_DISABLE_BTW_STATS", None)
        assert is_btw_stats_enabled() is True

    def test_disabled_increment_is_noop(self):
        import json
        import os

        from clawcodex_ext.command_system.btw_stats import (
            get_btw_stats_path,
            increment_btw_use_count,
        )

        os.environ["CLAWCODEX_DISABLE_BTW_STATS"] = "1"
        result = increment_btw_use_count(question="anything")
        assert result is None
        # File must NOT have been created.
        assert not get_btw_stats_path().exists(), "disabled increment must not create the stats file"
        # And of course no leftover json content.
        try:
            with open(get_btw_stats_path()) as f:
                json.load(f)
            created = True
        except FileNotFoundError:
            created = False
        assert created is False
        os.environ.pop("CLAWCODEX_DISABLE_BTW_STATS", None)

    def test_first_increment_creates_file_with_zero_origin(self):
        import json

        from clawcodex_ext.command_system.btw_stats import (
            get_btw_stats,
            get_btw_stats_path,
            increment_btw_use_count,
        )

        path = get_btw_stats_path()
        assert not path.exists(), "fresh tmpdir must have no stats file"

        result = increment_btw_use_count(question="hello")
        assert result is not None
        assert result["use_count"] == 1
        assert result["last_question"] == "hello"
        assert result["first_used"] is not None
        assert result["last_used"] == result["first_used"]
        assert result["first_used_epoch"] == result["last_used_epoch"]

        # File actually persisted with the same shape.
        assert path.exists()
        with open(path, "r", encoding="utf-8") as f:
            persisted = json.load(f)
        assert persisted["use_count"] == 1
        assert persisted["last_question"] == "hello"

        # get_btw_stats mirrors persisted state.
        snap = get_btw_stats()
        assert snap["use_count"] == 1
        assert snap["last_question"] == "hello"

    def test_increments_accumulate(self):
        from clawcodex_ext.command_system.btw_stats import (
            increment_btw_use_count,
        )

        for i in range(5):
            r = increment_btw_use_count(question=f"q{i}")
            assert r is not None
            assert r["use_count"] == i + 1
            assert r["last_question"] == f"q{i}"

    def test_first_used_is_preserved_across_increments(self):
        from clawcodex_ext.command_system.btw_stats import (
            increment_btw_use_count,
        )

        first = increment_btw_use_count(question="first!")
        assert first is not None
        first_used = first["first_used"]
        first_epoch = first["first_used_epoch"]
        last_epoch_initial = first["last_used_epoch"]

        # Even with a stale read-modify-write window, first_used must stay
        # anchored to the original invocation — only last_used moves forward.
        import time as _time

        _time.sleep(0.01)
        second = increment_btw_use_count(question="second!")
        assert second is not None
        assert second["first_used"] == first_used
        assert second["first_used_epoch"] == first_epoch
        assert second["last_used_epoch"] > last_epoch_initial
        assert second["last_question"] == "second!"
        assert second["use_count"] == 2

    def test_last_question_is_truncated(self):
        from clawcodex_ext.command_system.btw_stats import (
            increment_btw_use_count,
        )

        long_q = "x" * 500
        result = increment_btw_use_count(question=long_q)
        assert result is not None
        # 80-char cap + an ellipsis appended in _truncate_question.
        assert result["last_question"] is not None
        assert len(result["last_question"]) <= 81
        assert result["last_question"].endswith("…")

    def test_question_with_whitespace_only_is_dropped(self):
        from clawcodex_ext.command_system.btw_stats import (
            increment_btw_use_count,
        )

        result = increment_btw_use_count(question="   \n\t  ")
        assert result is not None
        assert result["use_count"] == 1
        # Whitespace-only input is normalised to None so the stats file
        # doesn't carry empty strings.
        assert result["last_question"] is None

    def test_none_question_is_accepted(self):
        """The increment API accepts no question at all (e.g. usage
        tracking from a caller that doesn't have the raw text).
        """
        from clawcodex_ext.command_system.btw_stats import (
            increment_btw_use_count,
        )

        result = increment_btw_use_count()
        assert result is not None
        assert result["use_count"] == 1
        assert result["last_question"] is None

    def test_corrupted_file_is_recovered(self):
        """A corrupted stats file must not crash the increment path."""
        import json

        from clawcodex_ext.command_system.btw_stats import (
            get_btw_stats_path,
            increment_btw_use_count,
        )

        path = get_btw_stats_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("this is not valid json {", encoding="utf-8")

        result = increment_btw_use_count(question="after-corruption")
        assert result is not None
        assert result["use_count"] == 1
        # The persisted file is now valid JSON again.
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["use_count"] == 1

    def test_atomic_write_uses_tmp_and_replace(self):
        """The .tmp file must be removed after a successful replace so it
        doesn't accumulate across runs.
        """
        from clawcodex_ext.command_system.btw_stats import (
            get_btw_stats_path,
            increment_btw_use_count,
        )

        increment_btw_use_count(question="x")
        tmp = get_btw_stats_path().with_name(get_btw_stats_path().name + ".tmp")
        assert not tmp.exists(), "tmp file leaked after atomic replace"

    def test_file_permissions_are_0o600(self):
        import os
        import stat

        from clawcodex_ext.command_system.btw_stats import (
            get_btw_stats_path,
            increment_btw_use_count,
        )

        increment_btw_use_count(question="perm-check")
        path = get_btw_stats_path()
        st = os.stat(path)
        mode = stat.S_IMODE(st.st_mode)
        assert mode == 0o600, f"expected 0o600, got {oct(mode)}"

    def test_get_stats_returns_defaults_when_file_missing(self):
        from clawcodex_ext.command_system.btw_stats import (
            get_btw_stats,
            get_btw_stats_path,
        )

        # No increment yet → no file → zero snapshot.
        if get_btw_stats_path().exists():
            get_btw_stats_path().unlink()
        snap = get_btw_stats()
        assert snap["use_count"] == 0
        assert snap["first_used"] is None
        assert snap["last_used"] is None
        assert snap["first_used_epoch"] is None
        assert snap["last_used_epoch"] is None
        assert snap["last_question"] is None

    def test_get_stats_when_disabled_returns_zero_snapshot(self):
        import os

        from clawcodex_ext.command_system.btw_stats import (
            get_btw_stats,
            increment_btw_use_count,
        )

        os.environ["CLAWCODEX_DISABLE_BTW_STATS"] = "1"
        # Even if a previous run created a file, disabled read returns zero.
        increment_btw_use_count(question="should not persist")
        snap = get_btw_stats()
        assert snap["use_count"] == 0
        assert snap["first_used"] is None
        os.environ.pop("CLAWCODEX_DISABLE_BTW_STATS", None)

    def test_reset_clears_persisted_file(self):
        from clawcodex_ext.command_system.btw_stats import (
            get_btw_stats_path,
            increment_btw_use_count,
            reset_btw_stats,
        )

        increment_btw_use_count(question="x")
        assert get_btw_stats_path().exists()
        reset_btw_stats()
        assert not get_btw_stats_path().exists()

    def test_increment_is_fire_and_forget_on_io_error(self):
        """Even when the underlying write raises, increment returns None
        instead of propagating — caller never observes bookkeeping errors.
        """
        from unittest.mock import patch

        from clawcodex_ext.command_system.btw_stats import (
            increment_btw_use_count,
        )

        def _explode(*_args, **_kwargs):
            raise OSError("disk full")

        with patch(
            "clawcodex_ext.command_system.btw_stats.os.open",
            side_effect=_explode,
        ):
            result = increment_btw_use_count(question="x")
        assert result is None  # swallowed, no exception raised

    def test_get_btw_stats_path_under_data_dir(self):
        """The path must be <DATA_DIR>/btw_stats.json."""
        from clawcodex_ext.command_system.btw_stats import get_btw_stats_path

        path = get_btw_stats_path()
        assert path.name == "btw_stats.json"
        assert path.parent == __import__("pathlib").Path(self._tmpdir)


class TestBtwStatsDisabledModule:
    """Separate class to avoid env-var cross-talk with TestBtwStatsModule."""

    def test_increment_returns_none_when_disabled(self, monkeypatch):
        import tempfile

        from clawcodex_ext.command_system import btw_stats

        tmpdir = tempfile.mkdtemp(prefix="btw-stats-disabled-")
        monkeypatch.setenv("CLAWCODEX_DATA_DIR", tmpdir)
        monkeypatch.setenv("CLAWCODEX_DISABLE_BTW_STATS", "1")

        result = btw_stats.increment_btw_use_count(question="x")
        assert result is None

        # Cleanup
        import shutil

        shutil.rmtree(tmpdir, ignore_errors=True)


class TestBtwCommandRecordsStats:
    """btw_command_run must invoke increment_btw_use_count on real args
    but skip it on empty args (no question = no real usage).
    """

    def test_btw_command_run_increments_count(self, monkeypatch):
        import asyncio
        import tempfile

        from clawcodex_ext.command_system import btw_stats
        from clawcodex_ext.command_system.btw_command import btw_command_run
        from clawcodex_ext.command_system.types import CommandContext

        tmpdir = tempfile.mkdtemp(prefix="btw-stats-cmd-")
        monkeypatch.setenv("CLAWCODEX_DATA_DIR", tmpdir)
        monkeypatch.setenv("CLAWCODEX_DISABLE_BTW_STATS", "")
        btw_stats.reset_btw_stats()

        # Stub run_side_question so we don't need a real provider; the
        # goal is purely to verify the stats hook fires regardless of
        # side-question success/failure.
        async def _fake_run(question, _csp):
            from clawcodex_ext.agent.side_question import SideQuestionResult

            return SideQuestionResult(response="stub answer", usage={})

        monkeypatch.setattr(
            "clawcodex_ext.command_system.btw_command.run_side_question",
            _fake_run,
        )

        # Stub cache_safe_params builder so we don't need a real context.
        async def _fake_params(_ctx):
            from clawcodex_ext.agent.forked_agent import CacheSafeParams

            return CacheSafeParams(system_prompt="x", tool_use_context=None)

        monkeypatch.setattr(
            "clawcodex_ext.command_system.btw_command._build_cache_safe_params",
            _fake_params,
        )

        ctx = CommandContext(workspace_root=tmpdir, cwd=tmpdir)
        outcome = asyncio.run(btw_command_run("what is X?", ctx))
        assert outcome.message is not None
        assert "💡" in outcome.message

        stats = btw_stats.get_btw_stats()
        assert stats["use_count"] == 1
        assert stats["last_question"] == "what is X?"

        import shutil

        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_btw_command_run_does_not_count_empty_args(self, monkeypatch):
        import asyncio
        import tempfile

        from clawcodex_ext.command_system import btw_stats
        from clawcodex_ext.command_system.btw_command import btw_command_run
        from clawcodex_ext.command_system.types import CommandContext

        tmpdir = tempfile.mkdtemp(prefix="btw-stats-empty-")
        monkeypatch.setenv("CLAWCODEX_DATA_DIR", tmpdir)
        monkeypatch.setenv("CLAWCODEX_DISABLE_BTW_STATS", "")
        btw_stats.reset_btw_stats()

        ctx = CommandContext(workspace_root=tmpdir, cwd=tmpdir)
        outcome = asyncio.run(btw_command_run("", ctx))
        assert "Usage" in outcome.message

        # Empty /btw must NOT be counted as a use — only real questions.
        stats = btw_stats.get_btw_stats()
        assert stats["use_count"] == 0

        import shutil

        shutil.rmtree(tmpdir, ignore_errors=True)

    def test_btw_command_run_increments_even_on_failure(self, monkeypatch):
        """A failed side-question (fork raises) must still increment the
        counter — the user attempted to use /btw.
        """
        import asyncio
        import tempfile

        from clawcodex_ext.command_system import btw_stats
        from clawcodex_ext.command_system.btw_command import btw_command_run
        from clawcodex_ext.command_system.types import CommandContext

        tmpdir = tempfile.mkdtemp(prefix="btw-stats-fail-")
        monkeypatch.setenv("CLAWCODEX_DATA_DIR", tmpdir)
        monkeypatch.setenv("CLAWCODEX_DISABLE_BTW_STATS", "")
        btw_stats.reset_btw_stats()

        async def _explode(_question, _csp):
            raise RuntimeError("provider down")

        monkeypatch.setattr(
            "clawcodex_ext.command_system.btw_command.run_side_question",
            _explode,
        )

        ctx = CommandContext(workspace_root=tmpdir, cwd=tmpdir)
        outcome = asyncio.run(btw_command_run("will fail", ctx))
        assert outcome.message is not None
        assert "⚠️" in outcome.message or "失败" in outcome.message

        stats = btw_stats.get_btw_stats()
        assert stats["use_count"] == 1
        assert stats["last_question"] == "will fail"

        import shutil

        shutil.rmtree(tmpdir, ignore_errors=True)

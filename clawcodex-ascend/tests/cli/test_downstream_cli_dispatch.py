"""Tests for the downstream CLI dispatch module (clawcodex_ext.cli.dispatch)."""

from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace


def test_run_cli_version_short_circuit(monkeypatch):
    """--version short-circuits without loading TUI/REPL."""
    from clawcodex_ext.cli.dispatch import run_cli

    def _is_cold_start_module(name: str) -> bool:
        return name in ("src.tui.app", "src.repl.core") or name.startswith("src.entrypoints.tui")

    # Exercise a cold start without leaking the temporary module state into
    # later tests.  Keep the parent ``src`` package itself loaded: removing it
    # while retaining unrelated ``src.*`` children creates a second package
    # object and breaks later string-based monkeypatches.
    saved_modules = {name: module for name, module in list(sys.modules.items()) if _is_cold_start_module(name)}
    for name in saved_modules:
        sys.modules.pop(name, None)

    try:
        rc = run_cli(["clawcodex", "--version"])
        assert rc == 0
        assert "src.tui.app" not in sys.modules
        assert "src.repl.core" not in sys.modules
    finally:
        for name in list(sys.modules):
            if _is_cold_start_module(name):
                sys.modules.pop(name, None)
        sys.modules.update(saved_modules)


def test_run_cli_with_args_config_skips_run_pre_action(monkeypatch):
    """--config calls show_config and skips run_pre_action."""
    from clawcodex_ext.cli.dispatch import run_cli

    pre_action_called = []

    def fake_run_pre_action(args):
        pre_action_called.append(args)

    monkeypatch.setattr("src.init.run_pre_action", fake_run_pre_action)
    monkeypatch.setattr("src.cli.show_config", lambda: 0)

    rc = run_cli(["clawcodex", "--config"])
    # --config short-circuits before run_pre_action
    assert pre_action_called == []
    assert rc == 0


def test_run_cli_default_invocation_calls_downstream_repl(monkeypatch):
    """Default invocation reaches REPLFrontend and creates the downstream REPL."""
    import src.entrypoints.tui as tui_module
    from clawcodex_ext.cli.dispatch import run_cli

    init_calls = []
    repl_calls = []

    monkeypatch.setattr("src.init.run_pre_action", lambda args: init_calls.append(args))
    monkeypatch.setattr("clawcodex_ext.cli.permissions.resolve_permission_state", lambda args: None)
    monkeypatch.setattr(tui_module, "should_use_tui", lambda explicit: False)

    def fake_repl_init(self, **kwargs):
        repl_calls.append(kwargs)

        # Don't call real run() — just return silently
        def noop_run():
            return 0

        self.run = noop_run

    monkeypatch.setattr("clawcodex_ext.repl.app.ClawCodexExtREPL.__init__", fake_repl_init)
    monkeypatch.setattr(
        "clawcodex_ext.frontend.repl_extensions.install_repl_extensions",
        lambda repl, ctx: None,
    )

    rc = run_cli(["clawcodex"])
    assert rc == 0
    assert len(init_calls) == 1, f"expected 1 init call, got {init_calls}"
    assert len(repl_calls) == 1, f"expected 1 repl call, got {repl_calls}"
    assert repl_calls[0]["permission_mode"] == "default"


def test_run_cli_permission_flags_resolved(monkeypatch):
    """With --dangerously-skip-permissions, RuntimeContext is built with bypass_available=True."""
    # Ensure src.entrypoints.tui is imported before patching.
    import src.entrypoints.tui as tui_ref
    from clawcodex_ext.cli.dispatch import run_cli

    repl_calls = []
    built_options = []

    monkeypatch.setattr("src.init.run_pre_action", lambda args: None)
    monkeypatch.setattr(tui_ref, "should_use_tui", lambda explicit: False)
    # The bypass flag is intentionally rejected for root outside a sandbox.
    # Mark this cross-platform dispatch test as sandboxed; the dedicated
    # dangerous-permission tests cover rejection for unsandboxed root.
    monkeypatch.setenv("IS_SANDBOX", "1")

    # Patch RuntimeContext.build to capture options and still allow REPL to run
    original_build = None

    def capture_runtime_build(cls, options):
        built_options.append(options)
        return original_build(cls, options)

    def fake_repl_init(self, **kwargs):
        repl_calls.append(kwargs)

        def noop_run():
            return 0

        self.run = noop_run

    # Load modules first
    from clawcodex_ext.runtime.context import RuntimeContext

    original_build = RuntimeContext.build.__func__
    monkeypatch.setattr(
        "clawcodex_ext.runtime.context.RuntimeContext.build",
        classmethod(lambda cls, opts: capture_runtime_build(cls, opts)),
    )

    monkeypatch.setattr("clawcodex_ext.repl.app.ClawCodexExtREPL.__init__", fake_repl_init)
    monkeypatch.setattr(
        "clawcodex_ext.frontend.repl_extensions.install_repl_extensions",
        lambda repl, ctx: None,
    )

    rc = run_cli(["clawcodex", "--dangerously-skip-permissions"])
    assert rc == 0
    assert len(repl_calls) == 1, f"expected 1 repl call, got {repl_calls}"
    # With the real resolve_permission_state (not patched), --dangerously-skip-permissions
    # sets args._resolved_is_bypass_available = True, which propagates to
    # RuntimeOptions.is_bypass_permissions_mode_available = True
    assert len(built_options) == 1
    assert built_options[0].is_bypass_permissions_mode_available is True


def test_run_cli_print_goal_summary_skips_runtime_provider(
    monkeypatch,
    tmp_path,
    capsys,
):
    from clawcodex_ext.cli.dispatch import run_cli

    monkeypatch.setenv("CLAWCODEX_HOME", str(tmp_path / "clawcodex-home"))
    monkeypatch.setattr("src.init.run_pre_action", lambda args: None)

    def _runtime_must_not_be_built(cls, options):
        raise RuntimeError("runtime provider should not be built for /goal summary")

    monkeypatch.setattr(
        "clawcodex_ext.runtime.context.RuntimeContext.build",
        classmethod(_runtime_must_not_be_built),
    )

    rc = run_cli(["clawcodex", "-p", "/goal"])

    captured = capsys.readouterr()
    assert rc == 0
    assert "/goal <condition> to set one" in captured.out
    assert "No goal set" in captured.out
    assert "runtime provider should not be built" not in captured.err


def test_run_cli_json_keeps_provider_diagnostic_off_stdout(
    monkeypatch,
    tmp_path,
    capsys,
):
    """Structured output remains directly parseable by JSON consumers."""
    from clawcodex_ext.cli.dispatch import run_cli

    monkeypatch.setenv("CLAWCODEX_HOME", str(tmp_path / "clawcodex-home"))
    monkeypatch.setattr("src.init.run_pre_action", lambda args: None)

    rc = run_cli(
        [
            "clawcodex",
            "-p",
            "/goal",
            "--output-format",
            "json",
            "--provider",
            "deepseek",
            "--model",
            "deepseek-v4-flash",
        ]
    )

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert rc == 0
    assert payload["type"] == "result"
    assert "Provider: deepseek, Model: deepseek-v4-flash" in captured.err


def test_run_cli_print_goal_clear_skips_runtime_provider(monkeypatch, tmp_path, capsys):
    from clawcodex_ext.cli.dispatch import run_cli

    monkeypatch.setenv("CLAWCODEX_HOME", str(tmp_path / "clawcodex-home"))
    monkeypatch.setattr("src.init.run_pre_action", lambda args: None)
    monkeypatch.setattr(
        "clawcodex_ext.runtime.context.RuntimeContext.build",
        classmethod(lambda cls, options: (_ for _ in ()).throw(AssertionError("provider runtime must not be built"))),
    )

    rc = run_cli(["clawcodex", "-p", "/goal clear"])

    captured = capsys.readouterr()
    assert rc == 0
    assert "No goal set" in captured.out
    assert "provider runtime must not be built" not in captured.err


def test_run_cli_marks_max_turns_as_explicit_for_goal_print(monkeypatch, tmp_path):
    from clawcodex_ext.cli.dispatch import run_cli

    captured = []
    monkeypatch.setenv("CLAWCODEX_HOME", str(tmp_path / "clawcodex-home"))
    monkeypatch.setattr("src.init.run_pre_action", lambda args: None)
    monkeypatch.setattr(
        "src.entrypoints.headless.run_headless",
        lambda options: captured.append(options) or 0,
    )

    rc = run_cli(["clawcodex", "-p", "/goal", "--max-turns", "3"])

    assert rc == 0
    assert len(captured) == 1
    assert captured[0].max_turns == 3
    assert captured[0].max_turns_explicit is True


def test_run_cli_print_goal_summary_forwards_resume_and_continue(monkeypatch, tmp_path):
    """The provider-free fast path keeps the resolved target session ID."""
    import src.entrypoints.headless as headless_module
    import src.services.session_storage as session_storage_module
    from clawcodex_ext.cli.dispatch import run_cli

    resumed_session_id = "resolved-goal-session"
    sessions_dir = tmp_path / "sessions"
    (sessions_dir / resumed_session_id).mkdir(parents=True)
    captured_options = []

    monkeypatch.setenv("CLAWCODEX_SESSIONS_DIR", str(sessions_dir))
    monkeypatch.setattr("src.init.run_pre_action", lambda args: None)

    def _list_sessions(cls, *args, tag_filter=None, **kwargs):
        if tag_filter is not None:
            return []
        return [SimpleNamespace(session_id=resumed_session_id)]

    monkeypatch.setattr(
        session_storage_module.SessionStorage,
        "list_sessions",
        classmethod(_list_sessions),
    )
    monkeypatch.setattr(
        "clawcodex_ext.runtime.context.RuntimeContext.build",
        classmethod(lambda cls, options: (_ for _ in ()).throw(AssertionError("provider runtime must not be built"))),
    )

    def _capture_headless(options):
        captured_options.append(options)
        return 0

    monkeypatch.setattr(headless_module, "run_headless", _capture_headless)

    resume_rc = run_cli(["clawcodex", "--resume", resumed_session_id, "-p", "/goal"])
    continue_rc = run_cli(["clawcodex", "--continue", "-p", "/goal"])

    assert resume_rc == 0
    assert continue_rc == 0
    assert [options.resume_session_id for options in captured_options] == [
        resumed_session_id,
        resumed_session_id,
    ]


def test_run_cli_agent_debug_resume_uses_isolated_sessions_dir(
    monkeypatch,
    tmp_path,
):
    """--resume must use the session directory selected by --agent-debug."""
    import src.entrypoints.headless as headless_module
    from clawcodex_ext.cli.dispatch import run_cli

    resumed_session_id = "agent-debug-goal-session"
    debug_dir = tmp_path / "debug-state"
    (debug_dir / "sessions" / resumed_session_id).mkdir(parents=True)
    captured_options = []

    for name in (
        "CLAWCODEX_AGENT_DEBUG",
        "CLAWCODEX_HOME",
        "CLAWCODEX_HISTORY_FILE",
        "CLAWCODEX_SESSIONS_DIR",
        "CLAW_TELEMETRY_STORAGE_DIR",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CLAWCODEX_AGENT_DEBUG_DIR", str(debug_dir))
    monkeypatch.setattr("src.init.run_pre_action", lambda args: None)
    monkeypatch.setattr(
        "clawcodex_ext.runtime.context.RuntimeContext.build",
        classmethod(lambda cls, options: (_ for _ in ()).throw(AssertionError("provider runtime must not be built"))),
    )

    def _capture_headless(options):
        captured_options.append(options)
        return 0

    monkeypatch.setattr(headless_module, "run_headless", _capture_headless)

    rc = run_cli(
        [
            "clawcodex",
            "--agent-debug",
            "--resume",
            resumed_session_id,
            "-p",
            "/goal",
        ]
    )

    assert rc == 0
    assert os.environ["CLAWCODEX_SESSIONS_DIR"] == str(debug_dir / "sessions")
    assert [options.resume_session_id for options in captured_options] == [resumed_session_id]


def test_run_cli_provider_model_fast_paths_skip_pre_action(monkeypatch, capsys):
    from clawcodex_ext.cli.dispatch import run_cli

    pre_action_calls = []
    monkeypatch.setattr("src.init.run_pre_action", lambda args: pre_action_calls.append(args))
    monkeypatch.setattr("src.config.set_default_provider", lambda provider: None)
    monkeypatch.setattr(
        "src.config.get_provider_config",
        lambda provider: {"api_key": "secret", "base_url": "https://custom.example"},
    )
    monkeypatch.setattr("src.config.set_api_key", lambda *args, **kwargs: None)

    provider_rc = run_cli(["clawcodex", "provider", "use", "glm"])
    provider_out = capsys.readouterr().out
    model_rc = run_cli(["clawcodex", "model", "use", "zai/glm-4", "--provider", "glm"])
    model_out = capsys.readouterr().out

    assert provider_rc == 0
    assert model_rc == 0
    assert pre_action_calls == []
    assert "Default provider set to: zai" in provider_out
    assert "Default model for zai set to: zai/glm-4" in model_out


def test_run_cli_model_flag_value_provider_does_not_route_as_subcommand(monkeypatch):
    import src.entrypoints.tui as tui_module
    from clawcodex_ext.cli.dispatch import run_cli

    init_calls = []
    repl_calls = []
    monkeypatch.setattr("src.init.run_pre_action", lambda args: init_calls.append(args))
    monkeypatch.setattr("clawcodex_ext.cli.permissions.resolve_permission_state", lambda args: None)
    monkeypatch.setattr(tui_module, "should_use_tui", lambda explicit: False)

    def fake_repl_init(self, **kwargs):
        repl_calls.append(kwargs)
        self.run = lambda: 0

    monkeypatch.setattr("clawcodex_ext.repl.app.ClawCodexExtREPL.__init__", fake_repl_init)
    monkeypatch.setattr(
        "clawcodex_ext.frontend.repl_extensions.install_repl_extensions",
        lambda repl, ctx: None,
    )

    rc = run_cli(["clawcodex", "--model", "claude-sonnet-4-6"])

    assert rc == 0
    assert len(init_calls) == 1
    assert len(repl_calls) == 1
    assert repl_calls[0]["provider"].model == "claude-sonnet-4-6"


def test_run_cli_schedule_get_and_run_fast_paths(tmp_path, monkeypatch, capsys):
    from clawcodex_ext.cli.dispatch import run_cli
    from clawcodex_ext.cron_system.tasks import add_cron_task

    task = add_cron_task(tmp_path, cron="*/5 * * * *", prompt="ping", durable=True, created_at=1_000)
    monkeypatch.chdir(tmp_path)

    get_rc = run_cli(["clawcodex", "schedule", "get", task.id])
    get_output = capsys.readouterr().out
    run_rc = run_cli(["clawcodex", "schedule", "run", task.id])
    run_output = capsys.readouterr().out

    assert get_rc == 0
    assert f"Trigger: {task.id}" in get_output
    assert "Prompt: ping" in get_output
    assert run_rc == 0
    assert f"Trigger {task.id} fired." in run_output
    assert "Run ID:" in run_output


def test_build_parser_produces_functional_parser():
    """build_parser() returns a parser that handles permission flags."""
    from clawcodex_ext.cli.parser import build_parser

    parser = build_parser()
    args = parser.parse_args(["--dangerously-skip-permissions", "--permission-mode", "plan"])
    assert args.dangerously_skip_permissions is True
    assert args.permission_mode == "plan"


def test_build_parser_accepts_agent_debug_flag():
    from clawcodex_ext.cli.parser import build_parser

    parser = build_parser()
    args = parser.parse_args(["--agent-debug"])

    assert args.agent_debug is True


def test_run_cli_agent_debug_sets_debug_environment(monkeypatch):
    import src.entrypoints.tui as tui_module
    from clawcodex_ext.cli.dispatch import run_cli

    for name in (
        "CLAWCODEX_AGENT_DEBUG",
        "CLAWCODEX_AGENT_DEBUG_DIR",
        "CLAWCODEX_HISTORY_FILE",
        "CLAWCODEX_SESSIONS_DIR",
        "CLAW_TELEMETRY_STORAGE_DIR",
    ):
        monkeypatch.delenv(name, raising=False)

    monkeypatch.setattr("src.init.run_pre_action", lambda args: None)
    monkeypatch.setattr("clawcodex_ext.cli.permissions.resolve_permission_state", lambda args: None)
    monkeypatch.setattr(tui_module, "should_use_tui", lambda explicit: False)

    class FakeFrontend:
        def run(self, ctx, args):
            return 0

    monkeypatch.setattr("clawcodex_ext.frontend.get_frontend", lambda name: FakeFrontend())

    rc = run_cli(["clawcodex", "--agent-debug"])

    assert rc == 0
    assert os.environ["CLAWCODEX_AGENT_DEBUG"] == "1"
    assert os.environ["CLAWCODEX_HISTORY_FILE"].endswith("/history")
    assert os.environ["CLAWCODEX_SESSIONS_DIR"].endswith("/sessions")
    assert os.environ["CLAW_TELEMETRY_STORAGE_DIR"].endswith("/telemetry")


def test_build_parser_handles_allow_dangerously_skip(monkeypatch):
    """build_parser() accepts --allow-dangerously-skip-permissions."""
    from clawcodex_ext.cli.parser import build_parser

    parser = build_parser()
    args = parser.parse_args(["--allow-dangerously-skip-permissions"])
    assert args.allow_dangerously_skip_permissions is True


def test_build_parser_accepts_gateway_repl_options():
    """REPL startup can opt into a gateway binding without env-only setup."""
    from clawcodex_ext.cli.parser import build_parser

    parser = build_parser()
    args = parser.parse_args(
        [
            "--gateway-origin",
            "wechat:direct:default:user1",
            "--gateway-sock",
            "/tmp/clawcodex-gateway.sock",
        ]
    )
    assert args.gateway_origin == "wechat:direct:default:user1"
    assert args.gateway_sock == "/tmp/clawcodex-gateway.sock"


def test_build_parser_accepts_gateway_all_private_switch():
    """REPL startup can opt into all WeChat private messages without origin details."""
    from clawcodex_ext.cli.parser import build_parser

    parser = build_parser()
    args = parser.parse_args(["--gateway"])
    assert args.gateway is True
    assert args.gateway_origin is None


def test_build_parser_backward_compat_gateway_aliases_still_parse():
    """--im-gateway* flags still parse (hidden aliases, same dest)."""
    from clawcodex_ext.cli.parser import build_parser

    parser = build_parser()
    args = parser.parse_args(
        [
            "--im-gateway",
            "--im-gateway-origin",
            "wechat:direct:default:user2",
            "--im-gateway-sock",
            "/tmp/old-gateway.sock",
        ]
    )
    assert args.gateway is True
    assert args.gateway_origin == "wechat:direct:default:user2"
    assert args.gateway_sock == "/tmp/old-gateway.sock"


def test_resolve_permission_state_sets_args_attributes(monkeypatch):
    """resolve_permission_state stashes _resolved_permission_mode on args."""
    from argparse import Namespace

    from clawcodex_ext.cli.permissions import resolve_permission_state

    # Mock the safety gate and permission mode resolution
    monkeypatch.setattr(
        "src.permissions.dangerous_safety.enforce_dangerous_skip_permissions_safety",
        lambda bypass_requested: None,
    )
    monkeypatch.setattr(
        "src.permissions.modes.initial_permission_mode_from_cli",
        lambda permission_mode_cli, dangerously_skip_permissions, settings_default_mode=None: ("bypassPermissions"),
    )
    monkeypatch.setattr("src.permissions.modes.has_allow_bypass_permissions_mode", lambda: False)

    args = Namespace(
        dangerously_skip_permissions=True,
        allow_dangerously_skip_permissions=False,
        permission_mode=None,
    )
    resolve_permission_state(args)
    assert args._resolved_permission_mode == "bypassPermissions"
    assert args._resolved_is_bypass_available is True


def test_split_csv_utility():
    """_split_csv helper handles comma-separated tools."""
    from clawcodex_ext.cli.runners import _split_csv

    assert _split_csv(None) == []
    assert _split_csv("") == []
    assert _split_csv("foo") == ["foo"]
    assert _split_csv("foo, bar, baz") == ["foo", "bar", "baz"]
    assert _split_csv("foo,, bar") == ["foo", "bar"]  # empty segments skipped

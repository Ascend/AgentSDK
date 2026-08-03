from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from clawcodex_ext.cli.model_cmd.commands import (
    format_model_current,
    format_model_list,
    run_model_command,
    use_model,
)
from clawcodex_ext.cli.model_cmd.errors import UnknownModelError
from clawcodex_ext.cli.provider_cmd.commands import format_provider_current, run_provider_command


def _wait_for_text(render, expected: str) -> str:
    deadline = time.monotonic() + 1
    while time.monotonic() < deadline:
        output = render()
        if expected in output:
            return output
        time.sleep(0.01)
    raise AssertionError(f"background model refresh did not publish {expected!r}")


def test_provider_current_formats_resolution(monkeypatch) -> None:
    """``current`` output is a clean ``provider: …\nmodel: …`` with no source labels."""
    monkeypatch.delenv("CLAWCODEX_PROVIDER", raising=False)
    monkeypatch.delenv("CLAWCODEX_MODEL", raising=False)
    monkeypatch.setattr("clawcodex_ext.cli.model_cmd.resolver.get_default_provider", lambda: "glm")
    monkeypatch.setattr(
        "clawcodex_ext.cli.model_cmd.resolver.get_provider_config",
        lambda provider: {"default_model": "zai/glm-4"},
    )

    assert format_provider_current().splitlines() == [
        "provider: zai",
        "model: zai/glm-4",
    ]
    assert format_model_current().splitlines() == [
        "provider: zai",
        "model: zai/glm-4",
    ]


def test_provider_command_use_persists_default(monkeypatch, capsys) -> None:
    calls: list[str] = []
    monkeypatch.setattr("src.config.set_default_provider", calls.append)

    rc = run_provider_command(["use", "glm"])

    assert rc == 0
    assert calls == ["zai"]
    out = capsys.readouterr().out
    assert "Default provider set to: zai" in out
    # Restart hint also appears so the user doesn't think it took effect immediately.
    assert "next REPL launch" in out


def test_model_use_sets_provider_and_model(monkeypatch) -> None:
    provider_calls: list[str] = []
    model_calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "clawcodex_ext.cli.model_cmd.store.ModelStore.set_default_provider",
        lambda self, provider, scope="user": provider_calls.append(provider),
    )
    monkeypatch.setattr(
        "clawcodex_ext.cli.model_cmd.store.ModelStore.set_default_model_persist_unknown",
        lambda self, provider, model, scope="user": model_calls.append((provider, model)),
    )

    lines = use_model("zai/glm-4", provider="glm")

    assert provider_calls == ["zai"]
    assert model_calls == [("zai", "zai/glm-4")]
    # Three lines: two persistence lines + the "takes effect on next REPL launch" hint.
    assert lines[0] == "Default provider set to: zai"
    assert lines[1] == "Default model for zai set to: zai/glm-4"
    assert "next REPL launch" in lines[2]


def test_model_use_infers_provider_without_running_discovery_hooks(monkeypatch) -> None:
    provider_calls: list[str] = []
    model_calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "clawcodex_ext.cli.model_cmd.registry.ModelRegistry.available_models",
        lambda self, provider: (_ for _ in ()).throw(AssertionError("model use must not run live discovery")),
    )
    monkeypatch.setattr(
        "clawcodex_ext.cli.model_cmd.store.ModelStore.set_default_provider",
        lambda self, provider, scope="user": provider_calls.append(provider),
    )
    monkeypatch.setattr(
        "clawcodex_ext.cli.model_cmd.store.ModelStore.set_default_model_persist_unknown",
        lambda self, provider, model, scope="user": model_calls.append((provider, model)),
    )

    use_model("zai/glm-4")

    assert provider_calls == ["zai"]
    assert model_calls == [("zai", "zai/glm-4")]


def test_model_command_invalid_provider_returns_exit_2(capsys) -> None:
    rc = run_model_command(["list", "--provider", "missing"])

    assert rc == 2
    assert "Unknown provider" in capsys.readouterr().err


def test_model_list_loads_codex_extension_and_live_discovery(monkeypatch, capsys, tmp_path) -> None:
    from clawcodex_ext.providers.model_catalog_cache import reset_model_catalog_cache

    monkeypatch.setenv("CLAWCODEX_HOME", str(tmp_path))
    reset_model_catalog_cache()
    monkeypatch.setattr(
        "clawcodex_ext.providers.openai_codex_provider.resolve_codex_runtime_credentials",
        lambda **kwargs: SimpleNamespace(api_key="live-access-token"),
    )
    monkeypatch.setattr(
        "clawcodex_ext.providers.openai_codex_provider.get_codex_model_ids",
        lambda token, **kwargs: ["gpt-live-cli-model"],
    )

    rc = run_model_command(["list", "--provider", "openai-codex"])

    assert rc == 0
    output = capsys.readouterr().out
    assert "gpt-live-cli-model" in output


def test_model_list_uses_selected_provider_runtime_catalog(monkeypatch, tmp_path) -> None:
    from clawcodex_ext.providers.model_catalog_cache import reset_model_catalog_cache

    monkeypatch.setenv("CLAWCODEX_HOME", str(tmp_path))
    reset_model_catalog_cache()
    provider = SimpleNamespace(
        discover_available_models=lambda: ["deepseek-live-account-model"],
        get_available_models=lambda: ["deepseek-stale-model"],
    )
    monkeypatch.setattr(
        "src.providers.runtime.build_provider_from_config",
        lambda provider_name, model=None: provider,
    )

    format_model_list("deepseek")
    output = _wait_for_text(lambda: format_model_list("deepseek"), "deepseek-live-account-model")
    assert "deepseek-live-account-model" in output


def test_model_list_returns_before_background_catalog_refresh(monkeypatch, tmp_path) -> None:
    from clawcodex_ext.providers.model_catalog_cache import reset_model_catalog_cache

    monkeypatch.setenv("CLAWCODEX_HOME", str(tmp_path))
    reset_model_catalog_cache()
    started = threading.Event()
    release = threading.Event()

    def discover():
        started.set()
        release.wait(timeout=2)
        return ["deepseek-live-account-model"]

    provider = SimpleNamespace(
        get_available_models=lambda: ["deepseek-fallback-model"],
        discover_available_models=discover,
    )
    monkeypatch.setattr(
        "src.providers.runtime.build_provider_from_config",
        lambda provider_name, model=None: provider,
    )

    before = time.perf_counter()
    output = format_model_list("deepseek")
    elapsed = time.perf_counter() - before

    assert elapsed < 0.25
    assert "deepseek-fallback-model" in output
    assert "refresh" in output.lower()
    assert started.wait(timeout=0.5)
    release.set()


def test_model_list_without_provider_discovers_current_provider(monkeypatch, tmp_path) -> None:
    from clawcodex_ext.providers.model_catalog_cache import reset_model_catalog_cache

    monkeypatch.setenv("CLAWCODEX_HOME", str(tmp_path))
    reset_model_catalog_cache()
    provider = SimpleNamespace(
        get_available_models=lambda: ["current-provider-fallback-model"],
        discover_available_models=lambda: ["current-provider-live-model"],
    )
    monkeypatch.setattr(
        "clawcodex_ext.cli.model_cmd.commands.resolve",
        lambda: SimpleNamespace(provider="deepseek", model="deepseek-chat"),
    )
    monkeypatch.setattr(
        "src.providers.runtime.build_provider_from_config",
        lambda provider_name, model=None: provider,
    )

    format_model_list()
    output = _wait_for_text(format_model_list, "current-provider-live-model")

    assert "  deepseek:" in output
    assert "current-provider-live-model" in output
    assert "  anthropic:" not in output


def test_model_list_canonicalizes_current_provider_and_marks_resolved_model(monkeypatch, tmp_path) -> None:
    from clawcodex_ext.providers.model_catalog_cache import reset_model_catalog_cache

    monkeypatch.setenv("CLAWCODEX_HOME", str(tmp_path))
    reset_model_catalog_cache()
    provider = SimpleNamespace(
        get_available_models=lambda: ["zai/glm-5", "zai/glm-4"],
        discover_available_models=lambda: ["zai/glm-5", "zai/glm-4"],
    )
    monkeypatch.setattr(
        "clawcodex_ext.cli.model_cmd.commands.resolve",
        lambda: SimpleNamespace(provider="glm", model="zai/glm-4"),
    )
    monkeypatch.setattr(
        "src.providers.runtime.build_provider_from_config",
        lambda provider_name, model=None: provider,
    )

    output = format_model_list()

    assert "  zai:" in output
    assert "    zai/glm-4 *" in output


def test_model_list_uses_static_fallback_without_second_discovery(monkeypatch, tmp_path) -> None:
    from clawcodex_ext.providers.model_catalog_cache import reset_model_catalog_cache

    monkeypatch.setenv("CLAWCODEX_HOME", str(tmp_path))
    reset_model_catalog_cache()
    calls = 0
    attempted = threading.Event()

    def fail_discovery():
        nonlocal calls
        calls += 1
        attempted.set()
        raise RuntimeError("offline")

    provider = SimpleNamespace(
        get_available_models=lambda: ["deepseek-fallback-model"],
        discover_available_models=fail_discovery,
    )
    monkeypatch.setattr(
        "src.providers.runtime.build_provider_from_config",
        lambda provider_name, model=None: provider,
    )

    output = format_model_list("deepseek")

    assert attempted.wait(timeout=0.5)
    second = format_model_list("deepseek")
    assert calls == 1
    assert "fallback" in output.lower()
    assert "offline" in second


def test_model_list_labels_provider_discovery_fallback(monkeypatch, tmp_path) -> None:
    from clawcodex_ext.providers.model_catalog_cache import reset_model_catalog_cache

    monkeypatch.setenv("CLAWCODEX_HOME", str(tmp_path))
    reset_model_catalog_cache()
    attempted = threading.Event()

    def fail_discovery():
        attempted.set()
        raise RuntimeError("TLS EOF")

    provider = SimpleNamespace(
        get_available_models=lambda: ["deepseek-fallback-model"],
        discover_available_models=fail_discovery,
    )
    monkeypatch.setattr(
        "src.providers.runtime.build_provider_from_config",
        lambda provider_name, model=None: provider,
    )

    format_model_list("deepseek")
    assert attempted.wait(timeout=0.5)
    output = format_model_list("deepseek")

    assert "refresh failed" in output
    assert "fallback" in output.lower()


# ---------------------------------------------------------------------------
# CLI surface: dead --scope parameter removed
# ---------------------------------------------------------------------------


def test_model_use_rejects_scope_flag(capsys) -> None:
    """``--scope`` is no longer accepted by ``clawcodex model use``."""
    rc = run_model_command(["use", "zai/glm-4", "--scope", "project"])

    assert rc == 2
    err = capsys.readouterr().err
    assert "--scope" in err
    assert "Unknown argument" in err


def test_provider_use_rejects_scope_flag(capsys) -> None:
    """``--scope`` is no longer accepted by ``clawcodex provider use``."""
    rc = run_provider_command(["use", "glm", "--scope", "project"])

    assert rc == 2
    err = capsys.readouterr().err
    assert "--scope" in err
    assert "Unknown argument" in err


def test_provider_unset_rejects_scope_flag(capsys) -> None:
    """``clawcodex provider unset --scope …`` is also rejected."""
    rc = run_provider_command(["unset", "--scope", "project"])

    assert rc == 2
    err = capsys.readouterr().err
    assert "--scope" in err
    assert "Unknown argument" in err


# ---------------------------------------------------------------------------
# CLI surface: no-args = current (zero-arg query idiom)
# ---------------------------------------------------------------------------


def test_model_no_args_equals_current(monkeypatch, capsys) -> None:
    """``clawcodex model`` (no args) prints the same as ``clawcodex model current``."""
    monkeypatch.delenv("CLAWCODEX_PROVIDER", raising=False)
    monkeypatch.delenv("CLAWCODEX_MODEL", raising=False)
    monkeypatch.setattr("clawcodex_ext.cli.model_cmd.resolver.get_default_provider", lambda: "glm")
    monkeypatch.setattr(
        "clawcodex_ext.cli.model_cmd.resolver.get_provider_config",
        lambda provider: {"default_model": "zai/glm-4"},
    )

    rc = run_model_command([])

    assert rc == 0
    assert capsys.readouterr().out == format_model_current() + "\n"


def test_provider_no_args_equals_current(monkeypatch, capsys) -> None:
    """``clawcodex provider`` (no args) prints the same as ``clawcodex provider current``."""
    monkeypatch.delenv("CLAWCODEX_PROVIDER", raising=False)
    monkeypatch.delenv("CLAWCODEX_MODEL", raising=False)
    monkeypatch.setattr("clawcodex_ext.cli.model_cmd.resolver.get_default_provider", lambda: "glm")
    monkeypatch.setattr(
        "clawcodex_ext.cli.model_cmd.resolver.get_provider_config",
        lambda provider: {"default_model": "zai/glm-4"},
    )

    rc = run_provider_command([])

    assert rc == 0
    assert capsys.readouterr().out == format_provider_current() + "\n"


# ---------------------------------------------------------------------------
# Error message guidance: ``format_model_show`` points the user at --provider
# ---------------------------------------------------------------------------


def test_format_model_show_unknown_model_suggests_provider(capsys) -> None:
    """Unknown models point the user at ``--provider`` and ``model list``."""
    rc = run_model_command(["show", "nonexistent-model-xyz"])

    assert rc == 2
    err = capsys.readouterr().err
    assert "--provider" in err
    assert "model list" in err


def test_format_model_show_raises_with_provider_hint() -> None:
    """The underlying ``format_model_show`` raises with the new guidance message."""
    from clawcodex_ext.cli.model_cmd.commands import format_model_show

    with pytest.raises(UnknownModelError) as exc_info:
        format_model_show("nonexistent-model-xyz")

    msg = str(exc_info.value)
    assert "--provider" in msg
    assert "model list" in msg


# ---------------------------------------------------------------------------
# CLI surface: use/parse hardening (treats --bogus as unknown flag, not model name)
# ---------------------------------------------------------------------------


def test_model_use_rejects_bogus_flag(capsys) -> None:
    """``--bogus`` is rejected as an unknown flag instead of being treated as the model name."""
    rc = run_model_command(["use", "--bogus"])

    assert rc == 2
    err = capsys.readouterr().err
    assert "--bogus" in err
    assert "Unknown argument" in err


def test_model_use_requires_name(capsys) -> None:
    """``use`` with no NAME prints a focused "NAME is required" hint, not a menu dump."""
    rc = run_model_command(["use"])

    assert rc == 2
    err = capsys.readouterr().err
    assert "NAME is required" in err
    assert "Example:" in err


def test_model_use_provider_requires_value(capsys) -> None:
    """``--provider`` alone (no value) is rejected with a clear message."""
    rc = run_model_command(["use", "zai/glm-4", "--provider"])

    assert rc == 2
    err = capsys.readouterr().err
    assert "--provider" in err
    assert "requires a value" in err


def test_model_show_rejects_bogus_flag(capsys) -> None:
    """``show --bogus`` is rejected as an unknown flag."""
    rc = run_model_command(["show", "--bogus"])

    assert rc == 2
    err = capsys.readouterr().err
    assert "--bogus" in err
    assert "Unknown argument" in err


def test_provider_use_rejects_bogus_flag(capsys) -> None:
    """``provider use --bogus`` is rejected as an unknown flag."""
    rc = run_provider_command(["use", "--bogus"])

    assert rc == 2
    err = capsys.readouterr().err
    assert "--bogus" in err
    assert "Unknown argument" in err


def test_provider_use_requires_name(capsys) -> None:
    """``provider use`` with no NAME prints a focused "NAME is required" hint."""
    rc = run_provider_command(["use"])

    assert rc == 2
    err = capsys.readouterr().err
    assert "NAME is required" in err
    assert "Example:" in err


# ---------------------------------------------------------------------------
# CLI surface: help subcommand (discoverability)
# ---------------------------------------------------------------------------


def test_model_help_subcommand_prints_usage(capsys) -> None:
    """``clawcodex model help`` exits 0 and prints the usage to stdout."""
    rc = run_model_command(["help"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "usage: clawcodex model" in out
    assert "Subcommands:" in out
    for sub in ("list", "show", "current", "use", "help"):
        assert sub in out


def test_model_dash_help_flag_prints_usage(capsys) -> None:
    """``--help`` and ``-h`` are equivalent to the ``help`` subcommand."""
    for flag in ("--help", "-h"):
        rc = run_model_command([flag])
        assert rc == 0
        out = capsys.readouterr().out
        assert "usage: clawcodex model" in out


def test_provider_help_subcommand_prints_usage(capsys) -> None:
    """``clawcodex provider help`` exits 0 and prints the usage to stdout."""
    rc = run_provider_command(["help"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "usage: clawcodex provider" in out
    assert "Subcommands:" in out
    for sub in ("list", "show", "current", "use", "unset", "help"):
        assert sub in out


def test_provider_dash_help_flag_prints_usage(capsys) -> None:
    """``provider --help`` and ``-h`` are equivalent to the ``help`` subcommand."""
    for flag in ("--help", "-h"):
        rc = run_provider_command([flag])
        assert rc == 0
        out = capsys.readouterr().out
        assert "usage: clawcodex provider" in out


# ---------------------------------------------------------------------------
# CLI surface: explore view (--list / ls for model, --all / ls for provider)
# ---------------------------------------------------------------------------


def test_model_list_flag_shows_explore_view(monkeypatch, capsys) -> None:
    """``clawcodex model --list`` and ``ls`` give current + full list (REPL-equivalent)."""
    monkeypatch.delenv("CLAWCODEX_PROVIDER", raising=False)
    monkeypatch.delenv("CLAWCODEX_MODEL", raising=False)
    monkeypatch.setattr("clawcodex_ext.cli.model_cmd.resolver.get_default_provider", lambda: "glm")
    monkeypatch.setattr(
        "clawcodex_ext.cli.model_cmd.resolver.get_provider_config",
        lambda provider: {"default_model": "zai/glm-4"},
    )

    for flag in ("--list", "ls"):
        rc = run_model_command([flag])
        assert rc == 0, f"{flag!r} should exit 0"
        out = capsys.readouterr().out
        assert "provider: zai" in out
        assert "model: zai/glm-4" in out
        assert "Models:" in out  # the REPL's `_format_configured_model_list` header


def test_provider_all_flag_shows_explore_view(monkeypatch, capsys) -> None:
    """``clawcodex provider --all`` and ``ls`` give current + list (REPL-equivalent)."""
    monkeypatch.delenv("CLAWCODEX_PROVIDER", raising=False)
    monkeypatch.delenv("CLAWCODEX_MODEL", raising=False)
    monkeypatch.setattr("clawcodex_ext.cli.model_cmd.resolver.get_default_provider", lambda: "glm")
    monkeypatch.setattr(
        "clawcodex_ext.cli.model_cmd.resolver.get_provider_config",
        lambda provider: {"default_model": "zai/glm-4"},
    )

    for flag in ("--all", "ls"):
        rc = run_provider_command([flag])
        assert rc == 0, f"{flag!r} should exit 0"
        out = capsys.readouterr().out
        assert "provider: zai" in out
        assert "Providers:" in out


# ---------------------------------------------------------------------------
# Output format unification: ``model show`` matches ``model current``
# ---------------------------------------------------------------------------


def test_model_show_uses_current_format(monkeypatch) -> None:
    """``model show NAME`` and ``model current`` produce the same canonical shape."""
    monkeypatch.delenv("CLAWCODEX_PROVIDER", raising=False)
    monkeypatch.delenv("CLAWCODEX_MODEL", raising=False)
    monkeypatch.setattr("clawcodex_ext.cli.model_cmd.resolver.get_default_provider", lambda: "anthropic")
    monkeypatch.setattr(
        "clawcodex_ext.cli.model_cmd.resolver.get_provider_config",
        lambda provider: {"default_model": "claude-sonnet-4-6"},
    )
    from clawcodex_ext.cli.model_cmd.commands import format_model_show

    # ``show sonnet`` resolves via registry; the canonical format is identical to current.
    out_show = format_model_show("claude-sonnet-4-6")
    out_current = format_model_current()
    assert out_show == out_current
    assert "[" not in out_show  # no source labels
    assert out_show.splitlines() == ["provider: anthropic", "model: claude-sonnet-4-6"]

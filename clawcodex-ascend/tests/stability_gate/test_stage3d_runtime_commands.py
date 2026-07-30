"""Stage 3d — /model 和 /provider 运行时命令可用性测试（< 5 秒）。

覆盖 F-43 的 /model 和 /provider 斜杠命令在两种场景下的正确性：

- **有 runtime_context**：命令应显示当前 provider/model + 可用列表
- **无 runtime_context**：命令应降级显示列表（不抛异常 / 不报 Unknown command）

验证方式：直接通过 ``execute_command_sync`` 调用命令并断言返回值，
不经过 REPL 完整调度层，保持测试轻量快速。
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _build_context(*, runtime_context: bool = False):
    """Build a minimal ``CommandContext`` for testing.

    Parameters
    ----------
    runtime_context : bool
        When *True*, attach a ``runtime_context`` namespace so the command
        shows the current-state header.  When *False* (default), omit it so
        the command must degrade gracefully.
    """
    from clawcodex_ext.command_system.engine import create_command_context

    provider = SimpleNamespace(
        provider_name="anthropic",
        model="test-model",
        get_available_models=lambda: ["test-model"],
    )
    kwargs: dict = {
        "workspace_root": Path("/tmp"),
        "provider": provider,
    }
    if runtime_context:
        kwargs["runtime_context"] = SimpleNamespace(
            provider_name="anthropic",
            options=SimpleNamespace(model="test-model"),
            provider=provider,
            tool_registry=None,
            tool_context=None,
            swap_provider=lambda p, m=None: None,
        )
    return create_command_context(**kwargs)


# ---------------------------------------------------------------------------
# 注册测试
# ---------------------------------------------------------------------------


class TestRuntimeCommandsRegistration:
    """register_runtime_commands 将 /model 和 /provider 注册为 LocalCommand。"""

    def test_register_runtime_commands_adds_model(self):
        """register_runtime_commands 为全局 registry 添加 model LocalCommand。"""
        from clawcodex_ext.command_system import get_command_registry
        from clawcodex_ext.command_system.types import CommandType

        get_command_registry().clear()
        from clawcodex_ext.cli.runtime_commands import register_runtime_commands

        register_runtime_commands(None)
        cmd = get_command_registry().get("model")
        assert cmd is not None, "model command should be registered"
        assert cmd.command_type == CommandType.LOCAL, f"expected LOCAL, got {cmd.command_type}"

    def test_register_runtime_commands_adds_provider(self):
        """register_runtime_commands 为全局 registry 添加 provider LocalCommand。"""
        from clawcodex_ext.command_system import get_command_registry
        from clawcodex_ext.command_system.types import CommandType

        get_command_registry().clear()
        from clawcodex_ext.cli.runtime_commands import register_runtime_commands

        register_runtime_commands(None)
        cmd = get_command_registry().get("provider")
        assert cmd is not None, "provider command should be registered"
        assert cmd.command_type == CommandType.LOCAL, f"expected LOCAL, got {cmd.command_type}"


# ---------------------------------------------------------------------------
# 命令执行测试 — 无 runtime_context（降级路径）
# ---------------------------------------------------------------------------


class TestRuntimeCommandsWithoutRuntimeContext:
    """/model 和 /provider 在缺少 runtime_context 时降级工作。"""

    def _ensure_registered(self):
        from clawcodex_ext import ensure_eager_extensions_installed
        from clawcodex_ext.cli.runtime_commands import register_runtime_commands
        from clawcodex_ext.command_system import get_command_registry

        # Downstream provider registrations (e.g. kimi-coding) are deferred
        # from package import time to a lazy init function.  Ensure they
        # are in place before exercising /model or /provider commands.
        ensure_eager_extensions_installed()

        get_command_registry().clear()
        register_runtime_commands(None)

    def test_model_no_args_without_context(self):
        """/model 无参调用，无 runtime_context，返回成功和模型列表。"""
        self._ensure_registered()
        from clawcodex_ext.command_system.builtins import execute_command_sync

        success, result_text, error = execute_command_sync("model", "", _build_context(runtime_context=False))
        assert success is True, f"expected success, got error={error!r}"
        assert result_text is not None, "expected result text"
        assert "Models:" in result_text, "output should contain 'Models:'"
        assert "anthropic:" in result_text, "output should list providers"

    def test_provider_no_args_without_context(self):
        """/provider 无参调用，无 runtime_context，返回成功和提供商列表。"""
        self._ensure_registered()
        from clawcodex_ext.command_system.builtins import execute_command_sync

        success, result_text, error = execute_command_sync("provider", "", _build_context(runtime_context=False))
        assert success is True, f"expected success, got error={error!r}"
        assert result_text is not None, "expected result text"
        assert "Providers:" in result_text, "output should contain 'Providers:'"
        assert "anthropic" in result_text, "output should list providers"

    def test_model_no_args_without_context_no_unknown_command(self):
        """/model 不报 Unknown command。"""
        self._ensure_registered()
        from clawcodex_ext.command_system.builtins import execute_command_sync

        success, _, error = execute_command_sync("model", "", _build_context(runtime_context=False))
        assert success is True, f"should not return Unknown command; got error={error!r}"

    def test_provider_no_args_without_context_no_unknown_command(self):
        """/provider 不报 Unknown command。"""
        self._ensure_registered()
        from clawcodex_ext.command_system.builtins import execute_command_sync

        success, _, error = execute_command_sync("provider", "", _build_context(runtime_context=False))
        assert success is True, f"should not return Unknown command; got error={error!r}"


# ---------------------------------------------------------------------------
# 命令执行测试 — 有 runtime_context（完整路径）
# ---------------------------------------------------------------------------


class TestRuntimeCommandsWithRuntimeContext:
    """/model 和 /provider 在有 runtime_context 时正确显示当前状态。"""

    def _ensure_registered(self):
        from clawcodex_ext.cli.runtime_commands import register_runtime_commands
        from clawcodex_ext.command_system import get_command_registry

        get_command_registry().clear()
        register_runtime_commands(None)

    def test_model_no_args_with_context(self):
        """/model 无参调用，有 runtime_context，显示当前状态 + 模型列表。"""
        self._ensure_registered()
        from clawcodex_ext.command_system.builtins import execute_command_sync

        success, result_text, error = execute_command_sync("model", "", _build_context(runtime_context=True))
        assert success is True, f"expected success, got error={error!r}"
        assert result_text is not None, "expected result text"
        # 当前状态行 (OKLCH-colored Rich markup) (OKLCH-colored Rich markup)
        assert "[primary]provider[/primary]" in result_text, "output should show current provider"
        assert "test-model" in result_text, "output should show current model"
        # 列表
        assert "Models:" in result_text, "output should contain 'Models:'"

    def test_provider_no_args_with_context(self):
        """/provider 无参调用，有 runtime_context，显示当前状态 + 提供商列表。"""
        self._ensure_registered()
        from clawcodex_ext.command_system.builtins import execute_command_sync

        success, result_text, error = execute_command_sync("provider", "", _build_context(runtime_context=True))
        assert success is True, f"expected success, got error={error!r}"
        assert result_text is not None, "expected result text"
        # 当前状态行 (OKLCH-colored Rich markup) (OKLCH-colored Rich markup)
        assert "[primary]provider[/primary]" in result_text, "output should show current provider"
        assert "test-model" in result_text, "output should show current model"
        # 列表
        assert "Providers:" in result_text, "output should contain 'Providers:'"


# ---------------------------------------------------------------------------
# F-100 / 100.4 — /dream slash skill
# ---------------------------------------------------------------------------


class TestDreamCommandRegistration:
    """``register_dream_skill`` wires ``/dream`` as a LocalCommand in the
    global command registry (F-100 / 100.4).
    """

    def test_register_dream_skill_adds_dream(self):
        """register_dream_skill adds a LocalCommand named ``dream``."""
        from clawcodex_ext.command_system import get_command_registry
        from clawcodex_ext.command_system.types import CommandType
        from extensions.skills_ext.bundled.dream import register_dream_skill

        get_command_registry().clear()
        register_dream_skill()

        cmd = get_command_registry().get("dream")
        assert cmd is not None, "dream command should be registered"
        assert cmd.command_type == CommandType.LOCAL, f"expected LOCAL, got {cmd.command_type}"


class TestDreamCommandExecution:
    """``/dream`` subcommands run via execute_command_sync (F-100 / 100.4)."""

    @pytest.fixture(autouse=True)
    def _isolate_dream_service(self):
        """Reset the dream service's closure-scoped runner state.

        The ``_service._runner`` module-level singleton carries a
        :class:`RuntimeTaskRegistry` reference. Tests in
        ``tests/dreaming/`` populate that registry; the stage-3d
        ``/dream status`` test must observe an empty registry even
        when the full suite runs ``tests/dreaming/`` first.
        """
        from clawcodex_ext.dreaming import service as _service

        _service._runner = None
        yield
        _service._runner = None

    def _ensure_registered(self):
        from clawcodex_ext.command_system import get_command_registry
        from extensions.skills_ext.bundled.dream import register_dream_skill

        get_command_registry().clear()
        register_dream_skill()

    def test_dream_no_args_shows_help(self):
        """``/dream`` with no args returns the usage help text."""
        self._ensure_registered()
        from clawcodex_ext.command_system.builtins import execute_command_sync

        success, result_text, error = execute_command_sync("dream", "", _build_context(runtime_context=False))
        assert success is True, f"expected success, got error={error!r}"
        assert result_text is not None, "expected result text"
        assert "Usage:" in result_text
        assert "run" in result_text
        assert "status" in result_text

    def test_dream_help_subcommand(self):
        """``/dream help`` returns the same usage."""
        self._ensure_registered()
        from clawcodex_ext.command_system.builtins import execute_command_sync

        success, result_text, _ = execute_command_sync("dream", "help", _build_context(runtime_context=False))
        assert success is True
        assert "Usage:" in result_text

    def test_dream_status_no_init(self):
        """``/dream status`` works even when the dream service was not
        initialized (returns the empty-state message).
        """
        self._ensure_registered()
        from clawcodex_ext.command_system.builtins import execute_command_sync

        success, result_text, _ = execute_command_sync("dream", "status", _build_context(runtime_context=False))
        assert success is True
        assert "No dream tasks in flight" in result_text

    def test_dream_unknown_subcommand_does_not_crash(self):
        """``/dream frobnicate`` returns a clean warning, not a stack trace."""
        self._ensure_registered()
        from clawcodex_ext.command_system.builtins import execute_command_sync

        success, result_text, _ = execute_command_sync("dream", "frobnicate", _build_context(runtime_context=False))
        # Engine treats unknown-subcommand as a successful help render.
        assert success is True
        assert "Unknown subcommand" in result_text
        assert "frobnicate" in result_text

    def test_dream_command_no_unknown_command(self):
        """``/dream`` must not return ``Unknown command`` (F-100/100.4 验收 #5)."""
        self._ensure_registered()
        from clawcodex_ext.command_system.builtins import execute_command_sync

        success, result_text, error = execute_command_sync("dream", "", _build_context(runtime_context=False))
        assert success is True, f"should not return Unknown command; got error={error!r}, result_text={result_text!r}"


# ---------------------------------------------------------------------------
# 竞态条件回归测试 — register_builtin_commands(None) 不得覆盖 LocalCommand
# ---------------------------------------------------------------------------
# 根因：后台线程 _warm_slash_suggestions_cache → build_command_suggestions
# 调用 register_builtin_commands(None) 覆盖了全局注册表的 LocalCommand
# 为 InteractiveCommand，导致 execute_command_sync 报
# "Command not implemented for sync execution"。
# 修复方案 (clawcodex_ext/tui/commands.py) 改用私有 CommandRegistry。
# 以下测试守卫：即使 register_builtin_commands(None) 再次被调用，
# model/provider 仍为 LOCAL 且可执行。
# ---------------------------------------------------------------------------


class TestRuntimeCommandsRaceCondition:
    """build_command_suggestions 不得污染全局注册表（竞态条件回归）。"""

    def test_build_command_suggestions_does_not_overwrite_model(self):
        """build_command_suggestions 调用后全局 registry 的 model 仍为 LOCAL。"""

        from clawcodex_ext.cli.runtime_commands import register_runtime_commands
        from clawcodex_ext.command_system import get_command_registry
        from clawcodex_ext.command_system.builtins import register_builtin_commands
        from clawcodex_ext.command_system.types import CommandType
        from clawcodex_ext.tui.commands import build_command_suggestions

        reg = get_command_registry()
        reg.clear()

        # 模拟 REPL 正常启动顺序
        register_builtin_commands(None)
        register_runtime_commands(None)

        # 验证初始状态
        assert reg.get("model").command_type == CommandType.LOCAL

        # 模拟后台线程调用 build_command_suggestions
        build_command_suggestions(Path("/tmp"))

        # 关键断言：全局注册表未被污染
        cmd = reg.get("model")
        assert cmd is not None, "model command must survive build_command_suggestions"
        assert cmd.command_type == CommandType.LOCAL, (
            f"expected LOCAL after build_command_suggestions, got {cmd.command_type}"
        )

    def test_build_command_suggestions_does_not_overwrite_provider(self):
        """build_command_suggestions 调用后全局 registry 的 provider 仍为 LOCAL。"""

        from clawcodex_ext.cli.runtime_commands import register_runtime_commands
        from clawcodex_ext.command_system import get_command_registry
        from clawcodex_ext.command_system.builtins import register_builtin_commands
        from clawcodex_ext.command_system.types import CommandType
        from clawcodex_ext.tui.commands import build_command_suggestions

        reg = get_command_registry()
        reg.clear()

        register_builtin_commands(None)
        register_runtime_commands(None)

        assert reg.get("provider").command_type == CommandType.LOCAL

        build_command_suggestions(Path("/tmp"))

        cmd = reg.get("provider")
        assert cmd is not None, "provider command must survive build_command_suggestions"
        assert cmd.command_type == CommandType.LOCAL, (
            f"expected LOCAL after build_command_suggestions, got {cmd.command_type}"
        )

    def test_model_executable_after_build_command_suggestions(self):
        """build_command_suggestions 后 model 仍可通过 execute_command_sync 执行。"""

        from clawcodex_ext import ensure_eager_extensions_installed
        from clawcodex_ext.cli.runtime_commands import register_runtime_commands
        from clawcodex_ext.command_system import get_command_registry
        from clawcodex_ext.command_system.builtins import (
            execute_command_sync,
            register_builtin_commands,
        )
        from clawcodex_ext.tui.commands import build_command_suggestions

        ensure_eager_extensions_installed()

        reg = get_command_registry()
        reg.clear()
        register_builtin_commands(None)
        register_runtime_commands(None)

        build_command_suggestions(Path("/tmp"))

        success, result_text, error = execute_command_sync("model", "", _build_context(runtime_context=False))
        assert success is True, f"should be executable after build_command_suggestions; got error={error!r}"
        assert "Models:" in (result_text or "")

    def test_provider_executable_after_build_command_suggestions(self):
        """build_command_suggestions 后 provider 仍可通过 execute_command_sync 执行。"""

        from clawcodex_ext.cli.runtime_commands import register_runtime_commands
        from clawcodex_ext.command_system import get_command_registry
        from clawcodex_ext.command_system.builtins import (
            execute_command_sync,
            register_builtin_commands,
        )
        from clawcodex_ext.tui.commands import build_command_suggestions

        reg = get_command_registry()
        reg.clear()
        register_builtin_commands(None)
        register_runtime_commands(None)

        build_command_suggestions(Path("/tmp"))

        success, result_text, error = execute_command_sync("provider", "", _build_context(runtime_context=False))
        assert success is True, f"should be executable after build_command_suggestions; got error={error!r}"
        assert "Providers:" in (result_text or "")


# ---------------------------------------------------------------------------
# 未知模型回退 provider 测试 — 使用当前运行时 provider 而非硬编码 anthropic
# ---------------------------------------------------------------------------
# 旧行为：infer_provider_for_model 失败时硬编码 provider = "anthropic"
# 新行为：使用运行时上下文的 provider_name，保留用户当前的 provider


class TestRuntimeCommandCompletion:
    """/model 和 /provider 在 build_command_suggestions / 补全弹窗中的可见性。"""

    def test_build_command_suggestions_includes_model(self):
        """build_command_suggestions 返回值应包含 model 条目。"""
        from clawcodex_ext.tui.commands import build_command_suggestions

        suggestions = build_command_suggestions(Path("/tmp"))
        names = [s.name for s in suggestions]
        assert "model" in names, f"build_command_suggestions must include 'model'; got {names}"

    def test_build_command_suggestions_includes_provider(self):
        """build_command_suggestions 返回值应包含 provider 条目。"""
        from clawcodex_ext.tui.commands import build_command_suggestions

        suggestions = build_command_suggestions(Path("/tmp"))
        names = [s.name for s in suggestions]
        assert "provider" in names, f"build_command_suggestions must include 'provider'; got {names}"

    def test_build_command_suggestions_model_entry_is_slash_completable(self):
        """model 条目应有非空的 slash 属性（能被 _SlashOnlyCompleter 补全）。"""
        from clawcodex_ext.tui.commands import build_command_suggestions

        suggestions = build_command_suggestions(Path("/tmp"))
        model_entry = next((s for s in suggestions if s.name == "model"), None)
        assert model_entry is not None, "model entry must exist"
        assert model_entry.slash == "/model", f"expected slash='/model', got {model_entry.slash!r}"

    def test_build_command_suggestions_provider_entry_is_slash_completable(self):
        """provider 条目应有非空的 slash 属性（能被 _SlashOnlyCompleter 补全）。"""
        from clawcodex_ext.tui.commands import build_command_suggestions

        suggestions = build_command_suggestions(Path("/tmp"))
        provider_entry = next((s for s in suggestions if s.name == "provider"), None)
        assert provider_entry is not None, "provider entry must exist"
        assert provider_entry.slash == "/provider", f"expected slash='/provider', got {provider_entry.slash!r}"

    def test_provider_appears_in_slash_only_completer_flat_words(self):
        """/provider 应出现在 _get_slash_command_words 扁平列表中（REPL 补全备用源）。"""
        # 模拟 REPL 的 _get_slash_command_words 逻辑
        from clawcodex_ext.cli.runtime_commands import register_runtime_commands
        from clawcodex_ext.command_system import get_command_registry
        from clawcodex_ext.command_system.builtins import register_builtin_commands
        from clawcodex_ext.tui.commands import build_command_words

        reg = get_command_registry()
        reg.clear()
        register_builtin_commands(None)
        register_runtime_commands(None)

        words = build_command_words(Path("/tmp"))
        assert "/provider" in words, f"flat words must include '/provider'; got {words}"
        assert "/model" in words, f"flat words must include '/model'; got {words}"


class TestModelProviderFallback:
    """未知模型回退到当前运行时 provider，而非硬编码 anthropic。"""

    def _ensure_registered(self):
        from clawcodex_ext.cli.runtime_commands import register_runtime_commands
        from clawcodex_ext.command_system import get_command_registry

        get_command_registry().clear()
        register_runtime_commands(None)

    def test_unknown_model_falls_back_to_runtime_provider(self):
        """未知模型使用运行时上下文的 provider，不是 anthropic。"""
        self._ensure_registered()

        # 构造一个 provider="openai" 的运行时上下文
        from types import SimpleNamespace

        from clawcodex_ext.command_system.engine import create_command_context

        provider = SimpleNamespace(
            model="gpt-4",
            get_available_models=lambda: ["gpt-4"],
        )
        context = create_command_context(
            workspace_root=Path("/tmp"),
            provider=provider,
            runtime_context=SimpleNamespace(
                provider_name="openai",
                options=SimpleNamespace(model="gpt-4"),
                provider=provider,
                tool_registry=None,
                tool_context=None,
                swap_provider=lambda p, m=None: None,
            ),
        )

        from clawcodex_ext.command_system.builtins import execute_command_sync

        success, result_text, error = execute_command_sync("model", "truly-unknown-model-xyz", context)
        assert success is True, f"expected success, got error={error!r}"
        # 应回退到 openai，而非 anthropic
        assert "provider: openai" in (result_text or ""), f"expected 'provider: openai' in output, got {result_text!r}"
        # 不应包含 "anthropic"
        assert "anthropic" not in (result_text or "").lower(), f"should not fall back to anthropic, got {result_text!r}"
        # 应包含 unknown model 警告
        assert "unknown model" in (result_text or "").lower(), f"expected unknown model warning, got {result_text!r}"

    def test_known_model_stays_on_current_provider(self):
        """已知模型（在当前 provider 列表中）不切换 provider。"""
        self._ensure_registered()

        from types import SimpleNamespace

        from clawcodex_ext.command_system.engine import create_command_context

        provider = SimpleNamespace(
            model="gpt-4",
            get_available_models=lambda: ["gpt-4", "another-model"],
        )
        context = create_command_context(
            workspace_root=Path("/tmp"),
            provider=provider,
            runtime_context=SimpleNamespace(
                provider_name="openai",
                options=SimpleNamespace(model="gpt-4"),
                provider=provider,
                tool_registry=None,
                tool_context=None,
                swap_provider=lambda p, m=None: None,
            ),
        )

        from clawcodex_ext.command_system.builtins import execute_command_sync

        success, result_text, error = execute_command_sync("model", "another-model", context)
        assert success is True, f"expected success, got error={error!r}"
        # provider 应保持 openai（another-model 在当前 provider 列表中）
        assert "provider: openai" in (result_text or ""), f"expected 'provider: openai', got {result_text!r}"


# ---------------------------------------------------------------------------
# 带参数的 /provider xxx 和 /model xxx 命令测试
# ---------------------------------------------------------------------------


class TestRuntimeCommandsWithArgs:
    """/provider <name> 和 /model <name> 带参调用不崩溃、不报 Unknown command。"""

    @pytest.fixture(autouse=True)
    def _no_config_persistence(self):
        """阻止测试中的 /provider 写入 ~/.clawcodex/config.json。

        /provider <name> 在生产中会持久化默认供应商（已知或未知）。
        测试不应触及实际配置文件，因此 mock 掉 set_default_provider。
        """
        from unittest.mock import patch

        with patch("src.config.set_default_provider"):
            yield

    def _ensure_registered(self):
        from clawcodex_ext.cli.runtime_commands import register_runtime_commands
        from clawcodex_ext.command_system import get_command_registry

        get_command_registry().clear()
        register_runtime_commands(None)

    # ── /model <name> ─────────────────────────────────────────────────

    def test_model_known_arg_without_context(self):
        """/model <已知模型> 无 runtime_context，返回成功且不报 Unknown command。"""
        self._ensure_registered()
        from clawcodex_ext.command_system.builtins import execute_command_sync

        success, _result_text, error = execute_command_sync(
            "model", "claude-sonnet-4-6", _build_context(runtime_context=False)
        )
        assert success is True, f"expected success, got error={error!r}"

    def test_model_unknown_arg_does_not_crash(self):
        """/model <未知模型> 不崩溃、不报 Unknown command。"""
        self._ensure_registered()
        from clawcodex_ext.command_system.builtins import execute_command_sync

        success, result_text, error = execute_command_sync(
            "model",
            "truly-nonexistent-model-xyz-12345",
            _build_context(runtime_context=False),
        )
        assert success is True, f"should not crash on unknown model; got error={error!r}"
        assert result_text is not None

    # ── /provider <name> ─────────────────────────────────────────────

    def test_provider_known_arg_without_context(self):
        """/provider <已知供应商> 无 runtime_context，不报 Unknown command。"""
        self._ensure_registered()
        from clawcodex_ext.command_system.builtins import execute_command_sync

        success, result_text, error = execute_command_sync("provider", "openai", _build_context(runtime_context=False))
        assert success is True, f"expected success, got error={error!r}"
        assert result_text is not None

    def test_provider_unknown_arg_does_not_crash(self):
        """/provider <未知供应商> 不崩溃，返回友好提示。"""
        self._ensure_registered()
        from clawcodex_ext.command_system.builtins import execute_command_sync

        success, result_text, error = execute_command_sync(
            "provider",
            "nonexistent-provider-xyz",
            _build_context(runtime_context=False),
        )
        assert success is True, f"should not crash on unknown provider; got error={error!r}"
        assert result_text is not None
        assert "provider" in result_text.lower(), f"expected provider-related output, got {result_text!r}"

    def test_provider_arg_with_runtime_context_does_not_crash(self):
        """/provider <name> 有 runtime_context，不因 model=None 崩溃（regression）。"""
        self._ensure_registered()

        from types import SimpleNamespace

        from clawcodex_ext.command_system.engine import create_command_context

        provider = SimpleNamespace(model="gpt-4", get_available_models=lambda: ["gpt-4"])
        context = create_command_context(
            workspace_root=Path("/tmp"),
            provider=provider,
            runtime_context=SimpleNamespace(
                provider_name="openai",
                options=SimpleNamespace(model="gpt-4"),
                provider=provider,
                tool_registry=None,
                tool_context=None,
                swap_provider=lambda p, m=None: None,
            ),
        )

        from clawcodex_ext.command_system.builtins import execute_command_sync

        success, result_text, error = execute_command_sync("provider", "openai", context)
        assert success is True, f"expected success, got error={error!r}"
        assert result_text is not None

    def test_provider_unknown_arg_with_runtime_context_does_not_crash(self):
        """/provider <未知供应商> 有 runtime_context 不崩溃
        （regression：_colorize_model_name 收到 model=None 时崩溃）。
        """
        self._ensure_registered()

        from types import SimpleNamespace

        from clawcodex_ext.command_system.engine import create_command_context

        provider = SimpleNamespace(model="gpt-4", get_available_models=lambda: ["gpt-4"])
        context = create_command_context(
            workspace_root=Path("/tmp"),
            provider=provider,
            runtime_context=SimpleNamespace(
                provider_name="openai",
                options=SimpleNamespace(model="gpt-4"),
                provider=provider,
                tool_registry=None,
                tool_context=None,
                swap_provider=lambda p, m=None: None,
            ),
        )

        from clawcodex_ext.command_system.builtins import execute_command_sync

        success, result_text, error = execute_command_sync("provider", "nonexistent-provider-xyz", context)
        assert success is True, f"should not crash; got error={error!r}"
        assert result_text is not None

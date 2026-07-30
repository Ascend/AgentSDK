"""Tests for the ``clawcodex mcp <verb>`` fast-path subcommand handler.

Covers ``list`` / ``add`` / ``remove`` verbs in ``src/entrypoints/mcp.py``,
including argument parsing, config-file I/O isolation, and exit-code
contract for error paths. Each test redirects ``CLAUDE_CONFIG_DIR`` to
a tmp dir and (where ``--scope project`` is exercised) ``chdir``'s into
a tmp directory so neither ``~/.claude/config.json`` nor any stray
``.mcp.json`` is touched.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from src.entrypoints.mcp import run_mcp_subcommand

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Point ``CLAUDE_CONFIG_DIR`` at ``tmp_path`` so user-scoped writes
    never touch the real ``~/.claude/config.json``.
    """
    config_dir = tmp_path / "claude_home"
    config_dir.mkdir()
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))
    return config_dir


@pytest.fixture
def project_cwd(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Switch cwd into ``tmp_path`` so project-scoped writes create the
    ``.mcp.json`` there instead of polluting the developer tree.
    """
    workdir = tmp_path / "project"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    return workdir


def _read_user_config(config_dir: Path) -> dict:
    path = config_dir / "config.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_project_mcp(workdir: Path) -> dict:
    path = workdir / ".mcp.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def _reset_mcp_disabled_state(monkeypatch: pytest.MonkeyPatch):
    """Clear the process-local ``_disabled_servers`` set before every test.

    ``set_mcp_server_enabled`` mutates a module-level set in
    ``clawcodex_ext.services.mcp.config``; without this fixture a
    ``disable`` in one test leaks into later tests.
    """
    try:
        from clawcodex_ext.services.mcp import config as _mcp_config

        monkeypatch.setattr(_mcp_config, "_disabled_servers", set())
    except Exception:  # pragma: no cover  # noqa: BLE001
        yield
        return
    yield


# ---------------------------------------------------------------------------
# help / unknown verb
# ---------------------------------------------------------------------------


class TestHelpAndDispatch:
    def test_help_exits_zero(self, capsys: pytest.CaptureFixture) -> None:
        assert run_mcp_subcommand(["--help"]) == 0
        out = capsys.readouterr().out
        assert "Usage: clawcodex mcp <verb>" in out
        assert "add" in out and "remove" in out and "list" in out

    def test_help_with_dash_h(self, capsys: pytest.CaptureFixture) -> None:
        assert run_mcp_subcommand(["-h"]) == 0
        assert "Usage" in capsys.readouterr().out

    def test_empty_args_shows_help(self, capsys: pytest.CaptureFixture) -> None:
        assert run_mcp_subcommand([]) == 0
        assert "Usage" in capsys.readouterr().out

    def test_unknown_verb_exits_two(
        self,
        capsys: pytest.CaptureFixture,
        isolated_config: Path,
    ) -> None:
        assert run_mcp_subcommand(["bogus"]) == 2
        err = capsys.readouterr().err
        assert "unknown verb 'bogus'" in err


# ---------------------------------------------------------------------------
# add — stdio
# ---------------------------------------------------------------------------


class TestAddStdio:
    def test_add_stdio_user_scope(
        self,
        capsys: pytest.CaptureFixture,
        isolated_config: Path,
    ) -> None:
        rc = run_mcp_subcommand(["add", "demo", "--scope", "user", "--", "echo", "hi"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "Added MCP server 'demo'" in out
        data = _read_user_config(isolated_config)
        assert "demo" in data["mcpServers"]
        assert data["mcpServers"]["demo"] == {"command": "echo", "args": ["hi"]}

    def test_add_stdio_default_scope_is_user(
        self,
        capsys: pytest.CaptureFixture,
        isolated_config: Path,
    ) -> None:
        """Omitting ``--scope`` must default to user (not project)."""
        rc = run_mcp_subcommand(["add", "demo", "--", "echo"])
        assert rc == 0
        assert "demo" in _read_user_config(isolated_config)["mcpServers"]

    def test_add_stdio_with_env(
        self,
        capsys: pytest.CaptureFixture,
        isolated_config: Path,
    ) -> None:
        rc = run_mcp_subcommand(
            [
                "add",
                "srv",
                "--env",
                "API=secret",
                "--env",
                "DEBUG=1",
                "--",
                "python",
                "-m",
                "srv",
            ]
        )
        assert rc == 0
        cfg = _read_user_config(isolated_config)["mcpServers"]["srv"]
        assert cfg["command"] == "python"
        assert cfg["args"] == ["-m", "srv"]
        assert cfg["env"] == {"API": "secret", "DEBUG": "1"}


# ---------------------------------------------------------------------------
# add — remote (URL)
# ---------------------------------------------------------------------------


class TestAddRemote:
    def test_add_http_default_type(
        self,
        capsys: pytest.CaptureFixture,
        isolated_config: Path,
    ) -> None:
        rc = run_mcp_subcommand(["add", "remote1", "--url", "https://example.com/mcp"])
        assert rc == 0
        cfg = _read_user_config(isolated_config)["mcpServers"]["remote1"]
        assert cfg == {"type": "http", "url": "https://example.com/mcp"}

    def test_add_sse(
        self,
        capsys: pytest.CaptureFixture,
        isolated_config: Path,
    ) -> None:
        rc = run_mcp_subcommand(
            [
                "add",
                "remote2",
                "--url",
                "https://example.com/sse",
                "--type",
                "sse",
            ]
        )
        assert rc == 0
        cfg = _read_user_config(isolated_config)["mcpServers"]["remote2"]
        assert cfg == {"type": "sse", "url": "https://example.com/sse"}

    def test_add_ws_with_header(
        self,
        capsys: pytest.CaptureFixture,
        isolated_config: Path,
    ) -> None:
        rc = run_mcp_subcommand(
            [
                "add",
                "remote3",
                "--url",
                "https://example.com",
                "--type",
                "ws",
                "--header",
                "Authorization:Bearer xyz",
            ]
        )
        assert rc == 0
        cfg = _read_user_config(isolated_config)["mcpServers"]["remote3"]
        assert cfg["type"] == "ws"
        assert cfg["url"] == "https://example.com"
        assert cfg["headers"] == {"Authorization": "Bearer xyz"}

    def test_add_url_and_command_is_rejected(
        self,
        capsys: pytest.CaptureFixture,
        isolated_config: Path,
    ) -> None:
        rc = run_mcp_subcommand(["add", "x", "--url", "https://x", "--", "echo"])
        assert rc == 1
        assert "not both" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# add — project scope
# ---------------------------------------------------------------------------


class TestAddProjectScope:
    def test_add_project_scope_writes_mcp_json(
        self,
        capsys: pytest.CaptureFixture,
        isolated_config: Path,
        project_cwd: Path,
    ) -> None:
        rc = run_mcp_subcommand(["add", "local", "--scope", "project", "--", "echo", "hi"])
        assert rc == 0
        data = _read_project_mcp(project_cwd)
        assert "local" in data["mcpServers"]
        # User-scope config must NOT be touched.
        assert _read_user_config(isolated_config) == {}


# ---------------------------------------------------------------------------
# add — error paths
# ---------------------------------------------------------------------------


class TestAddErrors:
    def test_missing_name(
        self,
        capsys: pytest.CaptureFixture,
        isolated_config: Path,
    ) -> None:
        assert run_mcp_subcommand(["add"]) == 1
        assert "missing server NAME" in capsys.readouterr().err

    def test_no_server_spec(
        self,
        capsys: pytest.CaptureFixture,
        isolated_config: Path,
    ) -> None:
        assert run_mcp_subcommand(["add", "orphan", "--scope", "user"]) == 1
        assert "no server specified" in capsys.readouterr().err

    def test_duplicate_name(
        self,
        capsys: pytest.CaptureFixture,
        isolated_config: Path,
    ) -> None:
        assert run_mcp_subcommand(["add", "demo", "--scope", "user", "--", "echo"]) == 0
        rc = run_mcp_subcommand(["add", "demo", "--scope", "user", "--", "echo"])
        assert rc == 1
        assert "already exists" in capsys.readouterr().err

    def test_invalid_scope(
        self,
        capsys: pytest.CaptureFixture,
        isolated_config: Path,
    ) -> None:
        rc = run_mcp_subcommand(["add", "x", "--scope", "enterprise", "--", "echo"])
        assert rc == 1
        err = capsys.readouterr().err
        assert "Cannot add MCP server to scope: enterprise" in err

    def test_invalid_type(
        self,
        capsys: pytest.CaptureFixture,
        isolated_config: Path,
    ) -> None:
        rc = run_mcp_subcommand(["add", "x", "--url", "https://x", "--type", "ftp"])
        assert rc == 1
        assert "invalid --type 'ftp'" in capsys.readouterr().err

    def test_bad_env_pair(
        self,
        capsys: pytest.CaptureFixture,
        isolated_config: Path,
    ) -> None:
        rc = run_mcp_subcommand(["add", "x", "--env", "NOVAL", "--", "echo"])
        assert rc == 1
        assert "expected KEY=VAL" in capsys.readouterr().err

    def test_extra_positional_before_dashdash(
        self,
        capsys: pytest.CaptureFixture,
        isolated_config: Path,
    ) -> None:
        rc = run_mcp_subcommand(["add", "a", "b", "--scope", "user", "--", "echo"])
        assert rc == 1
        assert "unexpected extra positional argument: 'b'" in capsys.readouterr().err

    def test_invalid_name_chars_rejected_by_underlying(
        self,
        capsys: pytest.CaptureFixture,
        isolated_config: Path,
    ) -> None:
        """``add_mcp_config`` enforces name charset; surface its error."""
        rc = run_mcp_subcommand(["add", "bad name!", "--scope", "user", "--", "echo"])
        assert rc == 1
        assert "Invalid name" in capsys.readouterr().err

    def test_bare_option_without_value(
        self,
        capsys: pytest.CaptureFixture,
        isolated_config: Path,
    ) -> None:
        """``--scope`` at the end of the arg list has no value → exit 1."""
        rc = run_mcp_subcommand(["add", "x", "--scope"])
        assert rc == 1
        assert "requires a value" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# remove
# ---------------------------------------------------------------------------


class TestRemove:
    def test_remove_existing(
        self,
        capsys: pytest.CaptureFixture,
        isolated_config: Path,
    ) -> None:
        assert run_mcp_subcommand(["add", "demo", "--scope", "user", "--", "echo"]) == 0
        rc = run_mcp_subcommand(["remove", "demo", "--scope", "user"])
        assert rc == 0
        assert "Removed MCP server 'demo'" in capsys.readouterr().out
        assert "demo" not in _read_user_config(isolated_config).get("mcpServers", {})

    def test_remove_missing(
        self,
        capsys: pytest.CaptureFixture,
        isolated_config: Path,
    ) -> None:
        rc = run_mcp_subcommand(["remove", "ghost", "--scope", "user"])
        assert rc == 1
        assert "No user-scoped MCP server found" in capsys.readouterr().err

    def test_remove_missing_name(
        self,
        capsys: pytest.CaptureFixture,
        isolated_config: Path,
    ) -> None:
        assert run_mcp_subcommand(["remove"]) == 1
        assert "missing server NAME" in capsys.readouterr().err

    def test_remove_invalid_scope(
        self,
        capsys: pytest.CaptureFixture,
        isolated_config: Path,
    ) -> None:
        rc = run_mcp_subcommand(["remove", "demo", "--scope", "bogus"])
        assert rc == 1
        assert "invalid --scope 'bogus'" in capsys.readouterr().err

    def test_remove_project_scope(
        self,
        capsys: pytest.CaptureFixture,
        isolated_config: Path,
        project_cwd: Path,
    ) -> None:
        # Add into project scope first.
        assert run_mcp_subcommand(["add", "local", "--scope", "project", "--", "echo"]) == 0
        assert run_mcp_subcommand(["remove", "local", "--scope", "project"]) == 0
        assert "local" not in _read_project_mcp(project_cwd).get("mcpServers", {})


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


class TestList:
    def test_list_empty(
        self,
        capsys: pytest.CaptureFixture,
        isolated_config: Path,
    ) -> None:
        assert run_mcp_subcommand(["list"]) == 0
        assert "(no MCP servers configured)" in capsys.readouterr().out

    def test_list_after_adds(
        self,
        capsys: pytest.CaptureFixture,
        isolated_config: Path,
    ) -> None:
        assert run_mcp_subcommand(["add", "zeta", "--scope", "user", "--", "echo"]) == 0
        capsys.readouterr()  # discard add stdout before list
        assert run_mcp_subcommand(["add", "alpha", "--scope", "user", "--", "echo"]) == 0
        capsys.readouterr()  # discard add stdout before list
        rc = run_mcp_subcommand(["list"])
        assert rc == 0
        out_lines = capsys.readouterr().out.strip().splitlines()
        # Sorted alphabetically (clawcodex_ext.services.mcp returns dict
        # from JSON, list() sorts here).
        assert out_lines == ["alpha", "zeta"]


# ---------------------------------------------------------------------------
# list — scope / format / --all
# ---------------------------------------------------------------------------


class TestListScope:
    def test_list_scope_user_filters(
        self,
        capsys: pytest.CaptureFixture,
        isolated_config: Path,
    ) -> None:
        assert run_mcp_subcommand(["add", "u", "--scope", "user", "--", "echo"]) == 0
        capsys.readouterr()
        rc = run_mcp_subcommand(["list", "--scope", "user"])
        assert rc == 0
        assert "u" in capsys.readouterr().out

    def test_list_scope_nonexistent_exits_one(
        self,
        capsys: pytest.CaptureFixture,
        isolated_config: Path,
    ) -> None:
        rc = run_mcp_subcommand(["list", "--scope", "nonexistent"])
        assert rc == 1
        assert "invalid --scope 'nonexistent'" in capsys.readouterr().err

    def test_list_scope_no_match_message(
        self,
        capsys: pytest.CaptureFixture,
        isolated_config: Path,
    ) -> None:
        rc = run_mcp_subcommand(["list", "--scope", "project"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "no MCP servers configured in scope 'project'" in out


class TestListFormat:
    def test_format_json(
        self,
        capsys: pytest.CaptureFixture,
        isolated_config: Path,
    ) -> None:
        assert run_mcp_subcommand(["add", "demo", "--scope", "user", "--", "echo"]) == 0
        capsys.readouterr()
        rc = run_mcp_subcommand(["list", "--format", "json"])
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert isinstance(payload, list)
        assert payload[0]["name"] == "demo"
        assert payload[0]["scope"] == "user"
        assert payload[0]["transport"] == "stdio"

    def test_format_table(
        self,
        capsys: pytest.CaptureFixture,
        isolated_config: Path,
    ) -> None:
        assert run_mcp_subcommand(["add", "remote", "--url", "https://example.com/mcp"]) == 0
        capsys.readouterr()
        rc = run_mcp_subcommand(["list", "--format", "table"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "remote" in out and "user" in out and "http" in out

    def test_format_invalid(
        self,
        capsys: pytest.CaptureFixture,
        isolated_config: Path,
    ) -> None:
        rc = run_mcp_subcommand(["list", "--format", "yaml"])
        assert rc == 1
        assert "invalid --format 'yaml'" in capsys.readouterr().err


class TestListAll:
    def test_all_marks_disabled(
        self,
        capsys: pytest.CaptureFixture,
        isolated_config: Path,
    ) -> None:
        assert run_mcp_subcommand(["add", "demo", "--scope", "user", "--", "echo"]) == 0
        capsys.readouterr()
        assert run_mcp_subcommand(["disable", "demo"]) == 0
        capsys.readouterr()
        rc = run_mcp_subcommand(["list", "--all"])
        assert rc == 0
        out = capsys.readouterr().out
        assert "demo  [disabled]" in out

    def test_all_without_disabled_no_marker(
        self,
        capsys: pytest.CaptureFixture,
        isolated_config: Path,
    ) -> None:
        assert run_mcp_subcommand(["add", "demo", "--scope", "user", "--", "echo"]) == 0
        capsys.readouterr()
        rc = run_mcp_subcommand(["list", "--all"])
        assert rc == 0
        out_lines = capsys.readouterr().out.strip().splitlines()
        assert out_lines == ["demo"]


# ---------------------------------------------------------------------------
# get
# ---------------------------------------------------------------------------


class TestGet:
    def test_get_stdio(
        self,
        capsys: pytest.CaptureFixture,
        isolated_config: Path,
    ) -> None:
        assert run_mcp_subcommand(["add", "demo", "--scope", "user", "--", "echo", "hi"]) == 0
        capsys.readouterr()
        rc = run_mcp_subcommand(["get", "demo"])
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["name"] == "demo"
        assert payload["scope"] == "user"
        assert payload["transport"] == "stdio"
        assert payload["config"]["command"] == "echo"
        assert payload["config"]["args"] == ["hi"]

    def test_get_remote(
        self,
        capsys: pytest.CaptureFixture,
        isolated_config: Path,
    ) -> None:
        assert run_mcp_subcommand(["add", "r", "--url", "https://example.com/mcp", "--type", "sse"]) == 0
        capsys.readouterr()
        rc = run_mcp_subcommand(["get", "r"])
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["transport"] == "sse"
        assert payload["config"]["url"] == "https://example.com/mcp"

    def test_get_missing(
        self,
        capsys: pytest.CaptureFixture,
        isolated_config: Path,
    ) -> None:
        rc = run_mcp_subcommand(["get", "ghost"])
        assert rc == 1
        assert "no MCP server found with name 'ghost'" in capsys.readouterr().err

    def test_get_no_name(
        self,
        capsys: pytest.CaptureFixture,
        isolated_config: Path,
    ) -> None:
        rc = run_mcp_subcommand(["get"])
        assert rc == 1
        assert "missing server NAME" in capsys.readouterr().err

    def test_get_extra_positional(
        self,
        capsys: pytest.CaptureFixture,
        isolated_config: Path,
    ) -> None:
        rc = run_mcp_subcommand(["get", "a", "b"])
        assert rc == 1
        assert "unexpected extra positional argument: 'b'" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# enable / disable
# ---------------------------------------------------------------------------


class TestEnableDisable:
    def test_disable_then_enable(
        self,
        capsys: pytest.CaptureFixture,
        isolated_config: Path,
    ) -> None:
        assert run_mcp_subcommand(["add", "demo", "--scope", "user", "--", "echo"]) == 0
        capsys.readouterr()
        rc = run_mcp_subcommand(["disable", "demo"])
        assert rc == 0
        assert "disabled (process-local" in capsys.readouterr().out

        rc = run_mcp_subcommand(["enable", "demo"])
        assert rc == 0
        assert "enabled" in capsys.readouterr().out

    def test_disable_unknown_exits_one(
        self,
        capsys: pytest.CaptureFixture,
        isolated_config: Path,
    ) -> None:
        rc = run_mcp_subcommand(["disable", "ghost"])
        assert rc == 1
        assert "no MCP server found with name 'ghost'" in capsys.readouterr().err

    def test_disable_no_name(
        self,
        capsys: pytest.CaptureFixture,
        isolated_config: Path,
    ) -> None:
        rc = run_mcp_subcommand(["disable"])
        assert rc == 1
        assert "missing server NAME" in capsys.readouterr().err

    def test_enable_no_name(
        self,
        capsys: pytest.CaptureFixture,
        isolated_config: Path,
    ) -> None:
        rc = run_mcp_subcommand(["enable"])
        assert rc == 1
        assert "missing server NAME" in capsys.readouterr().err

    def test_disable_extra_positional(
        self,
        capsys: pytest.CaptureFixture,
        isolated_config: Path,
    ) -> None:
        rc = run_mcp_subcommand(["disable", "a", "b"])
        assert rc == 1
        assert "unexpected extra positional argument: 'b'" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# doctor — only ``--quick`` (no live connect) so tests are hermetic
# ---------------------------------------------------------------------------


class TestDoctorQuick:
    def _seed(self, isolated_config: Path) -> None:
        assert run_mcp_subcommand(["add", "demo", "--scope", "user", "--", "echo"]) == 0

    def test_doctor_quick_text(
        self,
        capsys: pytest.CaptureFixture,
        isolated_config: Path,
    ) -> None:
        self._seed(isolated_config)
        capsys.readouterr()
        rc = run_mcp_subcommand(["doctor", "--quick"])
        # ``echo`` is a real binary so no warning is raised, but
        # ``status="unchecked"`` ≠ "healthy" → exit 1 by design.
        assert rc == 1
        out = capsys.readouterr().out
        assert "MCP Server Diagnostics" in out
        assert "demo" in out

    def test_doctor_quick_json(
        self,
        capsys: pytest.CaptureFixture,
        isolated_config: Path,
    ) -> None:
        self._seed(isolated_config)
        capsys.readouterr()
        rc = run_mcp_subcommand(["doctor", "--quick", "--json"])
        assert rc == 1
        payload = json.loads(capsys.readouterr().out)
        assert payload["total_count"] == 1
        assert payload["servers"][0]["name"] == "demo"
        assert payload["servers"][0]["status"] == "unchecked"

    def test_doctor_quick_static_warning_for_missing_command(
        self,
        capsys: pytest.CaptureFixture,
        isolated_config: Path,
    ) -> None:
        """A stdio command not on PATH must surface a warning in --quick."""
        assert run_mcp_subcommand(["add", "ghost-bin", "--scope", "user", "--", "definitely-not-a-real-cmd"]) == 0
        capsys.readouterr()
        rc = run_mcp_subcommand(["doctor", "--quick"])
        assert rc == 1
        out = capsys.readouterr().out
        assert "not found in PATH" in out

    def test_doctor_name_filters(
        self,
        capsys: pytest.CaptureFixture,
        isolated_config: Path,
    ) -> None:
        assert run_mcp_subcommand(["add", "a", "--scope", "user", "--", "echo"]) == 0
        assert run_mcp_subcommand(["add", "b", "--scope", "user", "--", "echo"]) == 0
        capsys.readouterr()
        rc = run_mcp_subcommand(["doctor", "--quick", "--name", "a"])
        assert rc == 1
        out = capsys.readouterr().out
        assert "Servers: 1 total" in out
        assert "a" in out and "b" not in out

    def test_doctor_name_unknown(
        self,
        capsys: pytest.CaptureFixture,
        isolated_config: Path,
    ) -> None:
        rc = run_mcp_subcommand(["doctor", "--name", "ghost"])
        assert rc == 1
        assert "no MCP server found with name 'ghost'" in capsys.readouterr().err

    def test_doctor_empty_config(
        self,
        capsys: pytest.CaptureFixture,
        isolated_config: Path,
    ) -> None:
        rc = run_mcp_subcommand(["doctor", "--quick"])
        # No servers → no unhealthy → exit 0.
        assert rc == 0
        assert "No MCP servers configured" in capsys.readouterr().out

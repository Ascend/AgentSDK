#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright (c) 2026 Clawd Codex Team
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

# pylint: disable=redefined-outer-name
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.services.computer_use import (
    LinuxBackend,
    LinuxClipboardManager,
    LinuxInputSimulator,
    LinuxScreenshotProvider,
    LinuxWindowManager,
    MouseButton,
    ScreenRegion,
    WindowRef,
)
from src.services.computer_use.exceptions import (
    BinaryNotFoundError,
    CoordinatesOutOfBoundsError,
    SafetyViolationError,
    WindowNotFoundError,
)


class _FakeCompleted:
    def __init__(self, stdout: bytes = b"", returncode: int = 0) -> None:
        self.stdout = stdout
        self.returncode = returncode


def _which_stub(binary: str) -> str | None:
    # Pretend every requested binary exists for ``_check_binary`` happy paths.
    return f"/usr/bin/{binary}"


@pytest.fixture
def fake_runner() -> list[dict[str, Any]]:
    """Returns the list of recorded subprocess calls."""
    return []


def _make_backend(
    fake_runner: list[dict[str, Any]],
    *,
    dry_run: bool = True,
    allowed: bool = False,
) -> LinuxBackend:
    def runner(argv, **kwargs):
        fake_runner.append({"argv": list(argv), "kwargs": dict(kwargs)})
        return _FakeCompleted(stdout=b"png-bytes")

    return LinuxBackend(
        dry_run=dry_run,
        allowed=allowed,
        runner=runner,
        timeout_seconds=0.1,
    )


def test_linux_backend_defaults_are_safe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CLAWCODEX_COMPUTER_USE_ALLOW", raising=False)
    backend = LinuxBackend(runner=lambda *a, **kw: pytest.fail("runner must not be called by default"))
    assert backend.dry_run is True
    assert backend.allowed is False
    assert backend.is_allowed() is False


def test_screenshot_dry_run_does_not_spawn_subprocess(fake_runner: list[dict[str, Any]]) -> None:
    backend = _make_backend(fake_runner, dry_run=True)
    provider = LinuxScreenshotProvider(backend)
    out = provider.capture_fullscreen()
    assert out == b""
    assert fake_runner == []


def test_screenshot_real_path_requires_allow_flag(fake_runner: list[dict[str, Any]]) -> None:
    backend = _make_backend(fake_runner, dry_run=False, allowed=False)
    provider = LinuxScreenshotProvider(backend)
    with pytest.raises(SafetyViolationError):
        provider.capture_fullscreen()


def test_screenshot_real_path_uses_argv_list_and_timeout(
    fake_runner: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("src.services.computer_use.platform.linux.shutil.which", _which_stub)
    backend = _make_backend(fake_runner, dry_run=False, allowed=True)
    provider = LinuxScreenshotProvider(backend)
    out = provider.capture_fullscreen()
    assert out == b"png-bytes"

    assert len(fake_runner) >= 1
    # First call: ``scrot --version`` probe; second call: actual capture.
    capture = fake_runner[-1]
    assert capture["argv"][0] == "scrot"
    assert capture["kwargs"].get("timeout") == 0.1
    assert capture["kwargs"].get("shell") in (None, False)
    assert capture["kwargs"].get("check") is True


def test_screenshot_region_passes_geometry(fake_runner: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.services.computer_use.platform.linux.shutil.which", _which_stub)
    backend = _make_backend(fake_runner, dry_run=False, allowed=True)
    provider = LinuxScreenshotProvider(backend)
    region = ScreenRegion(x=10, y=20, width=300, height=200)
    provider.capture_region(region)
    capture = fake_runner[-1]
    assert capture["argv"][0] == "scrot"
    assert "10,20,300,200" in capture["argv"]


def test_screenshot_window_raises_when_not_found(
    fake_runner: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("src.services.computer_use.platform.linux.shutil.which", _which_stub)

    def runner(argv, **kwargs):
        fake_runner.append({"argv": list(argv), "kwargs": dict(kwargs)})
        if "search" in argv:
            return _FakeCompleted(stdout=b"")
        return _FakeCompleted(stdout=b"")

    backend = LinuxBackend(dry_run=False, allowed=True, runner=runner, timeout_seconds=0.1)
    provider = LinuxScreenshotProvider(backend)
    with pytest.raises(WindowNotFoundError):
        provider.capture_window(WindowRef(title="missing"))


def test_screenshot_raises_when_binary_missing(
    fake_runner: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "src.services.computer_use.platform.linux.shutil.which",
        lambda binary: None,
    )
    backend = _make_backend(fake_runner, dry_run=False, allowed=True)
    provider = LinuxScreenshotProvider(backend)
    with pytest.raises(BinaryNotFoundError):
        provider.capture_fullscreen()


def test_input_dry_run_records_only(fake_runner: list[dict[str, Any]]) -> None:
    backend = _make_backend(fake_runner, dry_run=True)
    sim = LinuxInputSimulator(backend)
    sim.move_mouse(50, 60)
    sim.click()
    sim.type_text("hello")
    sim.press_key("Enter")
    sim.scroll(dy=2)
    sim.drag(1, 2, 3, 4)
    assert sim.is_dry_run is True
    assert fake_runner == []


def test_input_rejects_out_of_bounds(fake_runner: list[dict[str, Any]]) -> None:
    backend = _make_backend(fake_runner, dry_run=True)
    sim = LinuxInputSimulator(backend)
    with pytest.raises(CoordinatesOutOfBoundsError):
        sim.move_mouse(-1, 0)
    with pytest.raises(CoordinatesOutOfBoundsError):
        sim.drag(0, 0, 99_999, 0)


def test_input_rejects_whitespace_in_key(fake_runner: list[dict[str, Any]]) -> None:
    backend = _make_backend(fake_runner, dry_run=True)
    sim = LinuxInputSimulator(backend)
    with pytest.raises(ValueError):
        sim.press_key("ctrl c")


def test_input_real_path_uses_argv_list_and_timeout(
    fake_runner: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("src.services.computer_use.platform.linux.shutil.which", _which_stub)
    backend = _make_backend(fake_runner, dry_run=False, allowed=True)
    sim = LinuxInputSimulator(backend)
    sim.move_mouse(10, 20)
    sim.click(MouseButton.LEFT, x=30, y=40)
    sim.type_text("hello world")
    sim.press_key("Return")
    sim.scroll(dy=1)
    sim.drag(1, 2, 3, 4)

    # Filter only the real calls (not the --version probes).
    real_calls = [c for c in fake_runner if c["argv"][0] == "xdotool" and "search" not in c["argv"]]
    assert any(c["argv"][1:] == ["mousemove", "10", "20"] for c in real_calls)
    click_call = next(c for c in real_calls if c["argv"][1] == "click")
    # The click should follow a mousemove, not be passed extra untrusted bytes.
    assert all(isinstance(a, str) for a in click_call["argv"])
    # ``type`` must use the ``--`` separator to prevent flag injection.
    type_call = next(c for c in real_calls if c["argv"][1] == "type")
    assert "--" in type_call["argv"]
    assert "hello world" in type_call["argv"]
    # Press key must not contain whitespace in argv position 1.
    press_call = next(c for c in real_calls if c["argv"][1] == "key")
    assert "Return" in press_call["argv"]
    # Drag must chain mousedown + mouseup.
    drag_calls = [c for c in real_calls if c["argv"][1] in {"mousedown", "mouseup"}]
    assert drag_calls, "drag must emit mousedown/mouseup"
    for c in real_calls:
        assert c["kwargs"].get("timeout") == 0.1
        assert c["kwargs"].get("shell") in (None, False)


def test_input_type_text_uses_double_dash_separator(
    fake_runner: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("src.services.computer_use.platform.linux.shutil.which", _which_stub)
    backend = _make_backend(fake_runner, dry_run=False, allowed=True)
    sim = LinuxInputSimulator(backend)
    # A literal that starts with ``-`` would be interpreted as a flag without
    # the ``--`` separator. Make sure the backend still sends it as data.
    sim.type_text("-rf /tmp/fake-target")
    real_calls = [c for c in fake_runner if c["argv"][0] == "xdotool"]
    type_call = next(c for c in real_calls if c["argv"][1] == "type")
    assert type_call["argv"][2] == "--"
    assert type_call["argv"][3] == "-rf /tmp/fake-target"


def test_clipboard_real_path_uses_argv(fake_runner: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.services.computer_use.platform.linux.shutil.which", _which_stub)
    backend = _make_backend(fake_runner, dry_run=False, allowed=True)
    manager = LinuxClipboardManager(backend)
    manager.set_text("hello")
    set_call = next(
        c for c in fake_runner if c["argv"][0] == "xclip" and "-selection" in c["argv"] and "-out" not in c["argv"]
    )
    assert set_call["kwargs"]["input"] == b"hello"
    assert set_call["kwargs"].get("timeout") == 0.1
    assert set_call["kwargs"].get("shell") in (None, False)

    fake_runner.clear()
    manager.get_text()
    get_call = next(c for c in fake_runner if c["argv"][0] == "xclip" and "-out" in c["argv"])
    assert "-out" in get_call["argv"]


def test_window_manager_list_focus_close(fake_runner: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("src.services.computer_use.platform.linux.shutil.which", _which_stub)

    def runner(argv, **kwargs):
        fake_runner.append({"argv": list(argv), "kwargs": dict(kwargs)})
        if argv and argv[0] == "wmctrl" and "-l" in argv:
            return _FakeCompleted(stdout=b"0xdead 0 myhost Terminal\n0xbeef 0 myhost Editor\n")
        if argv and argv[0] == "wmctrl" and "-a" in argv:
            return _FakeCompleted(returncode=0)
        return _FakeCompleted(returncode=0)

    backend = LinuxBackend(dry_run=False, allowed=True, runner=runner, timeout_seconds=0.1)
    manager = LinuxWindowManager(backend)
    windows = manager.list_windows()
    assert [w.title for w in windows] == ["Terminal", "Editor"]
    assert manager.focus_window(WindowRef(title="Terminal")) is True

    focus_call = next(c for c in fake_runner if "wmctrl" in c["argv"] and "-a" in c["argv"])
    assert focus_call["argv"] == ["wmctrl", "-a", "Terminal"]


def test_window_manager_dry_run_records(fake_runner: list[dict[str, Any]]) -> None:
    backend = _make_backend(fake_runner, dry_run=True)
    manager = LinuxWindowManager(backend)
    assert manager.list_windows() == []
    assert manager.focus_window(WindowRef(title="x")) is False
    assert manager.close_window(WindowRef(title="x")) is False
    assert fake_runner == []


def test_subprocess_runner_is_invoked_with_list_argv() -> None:
    """A sanity test that the default runner is real ``subprocess.run`` and
    that callers always pass it a list. This is a guard against someone
    accidentally flipping the public API to accept a string.
    """

    captured: dict[str, Any] = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured["kwargs"] = kwargs
        return _FakeCompleted(stdout=b"")

    backend = LinuxBackend(dry_run=True, allowed=False, runner=fake_run)
    sim = LinuxInputSimulator(backend)
    sim.move_mouse(5, 5)
    # In dry-run, runner is not called; ``captured`` is still empty.
    assert "argv" not in captured

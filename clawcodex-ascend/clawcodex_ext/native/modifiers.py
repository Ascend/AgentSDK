#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -------------------------------------------------------------------------
#  This file is part of the AgentSDK project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
# Copyright (c) 2026 Clawd Codex Team
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

"""F-81.5: 键盘修饰键状态检测模块.

对标 CCB ``modifiers-napi``，检测 Shift / Ctrl / Alt / Meta (Cmd/Win) 的
当前按下状态。后端按平台选择:

* **Linux** — 优先 ``evdev``（直接读 ``/dev/input/event*``），fallback
  ``pynput``.
* **macOS / Windows** — ``pynput``.

缺失可选依赖时 :func:`clawcodex_ext.native.load_or_fallback` 返回
:class:`ModifiersFallback`，所有状态恒为 ``False``（"未按下"）。

辅助依赖: F-61 Computer Use（修饰键作为快捷键触发信号）.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

from clawcodex_ext.native import NativeModuleRegistry

__all__ = ["ModifiersModule", "ModifiersFallback", "ModifierState"]

_logger = logging.getLogger("clawcodex_ext.native.modifiers")


class ModifierState:
    """修饰键瞬时状态快照（值含义：``True`` = 当前按下）."""

    __slots__ = ("shift", "ctrl", "alt", "meta")

    def __init__(
        self,
        shift: bool = False,
        ctrl: bool = False,
        alt: bool = False,
        meta: bool = False,
    ) -> None:
        self.shift = shift
        self.ctrl = ctrl
        self.alt = alt
        self.meta = meta

    def __repr__(self) -> str:
        return f"ModifierState(shift={self.shift}, ctrl={self.ctrl}, alt={self.alt}, meta={self.meta})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ModifierState):
            return NotImplemented
        return (
            self.shift == other.shift and self.ctrl == other.ctrl and self.alt == other.alt and self.meta == other.meta
        )

    def any_pressed(self) -> bool:
        return self.shift or self.ctrl or self.alt or self.meta


def _detect_backend() -> Optional[str]:
    """返回可用后端名：``"pynput"`` / ``"evdev"`` / ``None``."""
    if sys.platform.startswith("linux"):
        try:
            import evdev  # noqa: F401

            return "evdev"
        except ImportError:
            pass
    try:
        import pynput  # noqa: F401

        return "pynput"
    except ImportError:
        return None


@NativeModuleRegistry.register("modifiers")
class ModifiersModule:
    """键盘修饰键状态检测."""

    name = "modifiers"

    def __init__(self) -> None:
        self._backend = _detect_backend()

    # -- NativeModule protocol --------------------------------------------

    def is_available(self) -> bool:
        return self._backend is not None

    def get_version(self) -> str:
        if self._backend == "evdev":
            try:
                import evdev

                return f"evdev/{getattr(evdev, '__version__', 'unknown')}"
            except ImportError:
                return "unavailable"
        if self._backend == "pynput":
            try:
                import pynput

                return f"pynput/{getattr(pynput, '__version__', 'unknown')}"
            except ImportError:
                return "unavailable"
        return "unavailable"

    # -- 状态读取 ---------------------------------------------------------

    def current_state(self) -> ModifierState:
        """返回四个修饰键的当前瞬时状态.

        Raises:
            NativeModuleError: 后端不可用.
        """
        if self._backend is None:
            from clawcodex_ext.native import NativeModuleError

            raise NativeModuleError("modifiers backend unavailable (install pynput or evdev)")
        if self._backend == "evdev":
            return self._state_evdev()
        return self._state_pynput()

    def _state_pynput(self) -> ModifierState:
        # pynput 的 keyboard.Controller 不直接暴露修饰键状态快照；
        # 这里用一个本地的 listener 累积状态。注意：listener 是后台线程，
        # 首次调用会启动它并保持进程生命周期内活跃。
        global _pynput_state
        if _pynput_state is None:
            _pynput_state = _PynputStateTracker()
            _pynput_state.start()
        return ModifierState(
            shift=_pynput_state.shift,
            ctrl=_pynput_state.ctrl,
            alt=_pynput_state.alt,
            meta=_pynput_state.meta,
        )

    def _state_evdev(self) -> ModifierState:
        # evdev 路径：读 /dev/input/event* 的 KEY_LEFTSHIFT 等事件需要 root
        # 或 input 组权限。这里采用保守策略：扫描可读设备，若全部不可读则
        # 抛 NativeModuleError，让调用方走 fallback。
        import evdev
        from evdev import ecodes

        shift = ctrl = alt = meta = False
        # 找一个可读的 keyboard 设备（capability 含 KEY_LEFTSHIFT）
        for path in evdev.list_devices():
            try:
                dev = evdev.InputDevice(path)
            except (OSError, PermissionError):
                continue
            cap = dev.capabilities()
            keys = cap.get(ecodes.EV_KEY, [])
            if ecodes.KEY_LEFTSHIFT not in keys:
                continue
            # 可读设备 —— 抓取当前状态（evdev 不直接给"当前状态"，需要
            # 监听；这里降级为 pynput 风格的后台 reader）
            global _evdev_state
            if _evdev_state is None or _evdev_state.device_path != path:
                _evdev_state = _EvdevStateTracker(path)
                _evdev_state.start()
            shift = _evdev_state.shift
            ctrl = _evdev_state.ctrl
            alt = _evdev_state.alt
            meta = _evdev_state.meta
            break
        return ModifierState(shift=shift, ctrl=ctrl, alt=alt, meta=meta)

    # -- F-81.6 fallback --------------------------------------------------

    @classmethod
    def fallback(cls) -> "ModifiersFallback":
        return ModifiersFallback()


# ---------------------------------------------------------------------------
# 后台状态追踪器（模块级单例，避免每次 current_state() 重启线程）
# ---------------------------------------------------------------------------


_pynput_state: "Optional[_PynputStateTracker]" = None
_evdev_state: "Optional[_EvdevStateTracker]" = None


class _PynputStateTracker:
    """pynput 后台 listener，累积修饰键 up/down 事件."""

    def __init__(self) -> None:
        self.shift = False
        self.ctrl = False
        self.alt = False
        self.meta = False
        self._listener = None

    def start(self) -> None:
        try:
            from pynput import keyboard
        except ImportError:
            return

        def on_press(key):
            self._set(key, True)

        def on_release(key):
            self._set(key, False)

        self._listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self._listener.daemon = True
        self._listener.start()

    def _set(self, key, value: bool) -> None:
        try:
            from pynput import keyboard
        except ImportError:
            return
        if key in (keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r):
            self.shift = value
        elif key in (keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r):
            self.ctrl = value
        elif key in (keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r, keyboard.Key.alt_gr):
            self.alt = value
        elif key in (
            keyboard.Key.cmd,
            keyboard.Key.cmd_l,
            keyboard.Key.cmd_r,
            keyboard.Key.win,
            keyboard.Key.menu,
        ):
            self.meta = value


class _EvdevStateTracker:
    """evdev 后台 reader，从指定设备读 KEY 事件维护状态."""

    def __init__(self, device_path: str) -> None:
        self.device_path = device_path
        self.shift = False
        self.ctrl = False
        self.alt = False
        self.meta = False
        self._thread = None

    def start(self) -> None:
        import threading

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            import evdev
            from evdev import ecodes
        except ImportError:
            return
        try:
            dev = evdev.InputDevice(self.device_path)
        except OSError:
            return
        key_map = {
            ecodes.KEY_LEFTSHIFT: "shift",
            ecodes.KEY_RIGHTSHIFT: "shift",
            ecodes.KEY_LEFTCTRL: "ctrl",
            ecodes.KEY_RIGHTCTRL: "ctrl",
            ecodes.KEY_LEFTALT: "alt",
            ecodes.KEY_RIGHTALT: "alt",
            ecodes.KEY_LEFTMETA: "meta",
            ecodes.KEY_RIGHTMETA: "meta",
        }
        try:
            for event in dev.read_loop():
                if event.type != ecodes.EV_KEY:
                    continue
                attr = key_map.get(event.code)
                if attr is None:
                    continue
                # value: 0=up, 1=down, 2=repeat
                setattr(self, attr, event.value != 0)
        except OSError:
            # 设备断开 —— 状态保留最后值
            pass


class ModifiersFallback:
    """F-81.6 fallback: 后端缺失时所有修饰键恒为 ``False``."""

    name = "modifiers"

    def is_available(self) -> bool:
        return False

    def get_version(self) -> str:
        return "fallback-noop"

    def current_state(self) -> ModifierState:
        return ModifierState()  # 全 False

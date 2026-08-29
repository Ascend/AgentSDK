#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
#
# Originally from Clawd Codex:
# https://github.com/agentforce314/clawcodex
# Copyright (c) 2026 Clawd Codex Team
# Licensed under the MIT License. See clawcodex-ascend/LICENSES/Clawd-Codex-MIT.txt.
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

"""Keyboard modifier-state detection."""

from __future__ import annotations

import logging
import sys
from typing import Optional

from clawcodex_ext.native import NativeModuleRegistry

__all__ = ["ModifiersModule", "ModifiersFallback", "ModifierState"]

_logger = logging.getLogger("clawcodex_ext.native.modifiers")


class ModifierState:
    """Snapshot of the current keyboard modifier state."""

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
    """Return the first available modifier backend."""
    if sys.platform.startswith("linux"):
        try:
            import evdev  # noqa: F401

            return "evdev"
        except ImportError:
            pass  # Optional integration is unavailable; keep the fallback.
    try:
        import pynput  # noqa: F401

        return "pynput"
    except ImportError:
        return None


@NativeModuleRegistry.register("modifiers")
class ModifiersModule:
    """Detect current keyboard modifier state."""

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

    # -- State reads -------------------------------------------------------

    def current_state(self) -> ModifierState:
        """Return the current keyboard modifier state."""
        if self._backend is None:
            from clawcodex_ext.native import NativeModuleError

            raise NativeModuleError("modifiers backend unavailable (install pynput or evdev)")
        if self._backend == "evdev":
            return self._state_evdev()
        return self._state_pynput()

    def _state_pynput(self) -> ModifierState:
        # pynput does not expose modifier snapshots, so a process-lifetime
        # background listener accumulates state after the first call.
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
        # Reading evdev events requires root or input-group access. Fail
        # conservatively when no keyboard device is readable.
        import evdev
        from evdev import ecodes

        shift = ctrl = alt = meta = False
        # Find a readable keyboard advertising KEY_LEFTSHIFT.
        for path in evdev.list_devices():
            try:
                dev = evdev.InputDevice(path)
            except (OSError, PermissionError):
                continue
            cap = dev.capabilities()
            keys = cap.get(ecodes.EV_KEY, [])
            if ecodes.KEY_LEFTSHIFT not in keys:
                continue
            # evdev has no snapshot API, so maintain state in a
            # pynput-style background reader.
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

    # -- fallback --------------------------------------------------

    @classmethod
    def fallback(cls) -> "ModifiersFallback":
        return ModifiersFallback()


# ---------------------------------------------------------------------------
# Module-level trackers avoid restarting threads for every state read.
# ---------------------------------------------------------------------------


_pynput_state: "Optional[_PynputStateTracker]" = None
_evdev_state: "Optional[_EvdevStateTracker]" = None


class _PynputStateTracker:
    """Track modifier state from pynput key events."""

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
    """Track modifier state from evdev key events."""

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
            # Retain the last state if the device disconnects.
            pass


class ModifiersFallback:
    """Report all modifiers as released when no backend is available."""

    name = "modifiers"

    def is_available(self) -> bool:
        return False

    def get_version(self) -> str:
        return "fallback-noop"

    def current_state(self) -> ModifierState:
        return ModifierState()  # Every modifier is released.

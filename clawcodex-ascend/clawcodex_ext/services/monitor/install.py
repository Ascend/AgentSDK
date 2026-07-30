"""F-88 Monitor extension installation hooks.

Called once per process from ``clawcodex_ext.ensure_eager_extensions_installed``
after all ``src/`` modules are loaded.  Command and tool registration happen
statically via ``clawcodex_ext.command_system.builtins`` and
``clawcodex_ext.tool_system.tools``; this module is responsible for runtime
wiring that must happen after upstream is initialised (e.g. TUI keybindings,
stall-watchdog exemption hooks).
"""

from __future__ import annotations

_installed: bool = False


def install_monitor_extensions() -> None:
    """Install F-88 runtime extensions once per process.

    Idempotent no-op when the feature gate is disabled; the command/tool
    ``is_enabled`` predicates already gate user-visible behaviour.
    """
    global _installed
    if _installed:
        return
    _installed = True

    # Future integration points:
    # 1. Stall-watchdog exemption: when a stall detector is introduced for
    #    background bash tasks, it should consult
    #    ``StallWatchdogExemptor.should_skip_stall_check(state)`` for
    #    ``kind='monitor'`` entries.
    # 2. TUI Shift+Down keybinding: when the monitor panel is loaded, its
    #    binding is injected here.


__all__ = ["install_monitor_extensions"]

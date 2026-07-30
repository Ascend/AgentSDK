"""Cross-platform ``watch -n <sec> <cmd>`` normalisation.

On Windows the ``watch`` utility is not generally available, so F-88 converts
it to a PowerShell ``while(1)`` loop.  On POSIX/macOS the command is returned
unchanged.
"""

from __future__ import annotations

import platform
import re
import shlex

# Matches ``watch -n <int> <rest>`` with flexible whitespace.
_WATCH_PATTERN = re.compile(r"^watch\s+-n\s+(\d+)\s+(.+)$", re.IGNORECASE)


def normalize_watch_command(command: str) -> str:
    """Return a platform-native equivalent of a ``watch -n`` command.

    On Windows:
      ``watch -n 5 git status`` →
      ``powershell -c "while(1){git status; Start-Sleep 5}"``

    On POSIX / macOS the command is returned as-is.

    The inner command is escaped with ``shlex.quote`` to reduce the risk of
    shell-injection via the watch interval command.
    """
    if platform.system() != "Windows":
        return command

    stripped = command.strip()
    m = _WATCH_PATTERN.match(stripped)
    if not m:
        return command

    interval_str, inner_cmd = m.group(1), m.group(2)
    # Ensure interval parses as a non-negative integer.
    try:
        interval = int(interval_str)
    except ValueError:
        return command
    if interval < 0:
        return command

    # Escape the inner command so special characters (quotes, $, etc.)
    # inside <cmd> do not break the PowerShell while-loop.
    safe_inner = shlex.quote(inner_cmd)
    # Remove the surrounding quotes added by shlex.quote so the command
    # runs naturally inside the PowerShell script block.  shlex.quote uses
    # single quotes on POSIX-style strings; on Windows the target shell is
    # PowerShell, which still accepts single-quoted string literals.
    if safe_inner.startswith("'") and safe_inner.endswith("'"):
        safe_inner = safe_inner[1:-1]

    return f'powershell -c "while(1){{{safe_inner}; Start-Sleep {interval}}}"'


__all__ = ["normalize_watch_command"]

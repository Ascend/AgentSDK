"""Lightweight read-only check for bash commands.

Splits on pipes and checks each sub-command's leading token against
a known allowlist.  Rejects shell meta-characters that could bypass
the simple token check (back-ticks, $(), etc.).

F-107: Extended with PowerShell read-only command set when
``shell="powershell"``.
"""

from __future__ import annotations

import re
import shlex

READONLY_COMMANDS: frozenset[str] = frozenset(
    [
        # POSIX commands
        "cat",
        "head",
        "tail",
        "wc",
        "stat",
        "strings",
        "hexdump",
        "od",
        "nl",
        "ls",
        "tree",
        "du",
        "exa",
        "eza",
        "grep",
        "rg",
        "find",
        "fd",
        "fdfind",
        "ag",
        "ack",
        "locate",
        "diff",
        "comm",
        "cmp",
        "id",
        "uname",
        "free",
        "df",
        "locale",
        "groups",
        "nproc",
        "basename",
        "dirname",
        "realpath",
        "readlink",
        "cal",
        "uptime",
        "date",
        "cut",
        "paste",
        "tr",
        "column",
        "tac",
        "rev",
        "fold",
        "expand",
        "unexpand",
        "fmt",
        "numfmt",
        "sort",
        "uniq",
        "pwd",
        "whoami",
        "which",
        "type",
        "file",
        "git",
        "true",
        "false",
        "sleep",
        "echo",
        "printf",
        "expr",
        "test",
        "getconf",
        "seq",
        "jq",
        "ps",
        "pgrep",
        "lsof",
        "netstat",
        "ss",
        "sha256sum",
        "sha1sum",
        "md5sum",
        "base64",
        "man",
        "info",
        "help",
        "hostname",
        "tput",
    ]
)

# ---------------------------------------------------------------------------
# F-107: PowerShell read-only command set
# ---------------------------------------------------------------------------

PWSH_READONLY_COMMANDS: frozenset[str] = frozenset(
    [
        # Search / content read
        "select-string",
        "sls",
        "findstr",
        "get-content",
        "gc",
        "type",
        "get-childitem",
        "gci",
        "dir",
        "ls",
        "get-item",
        "gi",
        "get-command",
        "gcm",
        "get-help",
        "help",
        "man",
        "get-process",
        "ps",
        "get-service",
        "get-date",
        "get-location",
        "pwd",
        "gl",
        "get-alias",
        "get-variable",
        "get-psdrive",
        "get-member",
        "gm",
        "get-unique",
        "gu",
        "get-random",
        "get-culture",
        "get-host",
        "get-history",
        "get-hotfix",
        "get-itemproperty",
        "gp",
        "get-pfxcertificate",
        "get-process",
        "get-wmiobject",
        "gwmi",
        "get-ciminstance",
        "gcim",
        "measure-object",
        "measure",
        "out-host",
        "oh",
        "out-default",
        "out-string",
        "compare-object",
        "diff",
        "compare",
        "where-object",
        "where",
        "sort-object",
        "sort",
        "group-object",
        "group",
        "select-object",
        "select",
        "format-table",
        "ft",
        "format-list",
        "fl",
        "format-wide",
        "fw",
        "format-custom",
        "fc",
        "tee-object",
        "write-host",
        "write-output",
        "echo",
        "write-warning",
        "write-verbose",
        "write-debug",
        "true",
        "false",
        "start-sleep",
        "sleep",
        "get-acl",
        "get-pssession",
        "get-pssessioncapability",
        "get-pssnapin",
        "get-module",
        "get-verb",
        "trace-command",
        "test-path",
        "test-connection",
        "test-netconnection",
        "test-wsman",
        "resolve-path",
        "convertfrom-json",
        "convertto-json",
        "convertfrom-csv",
        "convertto-csv",
        "convertfrom-stringdata",
        "export-csv",  # read-only if used without -NoTypeInformation changes
        "import-csv",
        "import-clixml",
    ]
)

_SHELL_METACHARS = re.compile(r"[;&|`$(){}><]")

# Pre-compute for O(1) lookup
_PWSH_READONLY_LOWER = frozenset(c.lower() for c in PWSH_READONLY_COMMANDS)


def is_command_read_only(command: str, shell: str = "bash") -> bool:
    """Return True when *command* is a pipeline of known read-only binaries.

    When *shell* is ``"powershell"``, uses ``PWSH_READONLY_COMMANDS`` instead.
    """
    stripped = command.strip()
    if not stripped:
        return False

    is_pwsh = shell == "powershell"

    for sub in re.split(r"\s*\|\s*", stripped):
        sub = sub.strip()
        if not sub:
            continue
        if _SHELL_METACHARS.search(sub):
            return False
        try:
            tokens = shlex.split(sub, posix=not is_pwsh)
        except ValueError:
            return False
        if not tokens:
            return False
        base = tokens[0].split("/")[-1]
        if base.endswith(".exe"):
            base = base[:-4]
        if is_pwsh:
            if base.lower() not in _PWSH_READONLY_LOWER:
                return False
        elif base not in READONLY_COMMANDS:
            return False
    return True

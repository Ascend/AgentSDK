"""Redaction and secret scanning for telemetry events.

Privacy boundary:

* No prompt / output / transcript / file content ever enters the payload.
* No API key, token, cookie, env var, full shell args, absolute path,
  username, hostname.
* Paths are normalized to ``<project>/relative/...`` or replaced with a
  salted basename hash. Stack frames outside the project roots are
  dropped.
* ``Redactor.scan_secrets`` is the second-pass gate applied to reporter
  markdown; if any suspected secret survives, the report is refused and
  a local ``reporter_blocked.jsonl`` row is appended.

This module is intentionally synchronous and pure — its only state is
``RedactionConfig`` and the project roots it is constructed with. Tests
can construct a redactor against a temp dir and assert behaviour without
monkey-patching.
"""

from __future__ import annotations

import hashlib
import os
import re
import traceback
from dataclasses import dataclass, field
from typing import Any, Final, Iterable

from .events import TelemetryEvent

# Patterns flagged as secrets by ``scan_secrets``. Kept conservative on
# purpose — false positives block a report, false negatives leak data.
# Adding a pattern here is a one-line change but does require updating
# the unit tests in ``tests/telemetry/test_redaction.py``.
_SECRET_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"\bghp_[A-Za-z0-9]{20,}"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._\-+/=]{8,}"),
    re.compile(r"(?i)password\s*[:=]\s*['\"]?[^\s'\"]{4,}"),
    re.compile(r"(?i)api[_-]?key\s*[:=]\s*['\"]?[A-Za-z0-9._\-+/=]{8,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
)

_REDACTION_PLACEHOLDER: Final[str] = "[REDACTED]"

# Keys that may never survive inside the analytics ``extra`` dict
# (case-insensitive). Same surface as the prompt/output block-list used
# by ``redact_event`` so a caller cannot smuggle user data into the
# payload via the analytics metadata bridge.
_BLOCKED_EXTRA_KEYS: Final[frozenset[str]] = frozenset(
    {"prompt", "output", "transcript", "messages", "input", "response"}
)


@dataclass(frozen=True)
class RedactionConfig:
    """Toggles that callers can opt into.

    Defaults are privacy-first — everything sensitive is off. Tests
    can flip individual toggles to exercise the path.
    """

    include_command_name: bool = True
    include_command_args: bool = False
    include_absolute_paths: bool = False
    include_stacktrace: bool = True
    include_prompts: bool = False
    include_outputs: bool = False
    stacktrace_max_lines: int = 20
    secret_hash_salt: str = "clawcodex-telemetry-v1"


# Whitelist of command names. Anything outside this set is bucketed as
# ``"other"`` so the payload never carries raw argv tokens.
_COMMAND_WHITELIST: Final[frozenset[str]] = frozenset(
    {
        "tui",
        "repl",
        "headless",
        "login",
        "config",
        "orchestrator",
        "mcp",
        "daemon",
        "doctor",
        "autonomy",
        "schedule",
        "version",
        "help",
        "telemetry",
        "viz",
    }
)


@dataclass
class Redactor:
    """Stateless redactor: ``redact_event`` and ``redact_text`` are pure."""

    cfg: RedactionConfig
    project_roots: tuple[str, ...] = field(default_factory=tuple)

    def _normalize_path(self, value: str) -> str:
        """Map an absolute path to ``<project>/...`` or ``path:<hash>``.

        * If ``value`` lives under a project root, return
          ``<project>/<relative>`` (or ``<project>`` when at the root).
        * If ``include_absolute_paths`` is False and the path is not in
          any project root, return the salted hash digest of the path.
        * If ``include_absolute_paths`` is True, still strip user/host
          prefixes that the path may carry (handled in callers).
        """
        if not value:
            return ""
        value = str(value)
        for root in self.project_roots:
            if not root:
                continue
            root = os.path.normpath(root)
            try:
                rel = os.path.relpath(value, root)
            except ValueError:
                continue
            if not rel.startswith(".."):
                return f"<project>/{rel.replace(os.sep, '/')}"
        if self.cfg.include_absolute_paths:
            return value
        digest = hashlib.sha1(
            (self.cfg.secret_hash_salt + "|" + value).encode("utf-8"),
            usedforsecurity=False,
        ).hexdigest()[:12]
        return f"path:{digest}"

    def _normalize_command(self, command: Any) -> str:
        if not isinstance(command, str) or not command:
            return "unknown"
        head = command.strip().split(maxsplit=1)[0]
        head = os.path.basename(head) or head
        head_lower = head.lower()
        if head_lower in _COMMAND_WHITELIST:
            return head_lower
        return "other"

    def _truncate_stacktrace(self, exc: BaseException | None) -> list[str]:
        if not self.cfg.include_stacktrace or exc is None:
            return []
        try:
            tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
        except Exception:
            return []
        lines: list[str] = []
        in_project_block = False
        for line in tb:
            if not line:
                continue
            is_project_line = bool(self.project_roots and any(marker in line for marker in self.project_roots))
            # Track whether we are inside a project-owned frame block.
            if is_project_line:
                in_project_block = True
            # Always keep the "Traceback" header and the final error line.
            is_header = line.startswith("Traceback")
            # Determine if this is the error line (e.g. "ValueError: ..." or "SyntaxError: ...").
            is_error = "Error" in line.split(":")[0] if ":" in line else False
            # Keep indented context lines (code, caret) that follow a project frame.
            is_context = in_project_block and (line.startswith(" ") or line.startswith("\t"))

            if is_project_line or is_header or is_error or is_context:
                if is_project_line:
                    safe_line = line.rstrip()
                    for root in self.project_roots:
                        if not root:
                            continue
                        norm_root = os.path.normpath(root)
                        if norm_root in safe_line:
                            safe_line = safe_line.replace(norm_root, "<project>")
                            break
                    lines.append(safe_line)
                else:
                    lines.append(line.rstrip())
                if len(lines) >= self.cfg.stacktrace_max_lines:
                    break
        return lines

    def redact_event(self, event: TelemetryEvent) -> TelemetryEvent:
        """Return a copy of ``event`` with sensitive fields scrubbed.

        The original ``event`` is not mutated so call sites can keep
        references in higher-level scopes.
        """
        new_fields = self._redact_fields(dict(event.fields))
        if not self.cfg.include_prompts:
            for key in ("prompt", "input", "messages", "transcript"):
                new_fields.pop(key, None)
        if not self.cfg.include_outputs:
            for key in ("output", "response", "result_text", "tool_output"):
                new_fields.pop(key, None)
        if "stacktrace" in new_fields and not self.cfg.include_stacktrace:
            new_fields.pop("stacktrace", None)
        return TelemetryEvent(
            type=event.type,
            timestamp=event.timestamp,
            session_id=event.session_id,
            schema_version=event.schema_version,
            fields=new_fields,
        )

    def _redact_fields(self, fields: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in fields.items():
            out[key] = self._redact_value(key, value)
        return out

    def _redact_value(self, key: str, value: Any) -> Any:
        if key in ("command", "command_name", "command_args"):
            return self._normalize_command(value)
        if key in ("cwd", "working_dir", "project_root", "path", "file_path"):
            if isinstance(value, str):
                return self._normalize_path(value)
            if isinstance(value, (list, tuple)):
                return [self._normalize_path(str(v)) for v in value]
            return self._normalize_path(str(value))
        if key == "env" and isinstance(value, dict):
            return {str(k): _REDACTION_PLACEHOLDER for k in value.keys()}
        if key == "stacktrace" and isinstance(value, (list, tuple)):
            return [str(line) for line in value][: self.cfg.stacktrace_max_lines]
        # ``extra`` is a caller-supplied dict from
        # ``SessionAnalyticsMetadata``. It must NEVER smuggle prompts,
        # outputs, transcripts, or messages into the payload. Recurse
        # through the dict after dropping blocked keys, so a sneaky
        # nested ``{"prompt": "..."}`` is also caught.
        if key == "extra" and isinstance(value, dict):
            safe: dict[str, Any] = {}
            for sub_key, sub_value in value.items():
                if sub_key.lower() in _BLOCKED_EXTRA_KEYS:
                    continue
                safe[str(sub_key)] = self._redact_value(str(sub_key), sub_value)
            return safe
        # v2 ``fingerprint`` is a structured dict whose ``hash``
        # field is a 16-char digest (low risk) and whose ``version`` /
        # ``method`` fields are low-cardinality identifiers. Run the
        # hash through ``redact_text`` so any embedded secret pattern
        # is caught; recurse on the rest.
        if key == "fingerprint" and isinstance(value, dict):
            safe_fp: dict[str, Any] = {}
            for sub_key, sub_value in value.items():
                if sub_key == "hash" and isinstance(sub_value, str):
                    safe_fp[sub_key] = self.redact_text(sub_value)
                else:
                    safe_fp[sub_key] = self._redact_value(str(sub_key), sub_value)
            return safe_fp
        if isinstance(value, str):
            return self.redact_text(value)
        if isinstance(value, dict):
            return {str(k): self._redact_value(str(k), v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._redact_value(key, v) for v in value]
        return value

    def redact_text(self, text: str) -> str:
        """Apply message-level secret patterns to ``text`` in place.

        ``scan_secrets`` is the gating check used before writing a
        report; this method is the in-event scrubber used during normal
        local writes.
        """
        if not text:
            return text
        for pattern in _SECRET_PATTERNS:
            text = pattern.sub(_REDACTION_PLACEHOLDER, text)
        return text

    def scan_secrets(self, text: str) -> list[str]:
        """Return a list of secret-pattern names that matched ``text``.

        Empty list means ``text`` is safe to include in a report. The
        caller is expected to refuse the report when this returns a
        non-empty list and to record the kind in the
        ``reporter_blocked.jsonl`` log.
        """
        if not text:
            return []
        hits: list[str] = []
        for pattern in _SECRET_PATTERNS:
            if pattern.search(text):
                hits.append(pattern.pattern)
        return hits

    def truncate_stacktrace(self, exc: BaseException | None) -> list[str]:
        """Public wrapper around the private ``_truncate_stacktrace``."""
        return self._truncate_stacktrace(exc)


def normalize_path_for(
    value: str,
    *,
    project_roots: Iterable[str] = (),
    include_absolute_paths: bool = False,
    salt: str = RedactionConfig().secret_hash_salt,
) -> str:
    """Module-level convenience for the CLI ``status`` preview path."""
    return Redactor(
        RedactionConfig(include_absolute_paths=include_absolute_paths, secret_hash_salt=salt),
        tuple(project_roots),
    )._normalize_path(value)

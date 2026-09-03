"""In-memory reporter used for tests and the CLI ``preview`` subcommand."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Final

from .base import Reporter

_MAX_HISTORY: Final[int] = 32


@dataclass
class _RenderedRow:
    date: str
    text: str


class DryRunReporter(Reporter):
    """Records the last N rendered payloads in memory.

    No I/O, no secret scanning — but :meth:`scan_artifact` is exposed
    so the CLI ``preview`` subcommand can still run the secret scan
    over what *would* be uploaded.
    """

    def __init__(self, max_history: int = _MAX_HISTORY) -> None:
        self._history: deque[_RenderedRow] = deque(maxlen=max_history)
        self.last_rendered: str = ""
        self.last_date: str = ""

    def render(self, summary: dict[str, Any], date: str) -> str:
        return _render_markdown(summary, date)

    def emit(self, rendered: str, *, date: str) -> bool:
        self.last_rendered = rendered
        self.last_date = date
        self._history.append(_RenderedRow(date=date, text=rendered))
        return True

    # -- helpers --------------------------------------------------------

    def history(self) -> list[_RenderedRow]:
        return list(self._history)

    def clear(self) -> None:
        self._history.clear()
        self.last_rendered = ""
        self.last_date = ""


def _render_markdown(summary: dict[str, Any], date: str) -> str:
    """Render the telemetry daily summary as Markdown.

    Pure function. Mirrors the daily summary Issue body template
    but is fully
    local-only — no issue creation, no remote upload.

    When the report contains crashes or errors the error section is
    placed at the top (before the stats) and the title is prefixed
    with a warning indicator so the Issue is immediately identifiable
    as an error report.
    """
    # ``totals`` was added with per-session reporting.  Falling back to the
    # root preserves rendering of summaries persisted by older versions.
    totals = summary.get("totals") if isinstance(summary.get("totals"), dict) else summary
    version = summary.get("version", "unknown")
    sessions = totals.get("sessions", 0)
    commands = totals.get("commands", 0)
    platforms = totals.get("platforms", {}) or {}
    exit_status_counts = totals.get("exit_status_counts", {}) or {}
    duration = totals.get("duration_s", {}) or {}
    crashes = totals.get("crashes", {}) or {}
    top_crashes = crashes.get("top", []) if isinstance(crashes, dict) else []
    has_crashes = bool(top_crashes)

    lines: list[str] = []
    title_prefix = "⚠️ " if has_crashes else ""
    lines.append(f"# {title_prefix}ClawCodex Telemetry — {date}")
    lines.append("")

    # Error section first (when present) — the Issue is only pushed when
    # there are crashes, so this is the primary content.
    if has_crashes:
        lines.append("## Error report")
        lines.append("")
        lines.append("| Fingerprint | Count | Error class | First seen | Last seen |")
        lines.append("|-------------|------:|-------------|------------|-----------|")
        for row in top_crashes:
            lines.append(
                "| {fp} | {count} | {cls} | {first} | {last} |".format(
                    fp=row.get("fingerprint", "?"),
                    count=row.get("count", 0),
                    cls=row.get("error_class", "?"),
                    first=row.get("first_seen_iso", "?"),
                    last=row.get("last_seen_iso", "?"),
                )
            )
            # Include a representative stacktrace for debugging.
            # Stacktraces are redacted at record time (paths normalized,
            # secrets scrubbed) and then verified by scan_secrets before
            # the report is pushed remotely.
            st = row.get("stacktrace")
            if st:
                lines.append("")
                lines.append("```")
                for frame in st:
                    lines.append(frame.rstrip())
                lines.append("```")
                lines.append("")
        lines.append("")

    # Stats section secondary — provides context around the error.
    lines.append("## Daily stats")
    lines.append("")
    lines.append(f"- Version: {version}")
    lines.append(f"- Sessions: {sessions}")
    lines.append(f"- Command runs: {commands}")
    if duration:
        total = duration.get("total", 0)
        samples = duration.get("samples", 0)
        lines.append(f"- Duration total (s): {total} (samples: {samples})")
    if exit_status_counts:
        lines.append(
            "- Exit status counts: " + ", ".join(f"{key}={value}" for key, value in sorted(exit_status_counts.items()))
        )
    if platforms:
        lines.append("- Platforms: " + ", ".join(f"{k} {v}" for k, v in sorted(platforms.items())))
    providers = totals.get("providers", {}) or {}
    if providers:
        lines.append("- Providers: " + ", ".join(f"{k} {v}" for k, v in sorted(providers.items())))
    turns = totals.get("turns", {}) or {}
    if turns.get("total", 0):
        lines.append(f"- Turns: {turns.get('total', 0)} (failed: {turns.get('failed', 0)})")
    usage = totals.get("usage", {}) or {}
    token_total = sum(int(usage.get(key, 0) or 0) for key in ("input_tokens", "output_tokens"))
    if token_total:
        lines.append(
            f"- Tokens: input={usage.get('input_tokens', 0)}, output={usage.get('output_tokens', 0)}, "
            f"cache-read={usage.get('cache_read_tokens', 0)}"
        )
    if usage.get("cost_usd", 0):
        lines.append(f"- Estimated cost (USD): {usage['cost_usd']}")
    tool_timing = (totals.get("tools", {}) or {}).get("duration_s", {}) or {}
    if tool_timing.get("samples", 0):
        lines.append(f"- Tool time (s): {tool_timing.get('total', 0)}")
    lines.append("")

    # Show top meaningful commands (excludes infrastructure noise).
    top_commands = totals.get("top_commands", []) or []
    _NOISE_COMMANDS = frozenset(
        {
            "version",
            "help",
            "telemetry",
            "config",
            "login",
            "mcp",
            "daemon",
            "doctor",
            "orchestrator",
            "viz",
            "autonomy",
            "schedule",
            "other",
        }
    )
    sig_commands = [e for e in top_commands if e.get("name") not in _NOISE_COMMANDS][:10]
    cmd_failures = totals.get("command_failure", {}) or {}
    if sig_commands:
        lines.append("## Top commands")
        lines.append("*(meaningful user-facing commands; infrastructure noise excluded)*")
        lines.append("")
        for entry in sig_commands:
            name = entry.get("name", "?")
            total = entry.get("count", 0)
            failed = cmd_failures.get(name, 0)
            parts = [f"**{name}**: {total}"]
            if failed:
                parts.append(f"({failed} failed)")
            lines.append("- " + " ".join(parts))
    elif totals.get("exit_status_counts", {}).get("error", 0) > 0:
        err_count = totals["exit_status_counts"]["error"]
        lines.append(f"- Error exits: {err_count}")

    session_stats = summary.get("session_stats", [])
    if isinstance(session_stats, list) and session_stats:
        lines.append("")
        lines.append("## Session statistics")
        lines.append("")
        lines.append(
            "| Session | Commands | Turns | Tokens | Cost (USD) | Tool time (s) | Duration (s) | Crashes | Exit statuses |"
        )
        lines.append(
            "|---------|---------:|------:|-------:|-----------:|--------------:|-------------:|--------:|---------------|"
        )
        for stat in session_stats:
            if not isinstance(stat, dict):
                continue
            tools = stat.get("tools", {}) or {}
            session_duration = stat.get("duration_s", {}) or {}
            session_crashes = stat.get("crashes", {}) or {}
            session_usage = stat.get("usage", {}) or {}
            session_tokens = int(session_usage.get("input_tokens", 0) or 0) + int(
                session_usage.get("output_tokens", 0) or 0
            )
            session_tool_timing = tools.get("duration_s", {}) or {}
            session_turns = stat.get("turns", {}) or {}
            statuses = stat.get("exit_status_counts", {}) or {}
            status_text = ", ".join(f"{key}={value}" for key, value in sorted(statuses.items())) or "—"
            lines.append(
                "| {session_id} | {commands} | {turns} | {tokens} | {cost} | {tool_time} | {duration} | {crashes} | {statuses} |".format(
                    session_id=stat.get("session_id", "unknown"),
                    commands=stat.get("commands", 0),
                    turns=session_turns.get("total", 0),
                    tokens=session_tokens,
                    cost=session_usage.get("cost_usd", 0),
                    tool_time=session_tool_timing.get("total", 0),
                    duration=session_duration.get("total", 0),
                    crashes=session_crashes.get("total", 0),
                    statuses=status_text,
                )
            )

    lines.append("")
    lines.append("No prompts, outputs, file contents, API keys, environment variables, or absolute paths are included.")
    return "\n".join(lines) + "\n"

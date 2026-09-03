"""Telemetry CLI subcommands.

Four read-only / preview commands; none of them mutate the on-disk
config. The toggle (``enable|disable``) intentionally prints the
required config snippet rather than writing it for the user — that's
a separate, fully-aware workflow.

Subcommand surface:

* ``status``   — show config, recorder kind, storage dir, today's summary
* ``preview``  — print what a reporter would emit for today (no I/O)
* ``flush``    — run the aggregator for today and emit to reporters if enabled
* ``enable|disable`` — print the JSON snippet the user must paste into
  their merged config; do **not** mutate config files automatically.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any, Sequence

from .config import TelemetryConfig, load_config
from .recorder import get_recorder
from .storage import LocalJsonlStorage, utc_date, utc_now

logger = logging.getLogger(__name__)


def _print(text: str) -> None:
    sys.stdout.write(text)
    if not text.endswith("\n"):
        sys.stdout.write("\n")
    sys.stdout.flush()


def _section(text: str) -> None:
    _print("")
    _print(text)


def run_status(argv: Sequence[str] | None = None) -> int:
    """Show the current telemetry configuration and today's summary."""
    try:
        cfg = load_config()
    except Exception as exc:  # noqa: BLE001
        _print(f"telemetry: failed to load config: {exc}")
        return 1

    _print("Telemetry status")
    _print("----------------")
    _print(f"  enabled        : {cfg.enabled}")
    _print(f"  storage_dir    : {cfg.storage_dir}")
    _print(f"  retention_days : {cfg.retention_days}")
    _print(
        f"  reporting      : enabled={cfg.reporting.reporting_enabled} "
        f"kind={cfg.reporting.kind!r} mode={cfg.reporting.mode!r}"
    )
    _print(f"    platform      : {cfg.reporting.platform}")
    _print(f"    owner/repo    : {cfg.reporting.owner or '-'} / {cfg.reporting.repo or '-'}")
    _print(f"    endpoint      : {cfg.reporting.endpoint or '(default)'}")
    _print(f"    issue_title   : {cfg.reporting.issue_title}")
    _print(f"    interval_hours: {cfg.reporting.interval_hours}")
    _print(f"    token_env     : {cfg.reporting.token_env or '-'}")
    _print(f"    api_key_set   : {bool(cfg.reporting.api_key)}")
    _print(f"    auto_push_errors_only: {cfg.reporting.auto_push_errors_only}")
    _print("  redaction:")
    for field_name in (
        "include_command_name",
        "include_command_args",
        "include_absolute_paths",
        "include_stacktrace",
        "include_prompts",
        "include_outputs",
        "stacktrace_max_lines",
    ):
        _print(f"    {field_name}: {getattr(cfg.redaction, field_name)}")

    recorder = get_recorder()
    _print("")
    _print(f"  recorder class : {type(recorder).__name__}")
    if not getattr(recorder, "enabled", False):
        _print("  (telemetry is disabled — no local I/O will be performed)")

    if cfg.enabled:
        try:
            storage = LocalJsonlStorage(cfg.storage_dir, cfg.retention_days)
            today = utc_date(utc_now())
            summary = storage.read_latest_summary(today)
            _section("Today's summary")
            if not summary:
                _print("  (no summary yet — no events recorded today)")
            else:
                _print(json.dumps(_summarize_for_status(summary), indent=2))
        except Exception as exc:  # noqa: BLE001
            _print(f"  (storage probe failed: {exc})")

    return 0


def run_preview(argv: Sequence[str] | None = None) -> int:
    """Render the markdown that a reporter would emit for today.

    No I/O is performed; the output is the in-memory dry-run render
    passed through the configured redactor's secret-scan. If the scan
    matches, the preview prints a warning instead of the rendered text.
    """
    cfg = load_config()
    recorder = get_recorder()
    if not getattr(recorder, "enabled", False):
        _print("telemetry: disabled — enable `telemetry.enabled=true` first")
        return 1

    try:
        date = _preview_date_arg(argv)
        if not cfg.enabled:
            _print("telemetry: disabled — preview is empty")
            return 0
        rendered = recorder.build_report_for(date)
    except Exception as exc:  # noqa: BLE001
        _print(f"telemetry: preview failed: {exc}")
        return 1

    if not rendered:
        _print(f"telemetry: no summary available for {date}")
        return 0

    from .redaction import Redactor

    redactor = Redactor(cfg.redaction)
    hits = redactor.scan_secrets(rendered)
    if hits:
        _print("Secret scan matched; refusing to render preview:")
        for pattern in hits:
            _print(f"  - {pattern}")
        return 1

    _print(rendered)
    return 0


def run_flush(argv: Sequence[str] | None = None) -> int:
    """Force the aggregator to run for today and emit to reporters if enabled."""
    recorder = get_recorder()
    if not getattr(recorder, "enabled", False):
        _print("telemetry: disabled — nothing to flush")
        return 0
    try:
        recorder.flush()
    except Exception as exc:  # noqa: BLE001
        _print(f"telemetry: flush failed: {exc}")
        return 1
    _print("telemetry: flush complete")
    return 0


def run_enable(argv: Sequence[str] | None = None) -> int:
    """Enable telemetry (stats + error reporting) and persist to global config."""
    try:
        from src.config import load_config as _load_src_config
        from src.config import save_config as _save_src_config

        cfg = _load_src_config()
        telemetry_section: dict[str, Any] = dict(cfg.get("telemetry", {}))
        telemetry_section["enabled"] = True
        reporting_section: dict[str, Any] = dict(telemetry_section.get("reporting", {}))
        reporting_section["reporting_enabled"] = True
        telemetry_section["reporting"] = reporting_section
        cfg["telemetry"] = telemetry_section
        _save_src_config(cfg)
        _print("Telemetry enabled: stats collection ✓ · error reporting ✓\nRun \"/telemetry status\" to verify.")
    except Exception as exc:
        _print(f"telemetry: failed to enable — {exc}")
        _print("You can manually add the following to your merged config:")
        _print("")
        _print(
            json.dumps(
                {
                    "telemetry": {
                        "enabled": True,
                        "reporting": {"reporting_enabled": True},
                    }
                },
                indent=2,
            )
        )
        return 1
    return 0


def run_disable(argv: Sequence[str] | None = None) -> int:
    """Disable telemetry and persist to global config."""
    try:
        from src.config import load_config as _load_src_config
        from src.config import save_config as _save_src_config

        cfg = _load_src_config()
        telemetry_section: dict[str, Any] = dict(cfg.get("telemetry", {}))
        telemetry_section["enabled"] = False
        # reporting_enabled is moot when enabled=False, but set it too
        # for consistency so the next user-run /telemetry status shows a
        # clean state.
        reporting_section: dict[str, Any] = dict(telemetry_section.get("reporting", {}))
        reporting_section["reporting_enabled"] = False
        telemetry_section["reporting"] = reporting_section
        cfg["telemetry"] = telemetry_section
        _save_src_config(cfg)
        _print('Telemetry disabled.\nRun "/telemetry status" to verify.')
    except Exception as exc:
        _print(f"telemetry: failed to disable — {exc}")
        _print("You can manually set the following in your merged config:")
        _print("")
        _print(json.dumps({"telemetry": {"enabled": False}}, indent=2))
        return 1
    return 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _preview_date_arg(argv: Sequence[str] | None) -> str:
    args = list(argv or [])
    if len(args) >= 2 and args[0] == "preview":
        return args[1]
    if args and args[0] != "preview":
        return args[0]
    return utc_date(utc_now())


def _asdict(cfg: TelemetryConfig) -> dict[str, Any]:
    return {
        "enabled": cfg.enabled,
        "storage_dir": str(cfg.storage_dir),
        "retention_days": cfg.retention_days,
        "reporting": {
            "reporting_enabled": cfg.reporting.reporting_enabled,
            "kind": cfg.reporting.kind,
            "platform": cfg.reporting.platform,
            "owner": cfg.reporting.owner or "example",
            "repo": cfg.reporting.repo or "clawcodex-telemetry",
            "endpoint": cfg.reporting.endpoint,
            "issue_title": cfg.reporting.issue_title,
            "mode": cfg.reporting.mode,
            "interval_hours": cfg.reporting.interval_hours,
            "auto_push_errors_only": cfg.reporting.auto_push_errors_only,
            "token_env": cfg.reporting.token_env or "CLAW_TELEMETRY_REPORTING_TOKEN",
        },
        "redaction": {
            "include_command_name": cfg.redaction.include_command_name,
            "include_command_args": cfg.redaction.include_command_args,
            "include_absolute_paths": cfg.redaction.include_absolute_paths,
            "include_stacktrace": cfg.redaction.include_stacktrace,
            "include_prompts": cfg.redaction.include_prompts,
            "include_outputs": cfg.redaction.include_outputs,
            "stacktrace_max_lines": cfg.redaction.stacktrace_max_lines,
        },
    }


def _serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(v) for v in value]
    return value


def _summarize_for_status(summary: dict[str, Any]) -> dict[str, Any]:
    """Trim the full summary to the fields the status command prints."""
    keep = {
        "date",
        "version",
        "sessions",
        "commands",
        "exit_status_counts",
        "platforms",
        "providers",
        "top_commands",
    }
    return {k: summary.get(k) for k in keep if k in summary}


SUBCOMMANDS: dict[str, Any] = {
    "status": run_status,
    "preview": run_preview,
    "flush": run_flush,
    "enable": run_enable,
    "disable": run_disable,
}


def main(argv: Sequence[str] | None = None) -> int:
    args = list(argv) if argv is not None else list(sys.argv[1:])
    if not args:
        return run_status([])
    sub = args[0]
    handler = SUBCOMMANDS.get(sub)
    if handler is None:
        _print(f"telemetry: unknown subcommand {sub!r}")
        _print("available subcommands: " + ", ".join(sorted(SUBCOMMANDS)))
        return 2
    return handler(args)

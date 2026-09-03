"""Global exception hooks.

The hooks wrap ``sys.excepthook`` and the asyncio loop exception
handler. When a previously-unhandled exception escapes the program, the
hooks ask the recorder to ``record_error`` with a synthesized session
id (the bootstrap session id if available, else a fresh uuid4).

The wrappers always defer to the previous hook so the interpreter's
default behavior (traceback to stderr, exit code, etc.) is preserved.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import threading
from typing import Any

logger = logging.getLogger(__name__)

_PREVIOUS_HOOKS: dict[str, Any] = {}
_INSTALLED = False
_LOCK = threading.Lock()


def _safe_session_id() -> str:
    try:
        from src.bootstrap.state import get_session_id  # type: ignore[import-not-found]

        sid = get_session_id()
        if isinstance(sid, str) and sid:
            return sid
    except Exception:  # nosec B110
        # Session-id lookup is optional inside a crash hook.
        pass
    import uuid

    return uuid.uuid4().hex


def _emit(exc: BaseException) -> None:
    """Record an error and immediately flush to remote reporter.

    Runs inside sys.excepthook, BEFORE atexit shutdown starts, so
    asyncio HTTP requests to GitCode/GitHub still work reliably.
    """
    try:
        from .recorder import get_recorder

        recorder = get_recorder()
        if not getattr(recorder, "enabled", False):
            return
        recorder.record_error(session_id=_safe_session_id(), exc=exc)
        # Flush immediately so the crash appears in the remote Issue
        # before the process exits. This runs inside sys.excepthook
        # (before atexit), so async HTTP is still available.
        try:
            recorder.flush()
            # Notify user on stderr (runs before traceback is printed)
            try:
                from .storage import LocalJsonlStorage

                storage = LocalJsonlStorage(recorder.config.storage_dir, recorder.config.retention_days)
                from .reporters.issue import issue_cursor_name

                cursor = storage.read_reporter_cursor(issue_cursor_name(recorder.config.reporting))
                issue_id = cursor.get("issue_id", "")
                if issue_id:
                    platform = cursor.get("platform", "")
                    owner = cursor.get("owner", "")
                    repo = cursor.get("repo", "")
                    url = {
                        "gitcode": f"https://gitcode.com/{owner}/{repo}/issues/{issue_id}",
                        "github": f"https://github.com/{owner}/{repo}/issues/{issue_id}",
                        "gitee": f"https://gitee.com/{owner}/{repo}/issues/{issue_id}",
                    }.get(platform, "")
                    if url:
                        sys.stderr.write("\n\033[1;33m⚠ Telemetry: error report pushed to %s\033[0m\n" % url)
            except Exception:  # nosec B110
                # Reporting diagnostics must not replace the original crash.
                pass
        except Exception as flush_exc:  # noqa: BLE001
            logger.debug("telemetry: post-error flush failed: %s", flush_exc)
    except Exception as exc_inner:  # noqa: BLE001
        logger.debug("telemetry: hook emit failed: %s", exc_inner)


def _wrap_excepthook(previous: Any) -> Any:
    def _hook(exc_type, exc_value, exc_tb):  # type: ignore[no-untyped-def]
        if isinstance(exc_value, BaseException):
            try:
                _emit(exc_value)
            except Exception:  # noqa: BLE001  # nosec B110
                # The original unhandled exception remains authoritative.
                pass
        if previous is not None:
            try:
                return previous(exc_type, exc_value, exc_tb)
            except Exception:  # noqa: BLE001  # nosec B110
                # A broken previous hook must not mask the original failure.
                pass
        return None

    return _hook


def _wrap_threading_excepthook(previous: Any) -> Any:
    def _hook(args):  # type: ignore[no-untyped-def]
        exc = getattr(args, "exc_value", None)
        if isinstance(exc, BaseException):
            try:
                _emit(exc)
            except Exception:  # noqa: BLE001  # nosec B110
                # The original thread exception remains authoritative.
                pass
        if previous is not None:
            try:
                return previous(args)
            except Exception:  # noqa: BLE001  # nosec B110
                # A broken previous hook must not mask the original failure.
                pass
        return None

    return _hook


def _wrap_asyncio_handler(previous: Any) -> Any:
    def _handler(loop: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
        exc = context.get("exception")
        if isinstance(exc, BaseException):
            try:
                _emit(exc)
            except Exception:  # noqa: BLE001  # nosec B110
                # The original event-loop exception remains authoritative.
                pass
        if previous is not None:
            try:
                return previous(loop, context)
            except Exception:  # noqa: BLE001  # nosec B110
                # A broken previous handler must not mask the original failure.
                pass
        return None

    return _handler


def install_exception_hooks() -> None:
    """Install wrapped global exception hooks.

    Idempotent — calling this twice does not double-wrap. Safe to call
    from :func:`src.init.init` after ``setup_graceful_shutdown``.

    Also installs the analytics → telemetry bridge so existing
    ``log_event()`` calls in image / PDF pipelines are forwarded into
    the live recorder. When telemetry is disabled the recorder is a
    no-op and the bridge becomes one too, so this is safe to leave
    installed permanently.
    """
    global _INSTALLED
    with _LOCK:
        if _INSTALLED:
            return
        try:
            _PREVIOUS_HOOKS["sys.excepthook"] = sys.excepthook
            sys.excepthook = _wrap_excepthook(_PREVIOUS_HOOKS["sys.excepthook"])
        except Exception as exc:  # noqa: BLE001
            logger.debug("telemetry: wrap sys.excepthook failed: %s", exc)
        try:
            _PREVIOUS_HOOKS["threading.excepthook"] = threading.excepthook
            threading.excepthook = _wrap_threading_excepthook(_PREVIOUS_HOOKS["threading.excepthook"])
        except Exception as exc:  # noqa: BLE001
            logger.debug("telemetry: wrap threading.excepthook failed: %s", exc)
        _INSTALLED = True
    try:
        from .bridge import install_analytics_bridge

        install_analytics_bridge()
    except Exception as exc:  # noqa: BLE001
        logger.debug("telemetry: install analytics bridge failed: %s", exc)


def install_asyncio_hook(loop: asyncio.AbstractEventLoop | None = None) -> None:
    """Install a wrapped asyncio loop exception handler.

    Safe to call repeatedly; the second invocation overwrites the
    previous wrap but still chains to the original handler.
    """
    try:
        target = loop or asyncio.get_event_loop()
    except RuntimeError:
        return
    if target is None:
        return
    previous = target.get_exception_handler()
    target.set_exception_handler(_wrap_asyncio_handler(previous))


def uninstall_exception_hooks() -> None:
    """Restore the previous hooks. Primarily for tests."""
    global _INSTALLED
    with _LOCK:
        if not _INSTALLED:
            return
        try:
            sys.excepthook = _PREVIOUS_HOOKS.get("sys.excepthook", sys.excepthook)
        except Exception:  # noqa: BLE001  # nosec B110
            # Hook restoration is best-effort during teardown.
            pass
        try:
            threading.excepthook = _PREVIOUS_HOOKS.get("threading.excepthook", threading.excepthook)
        except Exception:  # noqa: BLE001  # nosec B110
            # Hook restoration is best-effort during teardown.
            pass
        _PREVIOUS_HOOKS.clear()
        _INSTALLED = False

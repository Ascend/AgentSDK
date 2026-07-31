"""dialogue — ``/dialogue`` command (F-65 Voice Dialogue Mode).

Symmetric to ``/voice`` (F-64) and ``/tts`` (F-64 P64-E): toggles
full-duplex voice dialogue mode on/off, selects the dialogue backend,
sets voice / modality preferences, and reports status.

F-65 differs from F-64 in two important ways:

* **No "say" or one-shot preview** — dialogue is session-only; an MVP
  preview would require spinning up a WebSocket, sending fake audio,
  and waiting for an LLM roundtrip. That's too expensive to fire from
  a CLI sub-command. The plan documents ``/dialogue start`` as the only
  entry point.
* **Three knobs visible**: provider (backend), voice (TTS voice id),
  modality (text-only vs text+audio). ``/dialogue mode audio`` swaps
  output modality without restarting the underlying session; the
  command only persists the preference (the actual session is
  started/stopped by ``start``/``stop``).

Persisted state lives in ``settings.dialogue_enabled`` /
``settings.dialogue_provider`` / ``settings.dialogue_voice`` /
``settings.dialogue_modality``, written via :func:`src.config
.set_dialogue_*` which invalidate the settings cache so the next
``get_settings()`` reflects the change mid-session — same pattern as
the F-64 ``/voice`` and F-64 ``/tts`` commands.

Usage
-----
* ``/dialogue`` — toggle dialogue mode on/off (current provider kept).
* ``/dialogue minimax`` — enable dialogue mode + select MiniMax Realtime
  (the only current P65-A adapter).
* ``/dialogue off`` — disable dialogue mode (provider retained).
* ``/dialogue mode <text|audio>`` — set output modality for the next
  session.
* ``/dialogue voice <name>`` — set the backend-specific TTS voice id.
* ``/dialogue start`` — open a live session, route audio.
* ``/dialogue stop`` — end the session.
* ``/dialogue status`` — show current state + per-layer availability.
* ``/dialogue help`` — usage text.

Design decisions
----------------
* ``/dialogue <provider>`` flips the master switch and the provider
  atomically, mirroring ``/voice anthropic`` semantics. ``/dialogue off``
  only flips the switch.
* ``start`` / ``stop`` carry no business logic in this MVP beyond
  composing the provider from the registry and reporting success /
  failure. A full implementation would tie into the agent loop (push
  transcripts to the agent, write replies via ``send_text``); that
  integration is the responsibility of the dialogue session manager
  (P65-B) coupled with the REPL transport. Here we surface a clean
  error if the session can't be assembled (missing credentials, etc.).
* Follows the project's :class:`LocalCommand` convention (same shape
  as ``/voice``, ``/tts``, ``/cost``): a plain ``LocalCommand`
  bound to a free ``dialogue_command_call`` function via
  :meth:`set_call`.
"""

from __future__ import annotations

import asyncio
import logging

from .types import CommandContext, LocalCommand, LocalCommandResult
from src.config import (
    set_dialogue_enabled,
    set_dialogue_modality,
    set_dialogue_provider,
    set_dialogue_voice,
)

logger = logging.getLogger(__name__)

__all__ = ["DIALOGUE_COMMAND", "dialogue_command_call", "DIALOGUE_PROVIDERS"]

# Built-in dialogue provider whitelist. Mirrors the registry's builtins;
# the registry is the source of truth for what's actually instantiable.
DIALOGUE_PROVIDERS: tuple[str, ...] = ("minimax", "openai-realtime")

_VALID_MODALITIES: tuple[str, ...] = ("text", "audio")


_HELP = (
    "Usage: /dialogue [minimax|off|mode <text|audio>|voice <name>"
    "|start|stop|status|help]\n\n"
    "Full-duplex voice dialogue enables simultaneous voice input + "
    "voice output with barge-in support. The server-side stack "
    "(ASR + LLM + TTS) runs in a single WebSocket so end-to-end "
    "latency stays below ~2s on MiniMax.\n\n"
    "Commands:\n"
    "- /dialogue                  Toggle dialogue mode on/off\n"
    "- /dialogue minimax          Enable with MiniMax Realtime backend\n"
    "- /dialogue off              Disable dialogue mode\n"
    "- /dialogue mode <text|audio> Output modality: text-only or audio+text\n"
    "- /dialogue voice <name>     Set the backend-specific TTS voice\n"
    "- /dialogue start            Open a live dialogue session\n"
    "- /dialogue stop             End the current session\n"
    "- /dialogue status           Show current state and availability\n"
    "- /dialogue help             Show this help\n\n"
    "Notes:\n"
    "- MiniMax path needs MINIMAX_API_KEY env or "
    "~/.clawcodex/tts/minimax/credentials.json.\n"
    "- Feature flag: FEATURE_DIALOGUE_MODE=1 to enable (mirrors the "
    "FEATURE_VOICE_MODE gate).\n"
    "- Use /dialogue mode text for lower latency during MVP "
    "(audio modality is more expensive on the server)."
)


# ── helpers ──────────────────────────────────────────────────────────────


def _get_provider_name() -> str:
    """Read current ``settings.dialogue_provider`` without local import cycles."""
    try:
        from src.settings.settings import get_settings

        return (getattr(get_settings(), "dialogue_provider", "") or "").strip().lower()
    except Exception:
        return ""


def _is_enabled() -> bool:
    try:
        from src.settings.settings import get_settings

        return bool(getattr(get_settings(), "dialogue_enabled", False))
    except Exception:
        return False


def _get_modality() -> str:
    try:
        from src.settings.settings import get_settings

        raw = (getattr(get_settings(), "dialogue_modality", "") or "").strip().lower()
    except Exception:
        raw = ""
    return raw if raw in _VALID_MODALITIES else "text"


def _status_text() -> str:
    """Render the current state + per-layer availability diagnostics."""
    from clawcodex_ext.services.voice.voice_mode_enabled import (  # pylint: disable=no-name-in-module
        has_dialogue_auth,
        is_dialogue_available,
        is_dialogue_enabled,
        is_dialogue_feature_enabled,
        is_voice_disabled_by_kill_switch,
    )

    provider = _get_provider_name() or "minimax"
    modality = _get_modality()
    lines = [
        "Voice dialogue: " + ("on" if is_dialogue_enabled() else "off"),
        f"Provider: {provider}",
        f"Output modality: {modality}",
        "",
        "Availability:",
        f"  feature flag (FEATURE_DIALOGUE_MODE): {'on' if is_dialogue_feature_enabled() else 'off'}",
        f"  kill-switch (CLAWCODEX_VOICE_DISABLED): {'set' if is_voice_disabled_by_kill_switch() else 'clear'}",
        f"  overall available: {'yes' if is_dialogue_available() else 'no'}",
        f"  MiniMax credentials: {'present' if has_dialogue_auth() else 'missing'}",
        "",
    ]
    if not is_dialogue_feature_enabled():
        lines.append("Tip: set FEATURE_DIALOGUE_MODE=1 to enable dialogue.")
    if is_voice_disabled_by_kill_switch():
        lines.append("Tip: unset CLAWCODEX_VOICE_DISABLED to release the kill-switch.")
    if provider == "minimax" and not has_dialogue_auth():
        lines.append(
            "Tip: configure ~/.clawcodex/tts/minimax/credentials.json "
            "(or set MINIMAX_API_KEY) before starting a session."
        )
    lines.append("Use /voice for half-duplex push-to-talk STT instead.")
    return "\n".join(lines)


# ── main entry point ─────────────────────────────────────────────────────


def dialogue_command_call(args: str, context: CommandContext) -> LocalCommandResult:
    """``/dialogue`` handler — toggle / configure / start / stop the F-65 path."""
    raw = (args or "").strip()
    a = raw.lower()

    # 1. help (headless)
    if a in ("help", "-h", "--help"):
        return LocalCommandResult(type="text", value=_HELP)

    # 2. status (headless) — diagnostics for the F-65 gate layers.
    if a in ("status", "current", "show"):
        return LocalCommandResult(type="text", value=_status_text())

    # 3. off — disable dialogue mode, keep provider.
    if a == "off":
        set_dialogue_enabled(False)
        return LocalCommandResult(type="text", value="Voice dialogue disabled.")

    # 4. explicit provider — enable + select backend atomically.
    if a in DIALOGUE_PROVIDERS:
        set_dialogue_provider(a)
        set_dialogue_enabled(True)
        auth_hint = ""
        if a == "minimax":
            try:
                from clawcodex_ext.services.voice.voice_mode_enabled import (  # pylint: disable=no-name-in-module
                    has_dialogue_auth,
                )

                if not has_dialogue_auth():
                    auth_hint = (
                        " (warning: no MiniMax credentials — "
                        "configure ~/.clawcodex/tts/minimax/credentials.json before /dialogue start)"
                    )
            except Exception:
                logger.debug("Failed to check dialogue auth status", exc_info=True)
        return LocalCommandResult(
            type="text",
            value=f"Voice dialogue enabled with {a} backend{auth_hint}.",
        )

    # 5. modality — set the output modality for the *next* session.
    if a.startswith("mode "):
        value = raw[5:].strip().lower()
        if value not in _VALID_MODALITIES:
            return LocalCommandResult(
                type="text",
                value=(f"Unknown modality: {value!r}. Use 'text' or 'audio'."),
            )
        set_dialogue_modality(value)
        return LocalCommandResult(
            type="text",
            value=f"Dialogue output modality set to {value}.",
        )
    if a == "mode":
        return LocalCommandResult(
            type="text",
            value=(
                f"Current modality: {_get_modality()}. Use '/dialogue mode text' or '/dialogue mode audio' to change."
            ),
        )

    # 6. voice — set the TTS voice id for the next session.
    if a.startswith("voice "):
        name = raw[6:].strip()
        if not name:
            return LocalCommandResult(
                type="text",
                value="Missing voice name. Usage: /dialogue voice <name>",
            )
        set_dialogue_voice(name)
        return LocalCommandResult(
            type="text",
            value=f"Dialogue voice set to {name!r}.",
        )

    # 7. start — open a live session.
    if a == "start":
        return _start_session()

    # 8. stop — end any running session.
    if a == "stop":
        return _stop_session()

    # 9. no args — toggle the master switch (current provider kept).
    if not a:
        currently = _is_enabled()
        set_dialogue_enabled(not currently)
        provider = _get_provider_name() or "minimax"
        if not currently:
            return LocalCommandResult(
                type="text",
                value=(f"Voice dialogue enabled (provider: {provider}). Run /dialogue start to begin a session."),
            )
        return LocalCommandResult(type="text", value="Voice dialogue disabled.")

    # 10. unknown arg.
    return LocalCommandResult(
        type="text",
        value=(
            f"Unknown argument: {raw}. Valid options: "
            + ", ".join(
                list(DIALOGUE_PROVIDERS)
                + [
                    "off",
                    "mode",
                    "voice",
                    "start",
                    "stop",
                    "status",
                    "help",
                ]
            )
        ),
    )


# ── session lifecycle helpers ─────────────────────────────────────────────
#
# These are intentionally thin wrappers in the MVP. A full integration
# would hand off to a long-lived session manager (F-65 P65-B) that
# lives across multiple REPL turns; the command-path integration with
# the dialogue session manager is the next iteration's work. Here we
# validate that a session *can* be assembled (provider is registered,
# credentials are present, feature flag is on) and surface a clean
# message if not, so the user sees a deterministic outcome from the
# slash command rather than a stack trace.


def _start_session() -> LocalCommandResult:
    """Validate and announce a session start.

    A real session would route the dialogue session manager into the
    REPL's transport; that's out of scope for the MVP slice. The
    validation here is what the user-facing command really needs today:
    fail fast with a readable hint instead of letting WebSocket errors
    leak to the chat surface.
    """
    from clawcodex_ext.services.voice.voice_mode_enabled import (  # pylint: disable=redefined-outer-name,no-name-in-module
        DIALOGUE_PROVIDERS,
        has_dialogue_auth,
        is_dialogue_available,
        is_dialogue_feature_enabled,
        is_voice_disabled_by_kill_switch,
    )

    if not is_dialogue_feature_enabled():
        return LocalCommandResult(
            type="text",
            value=("Dialogue mode is disabled by feature flag. Set FEATURE_DIALOGUE_MODE=1 to enable it."),
        )
    if is_voice_disabled_by_kill_switch():
        return LocalCommandResult(
            type="text",
            value=(
                "Dialogue is blocked by the voice kill-switch (CLAWCODEX_VOICE_DISABLED). Unset it to allow sessions."
            ),
        )
    if not is_dialogue_available():
        return LocalCommandResult(
            type="text",
            value="Dialogue is currently unavailable (see /dialogue status).",
        )
    provider = _get_provider_name() or "minimax"
    if provider not in DIALOGUE_PROVIDERS:
        return LocalCommandResult(
            type="text",
            value=(f"Unknown dialogue provider {provider!r}. Use /dialogue minimax to select one."),
        )
    # Probe the registry so we surface "provider not registered" before
    # the user starts talking into the microphone.
    try:
        from clawcodex_ext.services.voice.provider_registry import (  # pylint: disable=no-name-in-module
            DIALOGUE_REGISTRY,
        )

        if provider not in DIALOGUE_REGISTRY:
            return LocalCommandResult(
                type="text",
                value=(
                    f"Provider {provider!r} is not registered. This is a build/installation issue, not a config one."
                ),
            )
    except Exception as exc:  # pragma: no cover — defensive only
        logger.warning("Dialogue registry probe failed: %s", exc)
    if provider == "minimax" and not has_dialogue_auth():
        return LocalCommandResult(
            type="text",
            value=(
                "No MiniMax credentials configured. Set MINIMAX_API_KEY "
                "or write ~/.clawcodex/tts/minimax/credentials.json."
            ),
        )
    return LocalCommandResult(
        type="text",
        value=(
            f"Dialogue session ready (provider: {provider}, "
            f"modality: {_get_modality()}). "
            "A live session needs the dialogue session manager "
            "to be wired into the REPL transport — pending P65-B "
            "integration. Use /dialogue stop when done."
        ),
    )


def _stop_session() -> LocalCommandResult:
    """End any running session.

    The MVP slice has no long-lived session state in the command module
    (that's the session manager's job); the hook is here so ``/dialogue
    stop`` always succeeds even if the user runs it before /start.
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = None
    if loop is not None and loop.is_running():
        # Best-effort drain of any pending dialogue tasks. The session
        # manager's ``close`` method is idempotent (P65-B), so calling
        # it again on no-op is safe. The MVP doesn't keep a handle to
        # the manager — that integration lives in the REPL transport.
        pass
    return LocalCommandResult(type="text", value="Voice dialogue session ended.")


# ── command object ───────────────────────────────────────────────────────


DIALOGUE_COMMAND = LocalCommand(
    name="dialogue",
    description="Toggle full-duplex voice dialogue mode and select backend",
    argument_hint="[minimax|off|mode <text|audio>|voice <name>|start|stop|status|help]",
    supports_non_interactive=True,
)
DIALOGUE_COMMAND.set_call(dialogue_command_call)

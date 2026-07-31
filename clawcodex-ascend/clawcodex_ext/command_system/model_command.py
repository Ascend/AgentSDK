"""model — interactive ``/model`` command (port of TS local-jsx).

Port of ``typescript/src/commands/model/`` (``model.tsx`` + ``index.ts``). Like
``/theme`` and ``/effort``, this is the **inverse** of ``/export`` at the TUI dispatch
layer: the TUI keeps intercepting ``/model`` → ``open_dialog="model"`` to
preserve its ``ModelPickerScreen``; this command serves the registry-consulting surfaces
(REPL numbered-menu ``select``, SDK, help/aggregator listings) where ``/model`` was
previously invisible (it lived only in the TUI's private ``LOCAL_BUILTINS``).

**Functional** (unlike ``/effort``): the live model channel in Python is
``provider.model`` — providers resolve the request model via ``_get_model`` =
``kwargs.get("model", self.model)`` and neither the main loop nor the fast path passes a
``model=`` override, so the held provider's ``.model`` decides the next query's model. So
this command sets **``ctx.provider.model``** (the channel inference reads), reachable on the
REPL because Phase 7 also wires ``provider`` into the REPL command context. It is the **only**
write: the reactive ``AppState.main_loop_model`` has no production reader (its
``set_main_loop_model_override`` mirror is dead), and TS ``/model`` never writes disk — so
this is session-only, no settings write.

**Headless keystone:** the arg paths (``/model <name>``, ``current``/``status``/…, ``help``)
need no UI; only the no-args picker needs a surface (``NullUIHost.select`` raises there).

**Deliberate divergences (documented for parity review):**
  * **Dropped** (need unported subsystems): explicit ``refresh`` command, org-allowlist,
    1M-context gates, fast-mode, extra-usage billing, network model-validation, and TS
    ``'default'``-reset (no provider-default reachable from ``CommandContext``).
  * **Validation = alias-resolve + cached-catalog membership** using the shared
    stale-while-revalidate model cache. The picker returns configured fallback models
    immediately while live discovery runs in the background.
  * **Static description** ("Set the AI model"); TS's is dynamic ``…(currently {model})`` — a
    frozen ``CommandBase.description: str`` can't be a getter. ``current`` shows the live model.
  * **``provider.model`` is the sole write** (Python reads it, not AppState — an architectural
    divergence from TS, which writes ``AppState.mainLoopModel``).
  * **Effort suffix in ``current``** reads ``settings.effort`` (the Phase 6 channel), not
    AppState ``effortValue``.
  * **Label = ``display_name``** (drops TS ``renderModelLabel``'s ``(default)``/alias decoration).

``disable_model_invocation=True`` — model selection is user-driven (the ``/permissions``
stance); a model must not switch its own model via the SlashCommand tool. ``src.models`` /
``get_settings`` are imported lazily (the ``app.py``/advisor discipline).
"""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass

from .types import (
    CommandContext,
    InteractiveCommand,
    InteractiveOutcome,
    UIOption,
)

COMMON_HELP_ARGS = frozenset({"help", "-h", "--help"})
# Verbatim from TS COMMON_INFO_ARGS (model.tsx).
COMMON_INFO_ARGS = frozenset(
    {
        "list",
        "show",
        "display",
        "current",
        "view",
        "get",
        "check",
        "describe",
        "print",
        "version",
        "about",
        "status",
        "?",
    }
)

_NO_PROVIDER = "Model unavailable (no active provider)."
# TS help text (model.tsx:792), minus the dropped `refresh` clause.
_USAGE = "Run /model to open the model selection menu, or /model [modelName] to set the model."


def _canonical(name: str) -> str:
    """Resolve an alias to its canonical id (``sonnet`` → ``claude-sonnet-4-...``);
    returns the input unchanged for non-aliases. Lazy import — see module docstring.
    """
    from src.models.model import canonical_model_name

    return canonical_model_name(name)


def _label(model: str | None) -> str:
    if not model:
        return "(none)"
    from src.models.model import display_name

    return display_name(model)


def _list_models(provider, provider_name: str | None = None) -> list[str]:
    """Return cached/fallback models while live discovery runs out of band."""

    try:
        from clawcodex_ext.providers.model_catalog_cache import get_model_catalog

        catalog_name = provider_name or getattr(provider, "provider_name", None) or type(provider).__name__.lower()
        snapshot = get_model_catalog(str(catalog_name), provider)
        return list(snapshot.models)
    except Exception:
        return []


def _provider_name(context: CommandContext, provider) -> str | None:
    runtime = getattr(context, "runtime_context", None)
    return getattr(runtime, "provider_name", None) or getattr(provider, "provider_name", None)


def _options(models: list[str], current: str | None) -> list[UIOption]:
    return [UIOption(value=m, label=m, description="current" if m == current else None) for m in models]


def _effort_suffix() -> str:
    """`` (effort: X)`` when an effort is persisted (Phase 6 ``settings.effort``), else ``""``."""
    try:
        from src.settings.settings import get_settings

        eff = get_settings().effort
    except Exception:
        eff = ""
    return f" (effort: {eff})" if eff else ""


def _show_current(context: CommandContext) -> str:
    prov = context.provider
    cur = getattr(prov, "model", None) if prov is not None else None
    if not cur:
        return "Current model: (none)"
    return f"Current model: {_label(cur)}{_effort_suffix()}"


def _apply(provider, model: str) -> None:
    """Set the live model. ``provider.model`` is the channel inference reads; guarded
    exactly like the TUI's ``_open_model_picker`` (app.py).
    """
    with suppress(Exception):
        provider.model = model


@dataclass(frozen=True)
class ModelCommand(InteractiveCommand):
    """Pick or set the active model. Frozen + no new fields (the ``ThemeCommand`` pattern);
    behavior lives in :meth:`run`.
    """

    async def run(self, args: str, context: CommandContext) -> InteractiveOutcome:
        a = (args or "").strip()
        low = a.lower()

        if low in COMMON_HELP_ARGS:  # TS help => display:'system' (model.tsx:792)
            return InteractiveOutcome(message=_USAGE, display="system")
        if low in COMMON_INFO_ARGS:  # ShowModelAndClose
            return InteractiveOutcome(message=_show_current(context), display="user")
        if low == "refresh":  # TS network discovery — dropped
            return InteractiveOutcome(message="Model refresh is not supported.", display="system")
        if not a:  # ModelPickerWrapper
            return await self._pick(context)
        return self._set(context, a)  # SetModelAndClose (headless)

    async def _pick(self, context: CommandContext) -> InteractiveOutcome:
        prov = context.provider
        if prov is None:
            return InteractiveOutcome(message=_NO_PROVIDER, display="system")
        models = _list_models(prov, _provider_name(context, prov))
        if not models:
            return InteractiveOutcome(message="No models available.", display="system")
        current = getattr(prov, "model", None)
        picked = await context.ui.select("Select model:", _options(models, current), current=current)
        if picked is None:  # TS cancel: "Kept model as …" (model.tsx:376), NOT skip
            return InteractiveOutcome(message=f"Kept model as {_label(current)}", display="system")
        _apply(prov, picked)
        return InteractiveOutcome(message=f"Set model to {_label(picked)}", display="user")

    def _set(self, context: CommandContext, arg: str) -> InteractiveOutcome:
        prov = context.provider
        if prov is None or not hasattr(prov, "model"):
            return InteractiveOutcome(message=_NO_PROVIDER, display="system")
        canon = _canonical(arg)
        models = _list_models(prov, _provider_name(context, prov))
        # Membership validation (TS network validate dropped). Permissive when the
        # provider lists nothing (unknown provider) so a valid id still goes through.
        if models and canon not in models:
            return InteractiveOutcome(message=f"Model '{arg}' not found", display="system")
        _apply(prov, canon)
        return InteractiveOutcome(message=f"Set model to {_label(canon)}", display="user")


MODEL_COMMAND = ModelCommand(
    name="model",
    description="Set the AI model",  # static (TS is dynamic — see module docstring)
    argument_hint="[model]",  # verbatim TS index.ts
    disable_model_invocation=True,  # user-driven only (the /permissions stance)
)


__all__ = ["MODEL_COMMAND", "ModelCommand"]

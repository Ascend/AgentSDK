"""F-97 LODESTONE — data models.

Frozen dataclasses describing the core types used by the parser, target
registry, resolver and renderer. Designed to be import-side-effect-free
(``from clawcodex_ext.services.lodestone.models import ...``) so callers
can adopt the types without paying for parser / resolver construction.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

AnchorKind = Literal[
    "file_path",  # path:line[:col][-end_line[:end_col]]
    "function_ref",  # module.func or module::func
    "git_blob",  # @<git_sha>:path  (e.g. ``@abc1234:src/foo.py``)
    "git_commit",  # 7-40 hex sha
    "tracker_issue",  # #123, ORG-456, [ORG-789]
    "url",  # already a url
]

Sink = Literal["text", "markdown", "osc8", "auto"]

# Anchor template placeholders that built-in / user targets are allowed
# to use in ``template``.  Anything outside this whitelist is rejected at
# register-time (see ``AnchorTargetRegistry.register``).
_ALLOWED_PLACEHOLDERS = frozenset(
    {
        "abs",
        "rel",
        "line",
        "col",
        "end_line",
        "end_col",
        "remote",
        "branch",
        "ref",
        "owner",
        "repo",
        "key",
        "host",
        "workspace",
    }
)


@dataclass(frozen=True)
class LodestoneAnchor:
    """A single anchor detected inside a piece of text.

    ``raw`` is exactly the substring that matched; ``span`` is the
    ``(start, end)`` character offset into the original input (or ``None``
    when constructed programmatically).
    """

    kind: AnchorKind
    raw: str
    file_path: str | None = None
    line: int | None = None
    column: int | None = None
    end_line: int | None = None
    end_column: int | None = None
    symbol: str | None = None
    git_sha: str | None = None
    # tracker_key[0] is the issue tracker host (``gitcode`` / ``linear`` …);
    # tracker_key[1] is the issue number / key string (``"123"`` / ``"LIN-456"``).
    tracker_key: tuple[str, str | None] | None = None
    url: str | None = None
    span: tuple[int, int] | None = None

    def link_text(self) -> str:
        """Stable human text that represents this anchor.

        ``renderer`` uses this as Markdown link text / OSC 8 display text
        when the caller does not supply an explicit rewrite.
        """
        if self.kind == "tracker_issue" and self.tracker_key is not None:
            prefix, key = self.tracker_key
            return f"#{key}" if prefix == "gitcode" else f"{prefix}-{key}" if prefix else f"#{key}"
        if self.kind == "git_commit" and self.git_sha:
            return self.git_sha[:7]
        if self.kind == "file_path" and self.file_path:
            base = self.file_path
            if self.end_line is not None:
                if self.end_column is not None:
                    return f"{base}:{self.line}-{self.end_line}:{self.end_column}"
                return f"{base}:{self.line}-{self.end_line}"
            if self.column is not None and self.line is not None:
                return f"{base}:{self.line}:{self.column}"
            if self.line is not None:
                return f"{base}:{self.line}"
            return base
        if self.kind == "function_ref" and self.symbol:
            return self.symbol
        return self.raw


@dataclass(frozen=True)
class AnchorContext:
    """Inputs the resolver / renderer consult.

    ``workspace_root`` is used for ``path`` resolution (absolute path),
    remote URL detection and the path-traversal guard.  ``session_id``
    is informational.  ``config`` carries the user-supplied preferences.
    ``env`` may be passed explicitly (tests do this) — by default the
    resolver reads ``os.environ``.
    """

    workspace_root: Path | None
    session_id: str | None
    config: LodestoneConfig
    remote_url: str | None = None
    branch: str | None = None
    is_collapsed: bool = False
    # Allow injection for testing; default reads os.environ on access.
    env: dict[str, str] | None = None

    def get_env(self, key: str, default: str | None = None) -> str | None:
        if self.env is not None:
            return self.env.get(key, default)
        import os

        return os.environ.get(key, default)


@dataclass(frozen=True)
class AnchorTarget:
    """Registration record for one editor / tracker / remote target.

    ``template`` is a string with ``{placeholder}`` substitutions; only
    the keys in ``_ALLOWED_PLACEHOLDERS`` are honoured.  ``requires`` is a
    tuple of names — environment variables that must be set before the
    resolver may pick this target (e.g. ``("DISPLAY",)`` for ``idea://``).
    """

    kind: AnchorKind  # which AnchorKind the target can render
    target_id: str
    template: str
    is_remote: bool = False
    requires: tuple[str, ...] = ()
    description: str = ""
    # Optional host whitelist (lower-cased, no scheme).  When non-empty
    # the resolver only uses this target for git-tracker issues that
    # belong to one of these hosts.
    hosts: tuple[str, ...] = ()


@dataclass(frozen=True)
class RenderedAnchor:
    """Result of resolving and rendering one ``LodestoneAnchor``."""

    anchor: LodestoneAnchor
    target: AnchorTarget | None
    link_text: str
    rendered: str
    is_anchor: bool
    fallback_reason: str | None = None
    sink: Sink = "text"


@dataclass(frozen=True)
class WorkspaceFingerprint:
    """Stable description of a workspace discovered at startup."""

    workspace_root: Path
    primary_remote_url: str | None
    primary_remote_host: str | None  # lower-cased ("gitcode.com")
    default_branch: str | None
    tracked_branches: tuple[str, ...] = ()
    has_git: bool = False
    trackers: tuple[str, ...] = ()  # recognised tracker adapters


@dataclass(frozen=True)
class LodestoneConfig:
    """User-facing configuration. Persisted to ``~/.clawcodex/lodestone.json``.

    ``enabled`` controls the entire feature.  When ``False`` the parser
    still runs but the renderer emits raw text (the disabling is a
    fail-closed guard against accidental link injection in chat logs).
    """

    enabled: bool = True
    default_editor: str = "vscode"
    fallback_editor: str = "file"
    auto_remote: bool = True
    disabled_kinds: tuple[AnchorKind, ...] = ()
    renderer: Sink = "auto"
    custom_targets: tuple[AnchorTarget, ...] = ()
    # Tracker-friendly workspace defaults — allow ``/link open #1`` to
    # route to GitCode without per-call host argument.
    # NOTE: stored as the *bare* domain (``gitcode.com`` etc.) so the
    # template ``{host}`` placeholder is direct.
    default_tracker_host: str = "gitcode.com"
    default_tracker_repo: tuple[str, str] | None = None  # (owner, repo)
    extra_hosts: tuple[str, ...] = ()  # additional trusted remote hosts
    custom_placeholder_resolvers: tuple[str, ...] = ()  # unused: reserved


# ---------------------------------------------------------------------------
# Target registry — minimal in-memory implementation, augmented by
# ``targets.build_default_registry`` with the built-in editor / remote /
# tracker targets.
# ---------------------------------------------------------------------------


_PLACEHOLDER_RE = re.compile(r"\{([a-z_][a-z0-9_]*)\}")


class AnchorTargetRegistry:
    """Holds registered ``AnchorTarget`` records and offers
    ``candidates(kind, ctx)`` / ``pick(kind, ctx)``.

    Thread-safe enough for the resolver's typical one-shot usage; not
    designed for hot-path concurrent writes.

    The same ``target_id`` can be registered for multiple ``kind`` values
    (e.g. ``github`` is a single editor identity used for ``file_path``,
    ``git_blob`` and ``git_commit``).  Internally the registry keys by
    ``(target_id, kind)``.
    """

    def __init__(self, config: LodestoneConfig | None = None) -> None:
        self._config = config or LodestoneConfig()
        self._targets: dict[tuple[str, AnchorKind], AnchorTarget] = {}
        self._by_id: dict[str, AnchorTarget] = {}

    # -- mutation -------------------------------------------------------------

    def register(self, target: AnchorTarget, *, overwrite: bool = False) -> None:
        if not target.target_id:
            raise ValueError("AnchorTarget.target_id must be non-empty")
        # Validate template — only allowed placeholders.
        for placeholder in _PLACEHOLDER_RE.findall(target.template):
            if placeholder not in _ALLOWED_PLACEHOLDERS:
                raise ValueError(
                    f"template contains disallowed placeholder {{{placeholder}}}; "
                    f"allowed: {sorted(_ALLOWED_PLACEHOLDERS)}"
                )
        # Same id+kind already registered → dedupe unless overwrite requested.
        key = (target.target_id, target.kind)
        existing = self._targets.get(key)
        if existing is not None and not overwrite:
            raise ValueError(
                f"target_id {target.target_id!r} (kind={target.kind}) already registered; pass overwrite=True"
            )
        self._targets[key] = target
        if target.target_id not in self._by_id or overwrite:
            self._by_id[target.target_id] = target

    def unregister(self, target_id: str, *, kind: AnchorKind | None = None) -> bool:
        if kind is not None:
            existed = self._targets.pop((target_id, kind), None) is not None
            return existed
        # kind=None → drop every record that carries this id
        keys = [k for k in self._targets if k[0] == target_id]
        for k in keys:
            del self._targets[k]
        removed = bool(keys)
        if removed:
            self._by_id.pop(target_id, None)
        return removed

    def clear(self) -> None:
        self._targets.clear()
        self._by_id.clear()

    # -- read -----------------------------------------------------------------

    def list(self) -> list[AnchorTarget]:
        return sorted(self._targets.values(), key=lambda t: (t.target_id, t.kind))

    def get(self, target_id: str, kind: AnchorKind | None = None) -> AnchorTarget | None:
        if kind is not None:
            return self._targets.get((target_id, kind))
        return self._by_id.get(target_id)

    def update_config(self, config: LodestoneConfig) -> None:
        self._config = config

    # -- resolution helpers ---------------------------------------------------

    def candidates(self, kind: AnchorKind, *, ctx: AnchorContext | None = None) -> list[AnchorTarget]:
        ctx = ctx or AnchorContext(
            workspace_root=None,
            session_id=None,
            config=self._config,
        )
        out: list[AnchorTarget] = []
        for t in self._targets.values():
            if t.kind != kind:
                continue
            if not _target_is_available(t, ctx):
                continue
            out.append(t)
        return out

    def pick(self, kind: AnchorKind, *, ctx: AnchorContext | None = None) -> AnchorTarget | None:
        candidates = self.candidates(kind, ctx=ctx)
        if not candidates:
            return None
        # Prefer the user-configured ``default_editor`` for file targets,
        # otherwise fall back to is_remote=False targets first.
        cfg = ctx.config if ctx is not None else self._config
        preferred = _pick_preferred(candidates, cfg)
        return preferred or candidates[0]


def _target_is_available(target: AnchorTarget, ctx: AnchorContext) -> bool:
    """Return True if every required env var is set in the context."""
    for var in target.requires:
        if ctx.get_env(var) is None:
            return False
    return True


def _pick_preferred(candidates: list[AnchorTarget], cfg: LodestoneConfig) -> AnchorTarget | None:
    """Apply user-configured preference over the candidate list.

    Order:
    1. ``default_editor`` matches target_id
    2. first non-remote candidate (local editor/file)
    3. first remote candidate
    4. None
    """
    for c in candidates:
        if c.target_id == cfg.default_editor:
            return c
    for c in candidates:
        if not c.is_remote:
            return c
    for c in candidates:
        if c.is_remote:
            return c
    return None


def extract_placeholders(template: str) -> list[str]:
    """Public helper for tests / docs — returns sorted unique placeholders."""
    return sorted(set(_PLACEHOLDER_RE.findall(template)))


def allowed_placeholders() -> frozenset[str]:
    """Expose the canonical placeholder whitelist."""
    return _ALLOWED_PLACEHOLDERS


__all__ = [
    "AnchorContext",
    "AnchorKind",
    "AnchorTarget",
    "AnchorTargetRegistry",
    "LodestoneAnchor",
    "LodestoneConfig",
    "RenderedAnchor",
    "Sink",
    "WorkspaceFingerprint",
    "allowed_placeholders",
    "extract_placeholders",
]


# Late import guard so type-only modules don't trigger cycles.
def _resolve_forward_refs() -> None:
    # LodestoneConfig is forward-declared in AnchorContext — no-op stub for IDEs.
    return None


_resolve_forward_refs()

# silence unused-import linting for ``field`` / ``Iterable`` / ``Any`` —
# these stay imported so downstream ``from .models import field`` keeps working.
_ = (field, Iterable, Any)

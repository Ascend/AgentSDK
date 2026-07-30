"""F-97 LODESTONE — high-level service facade.

Owns one :class:`AnchorTargetRegistry` and one :class:`AnchorResolver`
in process.  Two reasons we expose the facade rather than calling
:func:`AnchorResolver().resolve(...)` directly:

*   ``config`` is loaded once and shared; CLI ``/link config …`` mutates
    it through :meth:`LodestoneService.update_config`.
*   A built-in target registry (vscode / gitcode / …) is built once at
    construction, so the registry is hot-path-alloc-free.

A module-level :func:`get_lodestone_service` exposes the default
singleton.  Tests should pass ``config=`` and ``registry=`` explicitly
to :class:`LodestoneService` to keep state local.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

from .config import default_config, load_config
from .models import (
    AnchorContext,
    AnchorTargetRegistry,
    LodestoneAnchor,
    LodestoneConfig,
    RenderedAnchor,
    Sink,
)
from .renderer import AnchorRenderer
from .resolver import AnchorResolver
from .targets import build_default_registry


class LodestoneService:
    """Stateless facade — constructed once per process and re-used."""

    def __init__(
        self,
        *,
        config: LodestoneConfig | None = None,
        registry: AnchorTargetRegistry | None = None,
        resolver: AnchorResolver | None = None,
        renderer: AnchorRenderer | None = None,
    ) -> None:
        self._config = config or default_config()
        self._registry = registry or build_default_registry(self._config)
        self._renderer = renderer or AnchorRenderer()
        self._resolver = resolver or AnchorResolver(self._registry, self._renderer)
        self._lock = threading.Lock()

    # -- configuration --------------------------------------------------------

    @property
    def config(self) -> LodestoneConfig:
        return self._config

    @property
    def registry(self) -> AnchorTargetRegistry:
        return self._registry

    @property
    def resolver(self) -> AnchorResolver:
        return self._resolver

    @property
    def renderer(self) -> AnchorRenderer:
        return self._renderer

    def reload_config(self, path: Path | None = None) -> LodestoneConfig:
        """Re-read persisted config; the registry tracks the new config."""
        self._config = load_config(path)
        with self._lock:
            self._registry.update_config(self._config)
        return self._config

    def update_config(self, **changes: object) -> LodestoneConfig:
        """Return a new :class:`LodestoneConfig` with ``changes`` applied."""
        merged = {**self._config.__dict__, **changes}
        self._config = LodestoneConfig(**{k: v for k, v in merged.items() if k in LodestoneConfig.__dataclass_fields__})  # pylint: disable=no-member
        with self._lock:
            self._registry.update_config(self._config)
        return self._config

    # -- resolution entry points ---------------------------------------------

    def resolve_text(
        self,
        text: str,
        *,
        workspace_root: Path | None = None,
        session_id: str | None = None,
        sink: Sink | None = None,
        target_override: str | None = None,
        env: dict[str, str] | None = None,
    ) -> str:
        ctx = _build_context(
            self._config,
            workspace_root=workspace_root,
            session_id=session_id,
            env=env,
        )
        if not ctx.config.enabled:
            return text or ""
        sink = sink or self._config.renderer
        anchors = self._parser().parse(text or "")
        if not anchors:
            return text or ""
        sorted_anchors = sorted(anchors, key=lambda a: a.span[0] if a.span else 0, reverse=True)
        out = text or ""
        for anchor in sorted_anchors:
            if anchor.span is None:
                continue
            rendered = self._resolver.resolve(anchor, ctx=ctx, sink=sink, target_override=target_override)
            out = out[: anchor.span[0]] + rendered.rendered + out[anchor.span[1] :]
        return out

    def resolve_one(
        self,
        anchor: LodestoneAnchor,
        *,
        workspace_root: Path | None = None,
        session_id: str | None = None,
        sink: Sink | None = None,
        target_override: str | None = None,
        env: dict[str, str] | None = None,
    ) -> RenderedAnchor:
        ctx = _build_context(
            self._config,
            workspace_root=workspace_root,
            session_id=session_id,
            env=env,
        )
        return self._resolver.resolve(
            anchor,
            ctx=ctx,
            sink=sink or self._config.renderer,
            target_override=target_override,
        )

    def is_enabled(self) -> bool:
        return bool(self._config.enabled)

    # -- internal ------------------------------------------------------------

    def _parser(self):
        # Lazy-import to break load-cycle (parser ↔ models); no global state.
        from .parser import AnchorParser

        return AnchorParser()


def _build_context(
    cfg: LodestoneConfig,
    *,
    workspace_root: Path | None,
    session_id: str | None,
    env: dict[str, str] | None,
    branch: str | None = None,
    remote_url: str | None = None,
) -> AnchorContext:
    if workspace_root is not None:
        from .fingerprint import build_anchor_context, detect_workspace_fingerprint

        fp = detect_workspace_fingerprint(workspace_root, use_cache=False)
        ctx = build_anchor_context(
            workspace_root,
            cfg,
            session_id=session_id,
            fingerprint=fp,
            branch=branch,
            env=env,
        )
        return ctx
    return AnchorContext(
        workspace_root=None,
        session_id=session_id,
        config=cfg,
        branch=branch,
        remote_url=remote_url,
        env=env,
    )


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------


_default_singleton: LodestoneService | None = None
_singleton_lock = threading.Lock()


def get_lodestone_service() -> LodestoneService:
    """Return the process-wide default :class:`LodestoneService`."""
    global _default_singleton
    if _default_singleton is None:
        with _singleton_lock:
            if _default_singleton is None:
                cfg = load_config()
                _default_singleton = LodestoneService(config=cfg)
    return _default_singleton


def reset_default_service() -> None:
    """Drop the cached singleton — tests rely on this for isolation."""
    global _default_singleton
    _default_singleton = None


__all__ = [
    "LodestoneService",
    "get_lodestone_service",
    "reset_default_service",
]

# Silence unused-import lint for ``os`` callers.
_ = os

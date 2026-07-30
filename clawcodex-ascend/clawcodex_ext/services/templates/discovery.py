"""Default discovery paths for templates (F-85 / P85-B).

This module is the **path** half of P85-B: it computes where to look
for template files on disk. It is intentionally pure — no I/O, no
registry mutation — so tests can exercise the resolution logic
without touching the filesystem.

Three discovery sources are supported, mirroring the agent discovery
convention in :mod:`src.utils.markdown_config_loader`:

1. **User dir** — ``$CLAWCODEX_CONFIG_DIR/templates/`` or
   ``~/.clawcodex/templates/``.
2. **Project dirs** — ``<cwd>/.clawcodex/templates/`` walked upward,
   stopping at the nearest ``.git`` ancestor (or the filesystem
   root when no ``.git`` is present). The closest ancestor wins
   (innermost), so the caller iterates the returned list with
   ``overwrite=True`` and the innermost template wins.
3. **Managed dir** — ``$CLAWCODEX_MANAGED_CONFIG_DIR/templates/`` or
   ``/etc/clawcodex/templates/``.

The function names deliberately use ``CLAWCODEX_*`` env vars
(separate from upstream ``CLAUDE_*``) so clawcodex-specific config
stays isolated from ``~/.claude`` conventions. The defaults use
``~/.clawcodex`` (matches :data:`src.config.GLOBAL_CONFIG_DIR`).

All three resolvers return :class:`Path` objects even when the
directory does not yet exist on disk — callers MUST check
``Path.is_dir()`` before reading. This keeps the helpers cheap and
side-effect-free, and matches the behaviour of
:func:`pathlib.Path.home` which never raises on a missing home
directory.
"""

from __future__ import annotations

import os
from pathlib import Path

# Env-var names — exported so tests and bootstrap.py can refer to the
# same constants without re-stringifying them.
CLAWCODEX_CONFIG_DIR_ENV = "CLAWCODEX_CONFIG_DIR"
CLAWCODEX_MANAGED_CONFIG_DIR_ENV = "CLAWCODEX_MANAGED_CONFIG_DIR"

# Subdirectory name under each base that holds template files.
TEMPLATES_SUBDIR = "templates"

# Project config dir name; matches .claude convention but is
# clawcodex-specific to avoid mixing with upstream tooling.
PROJECT_CONFIG_DIR = ".clawcodex"


def _resolve_from_env(env_var: str) -> Path | None:
    """Return the env-var override as a resolved :class:`Path`, or ``None``.

    Empty / whitespace-only env values are treated as absent (matches
    :func:`os.environ.get` semantics but tolerates accidental ``""``
    exports from wrapper scripts).
    """
    raw = os.environ.get(env_var)
    if not raw or not raw.strip():
        return None
    return Path(raw).expanduser().resolve()


def get_user_templates_dir() -> Path:
    """Return the user-level templates directory (resolved, may not exist).

    Resolution order:

    1. ``$CLAWCODEX_CONFIG_DIR/templates/`` if the env var is set.
    2. ``~/.clawcodex/templates/`` otherwise.

    The returned path is resolved (symlinks followed, ``~`` expanded)
    so downstream equality checks against other resolved paths behave
    predictably. The path is returned even when the directory does
    not exist; callers MUST check :meth:`Path.is_dir` before reading.
    """
    override = _resolve_from_env(CLAWCODEX_CONFIG_DIR_ENV)
    if override is not None:
        return override / TEMPLATES_SUBDIR
    return Path.home() / ".clawcodex" / TEMPLATES_SUBDIR


def get_managed_templates_dir() -> Path:
    """Return the managed templates directory (resolved, may not exist).

    Resolution order:

    1. ``$CLAWCODEX_MANAGED_CONFIG_DIR/templates/`` if the env var is set.
    2. ``/etc/clawcodex/templates/`` otherwise.

    The function is pure — it never raises on a missing directory.
    Callers MUST check :meth:`Path.is_dir` before reading.
    """
    override = _resolve_from_env(CLAWCODEX_MANAGED_CONFIG_DIR_ENV)
    if override is not None:
        return override / TEMPLATES_SUBDIR
    return Path("/etc/clawcodex") / TEMPLATES_SUBDIR


def _find_git_root(start: Path) -> Path | None:
    """Walk upward from ``start`` until a ``.git`` directory is found.

    Returns ``None`` when the filesystem root is reached without
    finding one. ``start`` itself is checked first, so a ``.git`` in
    the cwd stops the walk.
    """
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def get_project_templates_dirs(cwd: Path | None = None) -> list[Path]:
    """Return project-level templates directories from cwd outward.

    Walks ``cwd`` upward until it hits a ``.git`` directory (or the
    filesystem root). Each ancestor's ``<ancestor>/.clawcodex/templates/``
    directory is yielded **only if it actually exists on disk**.

    The returned list is ordered **outermost-first**, so the caller
    iterates with ``overwrite=True`` and the innermost (most specific)
    template wins.

    Args:
        cwd: Project root to walk up from. ``None`` → :func:`os.getcwd`.
            Tests pass a tmp dir to keep the walker isolated from the
            real working directory.

    Returns:
        A list of :class:`Path` objects pointing at existing project
        templates directories, ordered from the outermost ancestor
        to the innermost. Empty when nothing matches.
    """
    base = Path(cwd) if cwd is not None else Path(os.getcwd())
    if not base.exists() or not base.is_dir():
        return []
    git_root = _find_git_root(base)
    out: list[Path] = []
    # Include cwd itself first; Path.parents excludes it. The walker
    # must check cwd's own .clawcodex/templates/ before climbing.
    chain = [base, *base.parents]
    if git_root is not None:
        try:
            idx = chain.index(git_root)
        except ValueError:
            idx = len(chain) - 1
        # Trim everything past the git root so a template outside the
        # repo cannot leak in. cwd stays in the chain because we
        # always include it as chain[0].
        chain = chain[: idx + 1]
    # outermost-first: chain is innermost-first, so reverse it.
    for ancestor in reversed(chain):
        candidate = ancestor / PROJECT_CONFIG_DIR / TEMPLATES_SUBDIR
        if candidate.is_dir():
            out.append(candidate)
    return out


__all__ = [
    "CLAWCODEX_CONFIG_DIR_ENV",
    "CLAWCODEX_MANAGED_CONFIG_DIR_ENV",
    "PROJECT_CONFIG_DIR",
    "TEMPLATES_SUBDIR",
    "get_managed_templates_dir",
    "get_project_templates_dirs",
    "get_user_templates_dir",
]

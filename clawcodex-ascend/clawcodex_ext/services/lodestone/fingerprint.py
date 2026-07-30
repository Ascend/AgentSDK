"""F-97 LODESTONE — workspace fingerprint.

Wraps the upstream ``clawcodex_ext.utils.git`` primitives to produce a
``WorkspaceFingerprint`` that the resolver consumes.

Detection ladder:

1.  ``git rev-parse --show-toplevel`` — confirm we are inside a repo.
2.  ``git remote -v`` — first non-empty fetch URL becomes the primary
    remote.  ``gitcode.com`` / ``github.com`` / ``gitee.com`` / ``linear.app``
    are matched case-insensitively.  SSH (``git@host:owner/repo.git``)
    and HTTPS (``https://host/owner/repo.git``) flavours are both
    normalised to ``host`` / ``owner`` / ``repo`` triples.
3.  ``git symbolic-ref refs/remotes/origin/HEAD`` → default branch, with
    fallback to ``main`` and ``master``.
4.  Trackers — for now we sniff for known config files:

    *   ``.clawcodex/orchestrator/gitcode.yaml``
    *   ``.clawcodex/orchestrator/linear.yaml``

    The orchestrator's adapter files (see ``extensions/orchestrator``)
    are the source of truth — once they exist at the workspace root we
    record their host in ``WorkspaceFingerprint.trackers``.

5.  Cache — the fingerprint is cached at
    ``<workspace_root>/.clawcodex/lodestone.json`` for 24 h; when the
    file is missing or stale we re-run the detection.

The implementation deliberately never depends on the orchestrator runtime;
if the adapter modules are importable we read them, otherwise we just
record the workspace path and ``has_git=True``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import asdict
from pathlib import Path

from .models import AnchorContext, WorkspaceFingerprint
from .targets import GITCODE_HOST as _GITCODE
from .targets import GITEE_HOST, GITHUB_HOST, LINEAR_HOST

log = logging.getLogger(__name__)


_CACHE_FILENAME = "lodestone.json"
_CACHE_TTL_SECONDS = 24 * 3600


_REMOTE_RE = re.compile(
    r"^(?P<proto>https?://|ssh://|git://|git\+ssh://)?"
    r"(?:[^@/]+@)?"  # optional user@
    r"(?P<host>[A-Za-z0-9._\-]+)"
    r"[:/](?P<owner>[^/\s]+)/(?P<repo>[^/\s]+?)(?:\.git)?/?$"
)


# ---------------------------------------------------------------------------
# Remote parsing
# ---------------------------------------------------------------------------


def parse_remote_url(raw: str) -> tuple[str, str, str] | None:
    """Extract ``(host, owner, repo)`` from any git remote URL flavour.

    Returns ``None`` if the URL does not parse cleanly.  ``urlparse`` is
    tried first; SSH ``git@host:owner/repo.git`` style is parsed
    manually as a fallback.
    """
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        from urllib.parse import urlparse

        if "://" in raw:
            u = urlparse(raw)
            host = (u.hostname or "").lower()
            if not host:
                return None
            parts = [p for p in u.path.split("/") if p]
            if len(parts) < 2:
                return None
            owner, repo = parts[0], parts[1]
            repo = repo.removesuffix(".git")
            return host, owner, repo
    except Exception:  # nosec B110
        pass
    # SSH fallback: ``git@gitcode.com:foo/bar.git``
    if "@" in raw and ":" in raw and "://" not in raw:
        user_host, path = raw.split(":", 1)
        host = user_host.split("@", 1)[-1].lower()
        parts = [p for p in path.split("/") if p]
        if len(parts) < 2:
            return None
        return host, parts[0], parts[1].removesuffix(".git")
    return None


def is_known_tracking_host(host: str) -> bool:
    return host.lower() in {_GITCODE, GITHUB_HOST, GITEE_HOST, LINEAR_HOST}


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def _safe_run_git(args: list[str], cwd: Path) -> tuple[str, int]:
    """Run a git command; return ``(stdout, rc)`` without raising."""
    import subprocess

    try:
        completed = subprocess.run(  # nosec B607  # pylint: disable=subprocess-run-check
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
        return completed.stdout.strip(), completed.returncode
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        log.debug("git %s failed: %s", " ".join(args), exc)
        return "", -1


def detect_workspace_fingerprint(
    root: Path,
    *,
    use_cache: bool = True,
    now: float | None = None,
) -> WorkspaceFingerprint:
    """Discover the workspace's git configuration.

    Falls back to a fingerprint with ``has_git=False`` when the directory
    is not a git working tree.
    """
    root = Path(root).resolve()
    fingerprint = WorkspaceFingerprint(
        workspace_root=root,
        primary_remote_url=None,
        primary_remote_host=None,
        default_branch=None,
        has_git=False,
    )

    repo_top, rc = _safe_run_git(["rev-parse", "--show-toplevel"], root)
    if rc != 0 or not repo_top:
        cached = _read_cache(root) if use_cache else None
        if cached is not None:
            return cached
        return fingerprint
    repo_root = Path(repo_top)
    fingerprint = WorkspaceFingerprint(
        workspace_root=repo_root,
        primary_remote_url=None,
        primary_remote_host=None,
        default_branch=None,
        has_git=True,
    )

    if use_cache:
        cached = _read_cache(repo_root)
        if cached is not None:
            return cached

    remote_url, host, _owner, _repo = _detect_primary_remote(repo_root)
    default_branch = _detect_default_branch(repo_root)

    trackers = _detect_trackers(repo_root)

    fingerprint = WorkspaceFingerprint(
        workspace_root=repo_root,
        primary_remote_url=remote_url,
        primary_remote_host=host,
        default_branch=default_branch,
        tracked_branches=(),
        has_git=True,
        trackers=trackers,
    )

    if use_cache:
        _write_cache(fingerprint, now=now)
    return fingerprint


def _detect_primary_remote(repo_root: Path) -> tuple[str | None, str | None, str, str]:
    """Return ``(url, host, owner, repo)`` — missing fields are ``None``."""
    stdout, rc = _safe_run_git(["remote", "-v"], repo_root)
    if rc != 0 or not stdout:
        return None, None, "", ""
    for line in stdout.splitlines():
        if "\t" not in line:
            continue
        remote, url = line.split("\t", 1)
        url = url.split(" ", 1)[0]  # drop "(fetch)" / "(push)"
        # Prefer ``origin`` if present; otherwise first remote.
        parsed = parse_remote_url(url)
        if parsed is None:
            continue
        host, owner, repo = parsed
        if remote == "origin":
            return url, host.lower(), owner, repo
    # Fall through — return the first parseable remote we saw.
    for line in stdout.splitlines():
        if "\t" not in line:
            continue
        _remote, url = line.split("\t", 1)
        url = url.split(" ", 1)[0]
        parsed = parse_remote_url(url)
        if parsed is not None:
            host, owner, repo = parsed
            return url, host.lower(), owner, repo
    return None, None, "", ""


def _detect_default_branch(repo_root: Path) -> str | None:
    """Best-effort: symbolic ref → main / master fallback → origin/HEAD."""
    stdout, rc = _safe_run_git(["symbolic-ref", "--short", "refs/remotes/origin/HEAD"], repo_root)
    if rc == 0 and stdout:
        # e.g. ``origin/main`` → ``main``
        return stdout.split("/", 1)[-1]
    for candidate in ("main", "master"):
        check, crc = _safe_run_git(["show-ref", "--verify", f"refs/heads/{candidate}"], repo_root)
        if crc == 0 and check:
            return candidate
    return None


def _detect_trackers(repo_root: Path) -> tuple[str, ...]:
    """Sniff for orchestrator adapter config files we already know about."""
    base = repo_root / ".clawcodex"
    found: list[str] = []
    if (base / "orchestrator" / "gitcode.yaml").exists():
        found.append("gitcode")
    if (base / "orchestrator" / "linear.yaml").exists():
        found.append("linear")
    if (base / "orchestrator" / "github.yaml").exists():
        found.append("github")
    return tuple(found)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def _cache_path(repo_root: Path) -> Path:
    return repo_root / ".clawcodex" / _CACHE_FILENAME


def _read_cache(repo_root: Path) -> WorkspaceFingerprint | None:
    path = _cache_path(repo_root)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    ts = float(raw.get("ts", 0))
    if (time.time() - ts) > _CACHE_TTL_SECONDS:
        return None
    try:
        return WorkspaceFingerprint(
            workspace_root=Path(raw["workspace_root"]),
            primary_remote_url=raw.get("primary_remote_url"),
            primary_remote_host=raw.get("primary_remote_host"),
            default_branch=raw.get("default_branch"),
            tracked_branches=tuple(raw.get("tracked_branches") or ()),
            has_git=bool(raw.get("has_git", False)),
            trackers=tuple(raw.get("trackers") or ()),
        )
    except (KeyError, ValueError):
        return None


def _write_cache(fp: WorkspaceFingerprint, *, now: float | None = None) -> None:
    path = _cache_path(fp.workspace_root)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(fp)
        payload["ts"] = now if now is not None else time.time()
        payload["workspace_root"] = str(fp.workspace_root)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError as exc:
        log.debug("failed to write lodestone cache: %s", exc)


def invalidate_cache(repo_root: Path) -> bool:
    """Remove the fingerprint cache; returns True on removal."""
    path = _cache_path(repo_root)
    if path.exists():
        try:
            path.unlink()
        except OSError:
            return False
        return True
    return False


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------


def build_anchor_context(
    workspace_root: Path,
    config,
    *,
    session_id: str | None = None,
    fingerprint: WorkspaceFingerprint | None = None,
    branch: str | None = None,
    env: dict[str, str] | None = None,
) -> AnchorContext:
    """Convenience builder — fingerprint auto-detected when not supplied.

    ``config`` may be a ``LodestoneConfig`` or ``None``.
    """
    fingerprint = fingerprint or detect_workspace_fingerprint(workspace_root)
    branch = branch or fingerprint.default_branch
    return AnchorContext(
        workspace_root=fingerprint.workspace_root,
        session_id=session_id,
        config=config,
        remote_url=fingerprint.primary_remote_url,
        branch=branch,
        env=env,
    )


__all__ = [
    "build_anchor_context",
    "detect_workspace_fingerprint",
    "invalidate_cache",
    "is_known_tracking_host",
    "parse_remote_url",
]


# silence unused-import linting
_ = (AnchorContext, os)

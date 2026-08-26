# pylint: disable=relative-beyond-top-level
"""Fetcher for the Community Feature Radar — shallow-clone releases, REST API for commits/PRs."""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import time
import urllib.request
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

from .models import Commit, FetchResult, PullRequest, Release, WatchSource

_log = logging.getLogger(__name__)


GITHUB_API_ROOT = "https://api.github.com"
DEFAULT_PAGE_SIZE = 30
DEFAULT_REQUEST_TIMEOUT = 15.0  # seconds
DEFAULT_USER_AGENT = "clawcodex-community-radar/0.1"
DEFAULT_CACHE_TTL_SECONDS = 30 * 24 * 60 * 60  # 30 days
CHANGELOG_CACHE_TTL_SECONDS = 24 * 60 * 60  # 24 hours
MAX_RELEASES_PER_SOURCE = 30
RATE_LIMIT_WARN_THRESHOLD = 20


# Cache helpers (commits / PRs)


def _list_cache_path(cache_dir: Path, kind: str, source: WatchSource) -> Path:
    key = f"{source.repo.replace('/', '_')}__{source.name}"
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", key) or "unknown"
    return cache_dir / kind / f"{safe}.json"


def _read_cached_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write_cached_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


# CHANGELOG raw-text cache helpers


def _changelog_cache_path(cache_dir: Path, source: WatchSource) -> Path:
    key = f"{source.repo.replace('/', '_')}__{source.name}"
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", key) or "unknown"
    return cache_dir / "changelogs" / f"{safe}.txt"


def _read_changelog_cache(cache_dir: Path, source: WatchSource, ttl: int = CHANGELOG_CACHE_TTL_SECONDS) -> str | None:
    path = _changelog_cache_path(cache_dir, source)
    if not path.exists():
        return None
    try:
        mtime = path.stat().st_mtime
        if time.time() - mtime > ttl:
            return None
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _write_changelog_cache(cache_dir: Path, source: WatchSource, text: str) -> None:
    path = _changelog_cache_path(cache_dir, source)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# Domain detection cache helpers


def _domain_cache_path(cache_dir: Path, repo: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", repo.replace("/", "_")) or "unknown"
    return cache_dir / "domains" / f"{safe}.json"


def _read_domain_cache(cache_dir: Path, repo: str, ttl: int = CHANGELOG_CACHE_TTL_SECONDS) -> str | None:
    path = _domain_cache_path(cache_dir, repo)
    if not path.exists():
        return None
    try:
        mtime = path.stat().st_mtime
        if time.time() - mtime > ttl:
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        cached = data.get("domain")
        if isinstance(cached, str):
            return cached
        return None
    except (OSError, ValueError, KeyError):
        return None


def _write_domain_cache(cache_dir: Path, repo: str, domain: str) -> None:
    path = _domain_cache_path(cache_dir, repo)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"domain": domain, "repo": repo, "cached_at": time.time()}),
        encoding="utf-8",
    )


# Keywords matched against GitHub topics + description to auto-detect
# a project's domain when the user has not set one explicitly.
_EMBODIED_AI_DOMAIN_KEYWORDS = [
    "robot",
    "robotics",
    "manipulation",
    "locomotion",
    "grasping",
    "embodied",
    "vla",
    "reinforcement-learning",
    "imitation-learning",
    "humanoid",
    "legged-robot",
    "mobile-manipulation",
    "teleoperation",
    "sim-to-real",
    "robot-learning",
]

_SPATIAL_DOMAIN_KEYWORDS = [
    "nerf",
    "neural-radiance",
    "3d-reconstruction",
    "3d-vision",
    "gaussian-splatting",
    "point-cloud",
    "slam",
    "lidar",
    "novel-view-synthesis",
    "volumetric",
    "mesh",
    "voxel",
    "spatial-intelligence",
    "radiance-field",
]


def detect_repo_domain(
    repo: str,
    cache_dir: Path | str,
    *,
    github_token: str | None = None,
) -> str | None:
    """Auto-detect a GitHub repo's domain from its topics and description.

    Calls ``GET /repos/{owner}/{repo}`` and matches keywords against the
    repo's ``topics`` list and ``description`` field.  Results are cached
    for 24 h under ``cache_dir/domains/`` so repeated scans don't burn
    API quota.

    Returns ``"embodied_ai"``, ``"spatial_intelligence"``, or ``None``
    (meaning stay ``general`` / software engineering).
    """
    cache_dir = Path(cache_dir)

    # Check cache first (negative results are cached as empty string)
    cached = _read_domain_cache(cache_dir, repo)
    if cached is not None:
        return cached

    owner, name = repo.split("/", 1)
    url = f"{GITHUB_API_ROOT}/repos/{quote(owner)}/{quote(name)}"

    req = urllib.request.Request(url)
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    req.add_header("User-Agent", DEFAULT_USER_AGENT)
    if github_token:
        req.add_header("Authorization", f"Bearer {github_token}")

    try:
        with urllib.request.urlopen(req, timeout=DEFAULT_REQUEST_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        _log.warning("domain detection failed for %s: %s", repo, exc)
        return None

    # Build a single lower-case blob from topics + description
    topics = [str(t).lower() for t in (data.get("topics") or [])]
    description = (data.get("description") or "").lower()
    combined = " ".join(topics) + " " + description

    detected: str | None = None
    for kw in _EMBODIED_AI_DOMAIN_KEYWORDS:
        if kw in combined:
            detected = "embodied_ai"
            break
    if detected is None:
        for kw in _SPATIAL_DOMAIN_KEYWORDS:
            if kw in combined:
                detected = "spatial_intelligence"
                break

    # Cache even negative results so we don't re-fetch on every scan
    _write_domain_cache(cache_dir, repo, detected or "")
    _log.info("auto-detected domain for %s: %s", repo, detected or "general")
    return detected


# Fetcher


class Fetcher:
    """Pulls incremental upstream data for a list of WatchSources.

    ``client`` accepts an injectable HTTP client so tests can swap in a
    fake. Defaults to :class:`httpx.Client` with a sane User-Agent and
    timeout.
    """

    def __init__(
        self,
        github_token: str | None = None,
        cache_dir: Path | str = ".cache/community-radar",
        *,
        client: Any | None = None,
        page_size: int = DEFAULT_PAGE_SIZE,
        timeout: float = DEFAULT_REQUEST_TIMEOUT,
        cache_ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
    ) -> None:
        self.github_token = github_token or os.environ.get("GITHUB_TOKEN")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.page_size = page_size
        self.timeout = timeout
        self.cache_ttl_seconds = cache_ttl_seconds
        self._owns_client = client is None
        self._client = client or self._build_client()

        if not self.github_token:
            _log.warning(
                "GITHUB_TOKEN is not set — anonymous GitHub API rate limit is "
                "60 requests/hour. Set the environment variable to avoid "
                "rate-limit errors. See https://github.com/settings/tokens"
            )
        else:
            self._validate_token()

    # Lifecycle

    def _build_client(self) -> Any:
        try:
            import httpx  # type: ignore
        except ImportError as exc:  # pragma: no cover - env dep
            raise RuntimeError(
                "httpx is required for the community radar fetcher. Install it with `pip install httpx`."
            ) from exc
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": DEFAULT_USER_AGENT,
        }
        if self.github_token:
            headers["Authorization"] = f"Bearer {self.github_token}"
        return httpx.Client(timeout=self.timeout, headers=headers, follow_redirects=True)

    def _validate_token(self) -> None:
        """Verify the GitHub token by calling ``GET /rate_limit``.

        A failing token (401 Bad credentials) is logged as a clear error
        so the operator can fix it before the scan starts making real API
        calls.  The call is cheap (does not count against the rate limit)
        and includes the remaining quota when the token is valid.
        """
        try:
            resp = self._client.get(
                f"{GITHUB_API_ROOT}/rate_limit",
                timeout=self.timeout,
            )
        except Exception as exc:  # noqa: BLE001
            _log.debug("rate_limit check failed (%s); token not verified", exc)
            return
        if resp.status_code == 200:
            data = resp.json()
            core = data.get("resources", {}).get("core", {})
            remaining = core.get("remaining", "?")
            limit = core.get("limit", "?")
            _log.info("GitHub token is valid (rate limit: %s/%s remaining)", remaining, limit)
        elif resp.status_code == 401:
            _log.warning(
                "GitHub token is invalid (401 Bad credentials). "
                "Check GITHUB_TOKEN and regenerate at "
                "https://github.com/settings/tokens"
            )
        elif resp.status_code == 403 and "rate limit" in (resp.text or "").lower():
            _log.warning("GitHub rate limit already exhausted before scan started")
        else:
            _log.debug("rate_limit check returned %s; token not verified", resp.status_code)

    def close(self) -> None:
        if self._owns_client:
            try:
                self._client.close()
            except Exception:  # noqa: BLE001  # nosec B110
                pass

    def __enter__(self) -> Fetcher:  # noqa: PYI034
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # Public API

    def fetch(
        self,
        source: WatchSource,
        *,
        incremental: bool = False,
        since: str | None = None,
    ) -> FetchResult:
        """Fetch everything ``source`` requested. Always returns a result."""
        result = FetchResult(source=source.name)
        try:
            if source.track_releases:
                result.releases = self.fetch_releases(source, incremental=incremental, since=since)
            if source.track_commits:
                result.commits = self.fetch_commits(source, since=since)
            if source.track_prs:
                result.prs = self.fetch_prs(source, since=since)
        except Exception as exc:  # noqa: BLE001 — log + degrade gracefully
            _log.exception("fetch failed for source %s: %s", source.name, exc)
            result.errors.append(f"{type(exc).__name__}: {exc}")
        return result

    def fetch_all(
        self,
        sources: Iterable[WatchSource],
        *,
        incremental: bool = False,
        since: str | None = None,
    ) -> list[FetchResult]:
        return [self.fetch(s, incremental=incremental, since=since) for s in sources]

    # Releases (four-layer fallback: API → Tags → CHANGELOG raw → git clone)

    def fetch_releases(
        self,
        source: WatchSource,
        *,
        incremental: bool = False,
        since: str | None = None,
    ) -> list[Release]:
        """Fetch releases via four-layer fallback strategy.

        Layer 1:  GitHub Releases API (structured, fast)
        Layer 1.5: Supplement with CHANGELOG raw_body when L1 succeeds
        Layer 2:  GitHub Tags API (covers tag-only projects)
        Layer 3:  GitHub Content API → CHANGELOG raw (no clone needed)
        Layer 4:  git clone + local file parse (last resort / non-GitHub)

        When *since* is given (ISO-8601), releases with ``published_at``
        older than *since* are dropped.
        """
        releases: list[Release] = []

        # ── Layer 1: GitHub Releases API ──────────────────────────
        releases = self._fetch_releases_api(source)
        if releases:
            # ── Layer 1.5: supplement with CHANGELOG raw_body ─────
            changelog_text = self._fetch_changelog_raw(source)
            if changelog_text:
                releases = self._merge_changelog_raw(releases, changelog_text)
            releases = self._apply_tag_filter(releases, source)
            releases = releases[:MAX_RELEASES_PER_SOURCE]
            if since:
                releases = [r for r in releases if r.published_at and r.published_at >= since]
            return releases

        # ── Layer 2: GitHub Tags API ──────────────────────────────
        releases = self._fetch_tags_api(source)
        if releases:
            releases = self._apply_tag_filter(releases, source)
            releases = releases[:MAX_RELEASES_PER_SOURCE]
            if since:
                releases = [r for r in releases if r.published_at and r.published_at >= since]
            return releases

        # ── Layer 3: GitHub Content API → CHANGELOG raw ──────────
        changelog_text = self._fetch_changelog_raw(source)
        if changelog_text:
            releases = self._parse_changelog(changelog_text, source=source)
            if releases:
                releases = self._apply_tag_filter(releases, source)
                releases = releases[:MAX_RELEASES_PER_SOURCE]
                if since:
                    releases = [r for r in releases if r.published_at and r.published_at >= since]
                return releases

        # ── Layer 4: git clone + local CHANGELOG file ─────────────
        return self._fetch_releases_via_clone(source, incremental=incremental)

    # ── Layer 1: GitHub Releases API ──────────────────────────────

    def _fetch_releases_api(self, source: WatchSource) -> list[Release]:
        """Fetch releases via ``GET /repos/{owner}/{repo}/releases``."""
        owner, name = source.repo.split("/", 1)
        url = f"{GITHUB_API_ROOT}/repos/{quote(owner)}/{quote(name)}/releases"
        params: dict[str, Any] = {"per_page": MAX_RELEASES_PER_SOURCE, "page": 1}
        try:
            resp = self._request("GET", url, params=params)
        except Exception as exc:  # noqa: BLE001
            _log.warning("Layer 1 (releases API) request failed for %s: %s", source.name, exc)
            return []
        self._check_rate_limit(resp)
        if resp.status_code == 200:
            payload = resp.json() or []
            return [
                Release(
                    tag=str(item.get("tag_name", "")),
                    name=str(item.get("name") or item.get("tag_name", "")),
                    body=str(item.get("body") or ""),
                    published_at=item.get("published_at"),
                    url=str(item.get("html_url") or ""),
                    is_prerelease=bool(item.get("prerelease", False)),
                )
                for item in payload
                if isinstance(item, dict)
            ]
        elif resp.status_code in (401, 403):
            _log.warning(
                "Layer 1 (releases API) returned %s for %s, falling back",
                resp.status_code,
                source.name,
            )
        elif resp.status_code == 404:
            _log.info("Layer 1 (releases API) 404 for %s (repo not found or private)", source.name)
        else:
            _log.warning("Layer 1 (releases API) unexpected %s for %s", resp.status_code, source.name)
        return []

    # ── Layer 1.5: merge CHANGELOG raw_body ───────────────────────

    @staticmethod
    def _merge_changelog_raw(releases: list[Release], changelog_text: str) -> list[Release]:
        """Match CHANGELOG sections to API releases and populate ``raw_body``."""
        sections = Fetcher._split_changelog_sections(changelog_text)
        if not sections:
            return releases
        for r in releases:
            for candidate in (r.tag, r.name, f"v{r.tag}", r.tag.lstrip("v")):
                if candidate in sections:
                    r.raw_body = sections[candidate]
                    break
        return releases

    @staticmethod
    def _split_changelog_sections(text: str) -> dict[str, str]:
        """Split changelog markdown into ``{version: body}`` mapping.

        Reuses the same heading-split strategy as ``_parse_changelog`` but
        returns a dict keyed by version string instead of ``Release`` objects.
        """
        blocks = re.split(r"^##\s+", text, flags=re.MULTILINE)
        if len(blocks) <= 1:
            return {}

        heading_re = re.compile(
            r"^\[?(?P<version>v?\d[\d.]*(?:-(?:rc|alpha|beta|dev|pre)\.?\d*)?)\]?"
            r"\s*(?:[-—]\s*(?P<date>\d{4}-\d{2}-\d{2}))?\s*$"
        )
        sections: dict[str, str] = {}
        for block in blocks[1:]:
            first_line_end = block.find("\n")
            heading_line = block[:first_line_end].strip() if first_line_end != -1 else block.strip()
            m = heading_re.match(heading_line)
            if not m:
                m2 = re.match(
                    r"^(?P<version>.+?)\s*[-—]\s*(?P<date>\d{4}-\d{2}-\d{2})\s*$",
                    heading_line,
                )
                version = (
                    m2.group("version").strip().lstrip("[").rstrip("]")
                    if m2
                    else heading_line.strip().lstrip("[").rstrip("]")
                )
            else:
                version = m.group("version")
            start = first_line_end + 1 if first_line_end != -1 else len(block)
            body = block[start:].strip()
            sections[version] = body
        return sections

    # ── Layer 2: GitHub Tags API ──────────────────────────────────

    def _fetch_tags_api(self, source: WatchSource) -> list[Release]:
        """Fetch tags via ``GET /repos/{owner}/{repo}/tags``."""
        owner, name = source.repo.split("/", 1)
        url = f"{GITHUB_API_ROOT}/repos/{quote(owner)}/{quote(name)}/tags"
        params: dict[str, Any] = {"per_page": MAX_RELEASES_PER_SOURCE, "page": 1}
        try:
            resp = self._request("GET", url, params=params)
        except Exception as exc:  # noqa: BLE001
            _log.warning("Layer 2 (tags API) request failed for %s: %s", source.name, exc)
            return []
        self._check_rate_limit(resp)
        if resp.status_code == 200:
            owner_repo = source.repo
            payload = resp.json() or []
            return [
                Release(
                    tag=str(item.get("name", "")),
                    name=str(item.get("name", "")),
                    body="",
                    published_at=None,
                    url=f"https://github.com/{owner_repo}/releases/tag/{quote(item.get('name', ''))}",
                    is_prerelease=bool(
                        re.search(r"-(?:rc|alpha|beta|dev|pre)\.?\d*$", str(item.get("name", "")), re.IGNORECASE)
                    ),
                )
                for item in payload
                if isinstance(item, dict)
            ]
        elif resp.status_code in (401, 403):
            _log.warning(
                "Layer 2 (tags API) returned %s for %s, falling back",
                resp.status_code,
                source.name,
            )
        elif resp.status_code == 404:
            _log.info("Layer 2 (tags API) 404 for %s", source.name)
        else:
            _log.warning("Layer 2 (tags API) unexpected %s for %s", resp.status_code, source.name)
        return []

    # ── Layer 3: GitHub Content API → CHANGELOG raw ──────────────

    def _fetch_changelog_raw(self, source: WatchSource) -> str | None:
        """Fetch CHANGELOG via ``GET /repos/{owner}/{repo}/contents/{path}``.

        Tries the same file paths as ``_read_changelog``, with a 24 h cache.
        Returns the raw markdown text or ``None`` if no file was found.
        """
        # Check cache first
        cached = _read_changelog_cache(self.cache_dir, source)
        if cached is not None:
            _log.debug("changelog cache hit for %s", source.name)
            return cached

        owner, name = source.repo.split("/", 1)
        paths_to_try: list[str] = []
        if source.changelog_path:
            paths_to_try.append(source.changelog_path)
        paths_to_try.extend(
            [
                "CHANGELOG.md",
                "CHANGELOG",
                "RELEASE.md",
                "RELEASE_NOTES.md",
                "History.md",
            ]
        )

        seen: set[str] = set()
        for rel_path in paths_to_try:
            if rel_path in seen:
                continue
            seen.add(rel_path)
            url = f"{GITHUB_API_ROOT}/repos/{quote(owner)}/{quote(name)}/contents/{quote(rel_path)}"
            try:
                resp = self._request("GET", url)
            except Exception as exc:  # noqa: BLE001
                _log.debug("Layer 3 (contents API) request failed for %s %s: %s", source.name, rel_path, exc)
                continue
            self._check_rate_limit(resp)
            if resp.status_code == 200:
                payload = resp.json()
                if isinstance(payload, dict) and payload.get("encoding") == "base64":
                    content_b64 = payload.get("content") or ""
                    try:
                        text = base64.b64decode(content_b64).decode("utf-8")
                    except (ValueError, UnicodeDecodeError) as exc:
                        _log.warning("Layer 3 base64 decode failed for %s %s: %s", source.name, rel_path, exc)
                        continue
                    _write_changelog_cache(self.cache_dir, source, text)
                    _log.debug("changelog fetched via API for %s (%s)", source.name, rel_path)
                    return text
                # Content API may return a list for directories — skip
                _log.debug("Layer 3 unexpected response shape for %s %s", source.name, rel_path)
            elif resp.status_code == 403 and "rate limit" in (resp.text or "").lower():
                _log.warning("Layer 3 (contents API) rate limited for %s", source.name)
                return None
            elif resp.status_code == 404:
                _log.debug("Layer 3 file not found for %s: %s", source.name, rel_path)
            else:
                _log.debug("Layer 3 returned %s for %s %s", resp.status_code, source.name, rel_path)
        return None

    # ── Layer 4: git clone + local CHANGELOG (V1 logic, preserved) ──

    def _fetch_releases_via_clone(self, source: WatchSource, *, incremental: bool = False) -> list[Release]:
        """Fallback: shallow-clone the repo and parse CHANGELOG from disk."""
        clone_dir = self._clone_dir(source)
        repo_url = f"https://github.com/{source.repo}.git"

        if not clone_dir.exists():
            self._git_clone(repo_url, clone_dir)
        elif incremental:
            self._git_fetch(clone_dir)

        changelog_text = self._read_changelog(clone_dir, source)
        if not changelog_text:
            _log.warning("no changelog found for %s (searched in %s)", source.name, clone_dir)
            return []

        releases = self._parse_changelog(changelog_text, source=source)
        releases = self._apply_tag_filter(releases, source)
        return releases[:MAX_RELEASES_PER_SOURCE]

    # ── Tag filter (applied across all layers) ────────────────────

    @staticmethod
    def _apply_tag_filter(releases: list[Release], source: WatchSource) -> list[Release]:
        """Filter releases matching ``source.release_tag_filter`` (regex, prefix-compatible)."""
        tag_filter = source.release_tag_filter
        if not tag_filter:
            return releases
        filtered = [r for r in releases if re.search(tag_filter, r.tag)]
        if not filtered and releases:
            _log.debug(
                "release_tag_filter '%s' excluded all %d releases for %s",
                tag_filter,
                len(releases),
                source.name,
            )
        return filtered

    # ── Rate limit helper ─────────────────────────────────────────

    @staticmethod
    def _check_rate_limit(resp: Any) -> None:
        """Log a warning when the GitHub API rate limit is running low."""
        try:
            remaining = resp.headers.get("X-RateLimit-Remaining")
            if remaining is not None:
                remaining = int(remaining)
                if remaining < RATE_LIMIT_WARN_THRESHOLD:
                    _log.warning(
                        "GitHub API rate limit low (%s remaining). Consider setting GITHUB_TOKEN to increase quota.",
                        remaining,
                    )
        except (ValueError, AttributeError):
            pass

    # Commits / PRs (lightweight, optional)

    def fetch_commits(self, source: WatchSource, *, since: str | None = None) -> list[Commit]:
        owner, name = source.repo.split("/", 1)
        cache_path = _list_cache_path(self.cache_dir, "commits", source)
        cached = _read_cached_json(cache_path) or {"fetched_at": 0.0, "items": []}
        fetched_at = float(cached.get("fetched_at") or 0.0)
        if not since and fetched_at and (time.time() - fetched_at) < self.cache_ttl_seconds:
            return [Commit.from_dict(item) for item in cached.get("items", [])]

        params: dict[str, Any] = {"per_page": self.page_size, "page": 1}
        if since:
            params["since"] = since
        url = f"{GITHUB_API_ROOT}/repos/{quote(owner)}/{quote(name)}/commits"
        response = self._request("GET", url, params=params)
        self._check_rate_limit(response)
        if response.status_code != 200:
            raise RuntimeError(f"GitHub commits {response.status_code}")
        payload = response.json() or []
        commits = [
            Commit(
                sha=str(item.get("sha", "")),
                message=str((item.get("commit") or {}).get("message") or ""),
                author=((item.get("commit") or {}).get("author") or {}).get("name"),
                committed_at=((item.get("commit") or {}).get("author") or {}).get("date"),
                url=str(item.get("html_url") or ""),
            )
            for item in payload
            if isinstance(item, dict)
        ]
        _write_cached_json(cache_path, {"fetched_at": time.time(), "items": [c.to_dict() for c in commits]})
        return commits

    def fetch_prs(self, source: WatchSource, *, since: str | None = None) -> list[PullRequest]:
        owner, name = source.repo.split("/", 1)
        cache_path = _list_cache_path(self.cache_dir, "prs", source)
        cached = _read_cached_json(cache_path) or {"fetched_at": 0.0, "items": []}
        fetched_at = float(cached.get("fetched_at") or 0.0)
        if not since and fetched_at and (time.time() - fetched_at) < self.cache_ttl_seconds:
            return [PullRequest.from_dict(item) for item in cached.get("items", [])]

        params: dict[str, Any] = {
            "state": "closed",
            "per_page": self.page_size,
            "page": 1,
            "sort": "updated",
            "direction": "desc",
        }
        if since:
            params["since"] = since
        url = f"{GITHUB_API_ROOT}/repos/{quote(owner)}/{quote(name)}/pulls"
        response = self._request("GET", url, params=params)
        self._check_rate_limit(response)
        if response.status_code != 200:
            raise RuntimeError(f"GitHub pulls {response.status_code}")
        payload = response.json() or []
        prs = [
            PullRequest(
                number=int(item.get("number") or 0),
                title=str(item.get("title") or ""),
                state=str(item.get("state") or ""),
                merged_at=item.get("merged_at"),
                url=str(item.get("html_url") or ""),
                body=str(item.get("body") or ""),
            )
            for item in payload
            if isinstance(item, dict)
        ]
        _write_cached_json(cache_path, {"fetched_at": time.time(), "items": [p.to_dict() for p in prs]})
        return prs

    # Git clone helpers

    @staticmethod
    def _safe_clone_name(source_name: str) -> str:
        return re.sub(r"[^A-Za-z0-9._-]", "_", source_name) or "unknown"

    def _clone_dir(self, source: WatchSource) -> Path:
        return self.cache_dir / "git-clones" / self._safe_clone_name(source.name)

    @staticmethod
    def _git_clone(repo_url: str, clone_dir: Path) -> None:
        """Shallow-clone *repo_url* to *clone_dir* (--depth 30)."""
        from clawcodex_ext.utils.git import _run_git  # type: ignore

        clone_dir.parent.mkdir(parents=True, exist_ok=True)
        _log.info("cloning %s (--depth 30) into %s", repo_url, clone_dir)
        _stdout, stderr, rc = _run_git(
            ["clone", "--depth", "30", repo_url, str(clone_dir)],
            timeout=120.0,
        )
        if rc != 0:
            raise RuntimeError(f"git clone failed: {stderr}")

    @staticmethod
    def _git_fetch(clone_dir: Path) -> None:
        """Shallow-fetch latest and reset (--depth 30 stays shallow)."""
        from clawcodex_ext.utils.git import _run_git  # type: ignore

        _log.info("fetching --depth 30 in %s", clone_dir)
        _stdout, stderr, rc = _run_git(["fetch", "--depth", "30", "origin"], cwd=str(clone_dir), timeout=60.0)
        if rc != 0:
            raise RuntimeError(f"git fetch failed: {stderr}")
        _run_git(["reset", "--hard", "origin/HEAD"], cwd=str(clone_dir), timeout=30.0)

    @staticmethod
    def _read_changelog(clone_dir: Path, source: WatchSource) -> str | None:
        """Read CHANGELOG content from the cloned repo.

        Tries *source.changelog_path* first, then common fallback names.
        """
        paths_to_try: list[str] = []
        if source.changelog_path:
            paths_to_try.append(source.changelog_path)
        paths_to_try.extend(
            [
                "CHANGELOG.md",
                "CHANGELOG",
                "RELEASE.md",
                "RELEASE_NOTES.md",
                "History.md",
            ]
        )

        seen: set[str] = set()
        for rel_path in paths_to_try:
            if rel_path in seen:
                continue
            seen.add(rel_path)
            full_path = clone_dir / rel_path
            if full_path.exists():
                _log.debug("reading changelog: %s", full_path)
                return full_path.read_text(encoding="utf-8", errors="replace")
        return None

    @staticmethod
    def _parse_changelog(
        text: str,
        *,
        source: WatchSource,
    ) -> list[Release]:
        """Parse Keep-a-Changelog-style markdown into Release objects.

        Supports::

            ## [1.0.0] - 2026-01-15
            ## 1.0.0 - 2026-01-15
            ## 1.0.0 (2026-01-15)
        """

        # Split on ``## `` headings. The first group is pre-heading content
        # (title, etc.) and is skipped.
        blocks = re.split(r"^##\s+", text, flags=re.MULTILINE)
        if len(blocks) <= 1:
            return []

        # Parse version + optional date from the heading line of each block.
        heading_re = re.compile(
            r"^\[?(?P<version>v?\d[\d.]*(?:-(?:rc|alpha|beta|dev|pre)\.?\d*)?)\]?"
            r"\s*(?:[-—]\s*(?P<date>\d{4}-\d{2}-\d{2}))?\s*$"
        )
        owner_repo = source.repo
        releases: list[Release] = []
        for block in blocks[1:]:  # skip pre-heading content
            first_line_end = block.find("\n")
            heading_line = block[:first_line_end].strip() if first_line_end != -1 else block.strip()
            m = heading_re.match(heading_line)
            if not m:
                # Try a looser match: anything up to a date-like pattern
                m2 = re.match(
                    r"^(?P<version>.+?)\s*[-—]\s*(?P<date>\d{4}-\d{2}-\d{2})\s*$",
                    heading_line,
                )
                if m2:
                    version = m2.group("version").strip().lstrip("[").rstrip("]")
                    date_str = m2.group("date")
                else:
                    version = heading_line.strip().lstrip("[").rstrip("]")
                    date_str = None
            else:
                version = m.group("version")
                date_str = m.group("date")

            start = first_line_end + 1 if first_line_end != -1 else len(block)
            body = block[start:].strip()
            tag = version
            published_at = f"{date_str}T00:00:00Z" if date_str else None
            url = f"https://github.com/{owner_repo}/releases/tag/{tag}"
            is_prerelease = bool(re.search(r"-(?:rc|alpha|beta|dev|pre)\.?\d*$", version, re.IGNORECASE))
            releases.append(
                Release(
                    tag=tag,
                    name=version,
                    body=body,
                    published_at=published_at,
                    url=url,
                    is_prerelease=is_prerelease,
                )
            )
        return releases

    # GitHub Search API

    def search_repositories(
        self,
        query: str,
        *,
        per_page: int = 30,
        sort: str = "stars",
        order: str = "desc",
    ) -> list[dict[str, Any]]:
        """Search GitHub repositories via ``GET /search/repositories``.

        Args:
            query: The raw ``q`` parameter value (already assembled by
                   :func:`build_search_query`).
            per_page: Results per page (max 100).
            sort: Sort field — ``stars``, ``forks``, ``updated``, or
                  ``help-wanted-issues``.
            order: ``desc`` or ``asc``.

        Returns:
            List of repository ``items`` from the GitHub response, or an
            empty list on any error.
        """
        params: dict[str, Any] = {
            "q": query,
            "sort": sort,
            "order": order,
            "per_page": min(per_page, 100),
        }
        url = f"{GITHUB_API_ROOT}/search/repositories"

        try:
            resp = self._client.get(url, params=params, timeout=self.timeout)
        except Exception as exc:  # noqa: BLE001
            _log.warning("search_repositories request failed: %s", exc)
            return []

        self._check_rate_limit(resp)

        if resp.status_code == 200:
            data = resp.json()
            items = data.get("items")
            if isinstance(items, list):
                total = data.get("total_count", 0)
                _log.info("GitHub Search returned %d total, %d in page", total, len(items))
                return items
            return []
        elif resp.status_code == 422:
            _log.warning("GitHub Search query invalid (422): %s", resp.text[:500])
            return []
        else:
            _log.warning("search_repositories returned %s", resp.status_code)
            return []

    # Internal HTTP wrapper (also reused by tests via monkeypatch)

    def _request(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        return self._client.request(method, url, params=params, headers=headers or {})


# Helpers usable from tests / CLI without constructing a full Fetcher


def make_fetcher(
    github_token: str | None = None,
    cache_dir: Path | str = ".cache/community-radar",
    *,
    client_factory: Callable[[], Any] | None = None,
) -> Fetcher:
    """Build a Fetcher; ``client_factory`` is honoured by tests only."""
    if client_factory is not None:
        # Tests inject a fake client. We mark ``_owns_client=False`` so
        # ``close()`` does not try to call ``.close()`` on the stub.
        fetcher = Fetcher(
            github_token=github_token,
            cache_dir=cache_dir,
            client=client_factory(),
        )
        return fetcher
    return Fetcher(github_token=github_token, cache_dir=cache_dir)


def isoformat_now() -> str:
    """Convenience for callers that want a deterministic timestamp."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

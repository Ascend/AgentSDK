"""Multi-platform issue registry and HTTP client for Community Radar.

Defines :class:`IssuePlatform` configurations for GitCode, GitHub, and
Gitee, plus a minimal :class:`IssueClient` that only implements the two
API operations the issue-sync feature needs:

* ``create_issue`` — POST a new issue with title, body, labels.
* ``list_issues``  — GET issues filtered by label (for L2 remote dedup).

The module is deliberately independent of ``extensions/orchestrator/``
to keep the three-layer architecture clean.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from typing import Any

import httpx

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Platform definitions
# ---------------------------------------------------------------------------


@dataclass
class IssuePlatform:
    """API configuration for a single code-hosting platform."""

    name: str  # "gitcode" | "github" | "gitee"
    default_endpoint: str  # API base URL
    web_host: str  # Web front-end domain (used for issue links)
    auth_mode: str  # "access_token" (query param) | "bearer" (header) | "token" (query param)
    auth_param: str  # Query param name or header name
    accept_header: str = "application/json"
    # API path templates
    create_issue_path: str = "/repos/{owner}/{repo}/issues"
    list_issues_path: str = "/repos/{owner}/{repo}/issues"
    get_issue_path: str = "/repos/{owner}/{repo}/issues/{issue_number}"
    # Whether ``{repo}`` is in the create-issue URL path.  When False, *repo*
    # is sent as a top-level field in the request body (GitCode uses this pattern).
    create_issue_repo_in_path: bool = True
    # Body serialisation for create-issue: "json" or "form" (GitCode requires "form").
    create_issue_body_format: str = "json"
    # When True, labels are omitted from the create-issue payload and added
    # in a follow-up call via :attr:`labels_path` (GitCode WAF blocks labels
    # in form-encoded POST bodies for non-member users).
    create_issue_skip_labels: bool = False
    # Path template for adding labels to an existing issue (used as the
    # follow-up when ``create_issue_skip_labels`` is True).
    labels_path: str = "/repos/{owner}/{repo}/issues/{issue_number}/labels"
    # Git remote URL patterns for auto-detection
    git_remote_patterns: list[str] = field(default_factory=list)
    # How labels must be serialized: "array" (JSON array) or "comma" (comma-separated string)
    labels_format: str = "array"
    # Environment variable names to look up the API token (checked in order)
    token_env_vars: tuple[str, ...] = ()


_PLATFORMS: dict[str, IssuePlatform] = {
    "gitcode": IssuePlatform(
        name="gitcode",
        default_endpoint="https://api.gitcode.com/api/v5",
        web_host="https://gitcode.com",
        auth_mode="access_token",
        auth_param="access_token",
        accept_header="application/json",
        # GitCode create-issue endpoint is /repos/{owner}/issues — repo is
        # passed as a form field rather than embedded in the URL path.
        create_issue_path="/repos/{owner}/issues",
        create_issue_repo_in_path=False,
        create_issue_body_format="form",
        create_issue_skip_labels=True,
        labels_format="comma",
        token_env_vars=("GITCODE_TOKEN", "GITCODE_API_KEY"),
        git_remote_patterns=[
            r"git@gitcode\.com[:/](.+?)/(.+?)(?:\.git)?$",
            r"https?://gitcode\.com/(.+?)/(.+?)(?:\.git)?$",
        ],
    ),
    "github": IssuePlatform(
        name="github",
        default_endpoint="https://api.github.com",
        web_host="https://github.com",
        auth_mode="bearer",
        auth_param="Authorization",
        accept_header="application/vnd.github+json",
        token_env_vars=("GITHUB_TOKEN", "GITHUB_API_KEY", "GH_TOKEN"),
        git_remote_patterns=[
            r"git@github\.com[:/](.+?)/(.+?)(?:\.git)?$",
            r"https?://github\.com/(.+?)/(.+?)(?:\.git)?$",
        ],
    ),
    "gitee": IssuePlatform(
        name="gitee",
        default_endpoint="https://gitee.com/api/v5",
        web_host="https://gitee.com",
        auth_mode="token",
        auth_param="access_token",
        accept_header="application/json",
        labels_format="comma",
        token_env_vars=("GITEE_TOKEN", "GITEE_API_KEY"),
        git_remote_patterns=[
            r"git@gitee\.com[:/](.+?)/(.+?)(?:\.git)?$",
            r"https?://gitee\.com/(.+?)/(.+?)(?:\.git)?$",
        ],
    ),
}


# ---------------------------------------------------------------------------
# Target resolution
# ---------------------------------------------------------------------------


@dataclass
class ResolvedTarget:
    """A fully-resolved platform + repo + token ready for API calls."""

    platform: IssuePlatform
    owner: str
    repo: str
    api_token: str

    @property
    def repo_slug(self) -> str:
        return f"{self.platform.name}/{self.owner}/{self.repo}"

    @property
    def web_url(self) -> str:
        return f"{self.platform.web_host}/{self.owner}/{self.repo}"


def _infer_platform_from_url(url: str) -> str | None:
    """Return the platform name inferred from a repo-URL domain, or None."""
    for name, plat in _PLATFORMS.items():
        for pattern in plat.git_remote_patterns:
            if re.search(pattern, url):
                return name
    # Fallback: check if the domain contains a known platform name
    lower = url.lower()
    for name in _PLATFORMS:
        if name in lower:
            return name
    return None


_GIT_TIMEOUT: float = 60.0


def _run_git(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a git command with captured output and a hard timeout.

    A timed-out command yields a non-zero result instead of raising, so
    callers degrade gracefully (no remote detected) instead of crashing.
    """
    try:
        return subprocess.run(  # nosec B607
            args,
            capture_output=True,
            text=True,
            check=False,
            timeout=_GIT_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(args, 1, "", "")


def _detect_from_git_remote() -> tuple[IssuePlatform, str, str] | None:
    """Try to detect platform + owner + repo from the current git clone.

    Priority:
        1. Tracking remote of the current branch
        2. ``origin``
        3. First remote whose URL matches any registered platform

    Returns ``(platform, owner, repo)`` or ``None``.
    """
    # Collect candidate remote URLs
    candidates: list[str] = []

    # 1. Tracking remote
    r = _run_git(["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"])
    if r.returncode == 0:
        remote_name = r.stdout.strip().split("/")[0]
        r2 = _run_git(["git", "remote", "get-url", remote_name])
        if r2.returncode == 0:
            candidates.append(r2.stdout.strip())

    # 2. origin
    r = _run_git(["git", "remote", "get-url", "origin"])
    if r.returncode == 0 and r.stdout.strip() not in candidates:
        candidates.append(r.stdout.strip())

    # 3. All remotes
    r = _run_git(["git", "remote", "-v"])
    if r.returncode == 0:
        for line in r.stdout.strip().split("\n"):
            parts = line.split()
            if len(parts) >= 2 and parts[1] not in candidates:
                candidates.append(parts[1])

    # Try to match each URL against registered platforms
    for url in candidates:
        for platform in _PLATFORMS.values():
            for pattern in platform.git_remote_patterns:
                m = re.match(pattern, url)
                if m:
                    return platform, m.group(1), m.group(2)

    return None


def _parse_explicit_repo(raw: str) -> tuple[str | None, str, str]:
    """Parse ``owner/repo`` or ``platform.com/owner/repo``.

    Returns ``(platform_name_or_none, owner, repo)``.
    When the input cannot be parsed (no ``/`` separator), returns
    ``(None, "", "")`` so callers can treat it as invalid.
    """
    stripped = raw.strip().rstrip("/").removesuffix(".git")
    if "/" not in stripped:
        return None, "", ""
    parts = stripped.split("/")
    if len(parts) >= 3 and "." in parts[0]:
        pname = _infer_platform_from_url(stripped)
        return pname, parts[-2], parts[-1]
    elif len(parts) == 2:
        return None, parts[0], parts[1]
    return None, "", ""


def _resolve_from_repo_string(
    raw: str,
    explicit_platform: str | None = None,
) -> tuple[IssuePlatform | None, str, str]:
    """Resolve platform + owner + repo from a raw repo string.

    ``explicit_platform`` (e.g. a CLI ``--platform`` flag) overrides the
    platform inferred from the URL. When neither identifies a known
    platform, falls back to the default (gitcode). The owner and repo
    are always parsed out of *raw*; pass ``""`` to mean "do not parse".
    """
    pname, owner, repo = _parse_explicit_repo(raw)
    if explicit_platform:
        pname = explicit_platform
    platform: IssuePlatform | None = None
    if pname:
        platform = _PLATFORMS.get(pname)
    else:
        detected = _infer_platform_from_url(raw)
        if detected:
            platform = _PLATFORMS.get(detected)
        elif _PLATFORMS:
            platform = _PLATFORMS.get("gitcode")
    return platform, owner, repo


def _resolve_token(platform: IssuePlatform, config_token: str = "") -> str | None:  # nosec B107
    """Resolve API token for *platform*.

    Priority:
        1. Platform-specific env vars (e.g. GITCODE_TOKEN)
        2. Generic CLAWCODEX_ISSUE_TOKEN env var
        3. Config file api_token field
    """
    for env_name in platform.token_env_vars:
        token = os.environ.get(env_name)
        if token:
            return token
    generic = os.environ.get("CLAWCODEX_ISSUE_TOKEN")
    if generic:
        return generic
    if config_token:
        return config_token
    return None


def resolve_target(
    *,
    config_target_repo: str = "",
    config_api_token: str = "",
    cli_repo: str | None = None,
    cli_platform: str | None = None,
) -> ResolvedTarget | None:
    """Resolve the target platform + owner + repo + token.

    Priority (highest to lowest):
        1. CLI ``--repo`` / ``--platform`` flags
        2. Config file ``target_repo``
        3. Git remote auto-detection

    Returns ``None`` when no target can be determined — callers should
    print a warning and exit gracefully.
    """
    platform: IssuePlatform | None = None
    owner: str = ""
    repo: str = ""

    # ── 1. CLI flags ──
    if cli_repo:
        platform, owner, repo = _resolve_from_repo_string(cli_repo, cli_platform)

    # ── 2. Config file ──
    if not platform and config_target_repo:
        platform, owner, repo = _resolve_from_repo_string(config_target_repo)

    # ── 3. Git remote ──
    if not platform:
        result = _detect_from_git_remote()
        if result:
            platform, owner, repo = result

    if not platform or not owner or not repo:
        return None

    # ── Token ──
    token = _resolve_token(platform, config_api_token)
    if not token:
        # Still return the target but without a token — callers can warn
        pass

    return ResolvedTarget(
        platform=platform,
        owner=owner,
        repo=repo,
        api_token=token or "",
    )


# ---------------------------------------------------------------------------
# Minimal HTTP client
# ---------------------------------------------------------------------------


class IssueClient:
    """Synchronous HTTP client for issue CRUD across multiple platforms.

    Only exposes the two operations needed by Community Radar's issue-sync
    feature. Uses ``httpx.Client`` (synchronous) so the pipeline doesn't
    need an event loop.
    """

    def __init__(self, target: ResolvedTarget, timeout: float = 30.0) -> None:
        self._target = target
        self._client = httpx.Client(timeout=timeout)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_issue(
        self,
        title: str,
        body: str,
        labels: list[str] | None = None,
    ) -> dict[str, Any] | None:
        """Create an issue and return the API response dict.

        Returns ``None`` on failure.
        """
        plat = self._target.platform
        payload: dict[str, Any] = {"title": title, "body": body}
        if not plat.create_issue_repo_in_path:
            payload["repo"] = self._target.repo
        if labels and not plat.create_issue_skip_labels:
            if plat.labels_format == "comma":
                payload["labels"] = ",".join(labels)
            else:
                payload["labels"] = labels

        path = plat.create_issue_path.format(
            owner=self._target.owner,
            repo=self._target.repo,
        )

        if plat.create_issue_body_format == "form":
            return self._request("POST", path, data=payload)
        return self._request("POST", path, json=payload)

    def list_issues(
        self,
        *,
        label: str | None = None,
        state: str = "open",
        per_page: int = 100,
        max_pages: int = 50,
    ) -> list[dict[str, Any]]:
        """List issues, optionally filtered by *label*.

        Paginates until a page returns fewer than *per_page* items or
        *max_pages* is reached, so L2 remote dedup sees every issue.
        """
        params: dict[str, Any] = {"state": state, "per_page": per_page}
        if label:
            params["labels"] = label
        path = self._target.platform.list_issues_path.format(
            owner=self._target.owner,
            repo=self._target.repo,
        )
        issues: list[dict[str, Any]] = []
        for page in range(1, max_pages + 1):
            params["page"] = page
            result = self._request("GET", path, params=params)
            if not isinstance(result, list):
                break
            issues.extend(result)
            if len(result) < per_page:
                break
        return issues

    def get_issue(self, issue_number: int | str) -> dict[str, Any] | None:
        """Fetch a single issue by number."""
        result = self._request(
            "GET",
            self._target.platform.get_issue_path.format(
                owner=self._target.owner,
                repo=self._target.repo,
                issue_number=issue_number,
            ),
        )
        if isinstance(result, dict):
            return result
        return None

    def add_labels_to_issue(
        self,
        issue_number: int | str,
        labels: list[str],
    ) -> bool:
        """Add *labels* to an existing issue via the platform's labels endpoint.

        Returns ``True`` if the labels were applied successfully.  Used as a
        follow-up when ``create_issue_skip_labels`` is True — the create
        request omits labels to avoid WAF / permission issues, and this
        call attaches them afterwards via a JSON body.
        """
        plat = self._target.platform
        path = plat.labels_path.format(
            owner=self._target.owner,
            repo=self._target.repo,
            issue_number=issue_number,
        )
        result = self._request("POST", path, json=labels)
        # An empty list means the platform returned 200/201 but did not
        # actually apply any labels (e.g. GitCode silently ignores label
        # additions when the token lacks write permission on the repo).
        return isinstance(result, list) and len(result) > 0

    def close(self) -> None:
        """Release the underlying HTTP client."""
        self._client.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | list[Any] | None = None,
        data: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        """Send an HTTP request and return the parsed JSON body."""
        plat = self._target.platform
        url = f"{plat.default_endpoint}{path}"

        headers: dict[str, str] = {"Accept": plat.accept_header}

        # Auth — three modes (mirrors orchestrator's RepositoryIssueClient)
        if plat.auth_mode == "bearer":
            headers[plat.auth_param] = f"Bearer {self._target.api_token}"
        elif plat.auth_mode in ("access_token", "token"):
            if params is None:
                params = {}
            params[plat.auth_param] = self._target.api_token

        try:
            if json is not None:
                resp = self._client.request(
                    method,
                    url,
                    headers=headers,
                    params=params,
                    json=json,
                )
            elif data is not None:
                resp = self._client.request(
                    method,
                    url,
                    headers=headers,
                    params=params,
                    data=data,
                )
            else:
                resp = self._client.request(
                    method,
                    url,
                    headers=headers,
                    params=params,
                )
            resp.raise_for_status()
            return resp.json()
        except httpx.HTTPStatusError as exc:
            _log.warning(
                "issue client HTTP %d for %s %s: %s",
                exc.response.status_code,
                method,
                url,
                exc.response.text[:500],
            )
            return None
        except Exception as exc:  # noqa: BLE001
            _log.warning("issue client request failed: %s %s: %s", method, url, exc)
            return None

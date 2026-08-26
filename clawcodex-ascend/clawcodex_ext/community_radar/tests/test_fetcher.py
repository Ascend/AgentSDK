"""Tests for clawcodex_ext.community_radar.fetcher.

Release fetching now uses a four-layer fallback:
  L1: GitHub Releases API → L1.5: CHANGELOG raw_body merge
  L2: GitHub Tags API
  L3: GitHub Content API → CHANGELOG raw
  L4: git clone + local file parse (preserved)

Tests mock HTTP responses so no network or git binary is involved.
"""

# pylint: disable=use-implicit-booleaness-not-comparison,redefined-outer-name,reimported
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from clawcodex_ext.community_radar.fetcher import (
    CHANGELOG_CACHE_TTL_SECONDS,
    MAX_RELEASES_PER_SOURCE,
    RATE_LIMIT_WARN_THRESHOLD,
    Fetcher,
    _read_changelog_cache,
    _write_changelog_cache,
)
from clawcodex_ext.community_radar.models import Release, WatchSource


def _source(name: str = "aider") -> WatchSource:
    return WatchSource.from_dict({"name": name, "repo": "foo/bar"})


# ---------------------------------------------------------------------------
# Git path tests (default — no API)
# ---------------------------------------------------------------------------


def test_fetch_releases_via_git_with_precreated_clone_dir(tmp_path: Path) -> None:
    """When the clone dir already exists with a CHANGELOG, git clone is skipped."""
    source = WatchSource.from_dict(
        {
            "name": "testproj",
            "repo": "owner/testproj",
            "changelog_path": "CHANGELOG.md",
        }
    )
    safe = Fetcher._safe_clone_name(source.name)
    clone_dir = tmp_path / "git-clones" / safe
    clone_dir.mkdir(parents=True)
    (clone_dir / "CHANGELOG.md").write_text("## 1.0.0 - 2026-06-01\n### Added\n- Test feature\n", encoding="utf-8")

    fetcher = Fetcher(cache_dir=tmp_path)
    releases = fetcher.fetch_releases(source)
    assert len(releases) == 1
    assert releases[0].tag == "1.0.0"
    fetcher.close()


def test_fetch_releases_no_changelog_returns_empty(tmp_path: Path) -> None:
    """When the clone dir exists but has no changelog file, empty list is returned."""
    source = WatchSource.from_dict(
        {
            "name": "nolog",
            "repo": "owner/nolog",
        }
    )
    safe = Fetcher._safe_clone_name(source.name)
    clone_dir = tmp_path / "git-clones" / safe
    clone_dir.mkdir(parents=True)
    # No CHANGELOG — the dir is empty

    fetcher = Fetcher(cache_dir=tmp_path)
    releases = fetcher.fetch_releases(source)
    assert releases == []
    fetcher.close()


def test_fetch_swallows_errors_into_result(tmp_path: Path) -> None:
    """Errors during fetch_releases are caught and stored in result.errors."""
    source = WatchSource.from_dict(
        {
            "name": "badrepo",
            "repo": "owner/badrepo",
        }
    )
    # No pre-created clone dir → Fetcher will call _git_clone which needs
    # the real git binary and a real remote.  Patch _git_clone to raise.
    with patch.object(Fetcher, "_git_clone", side_effect=RuntimeError("clone failed")):
        fetcher = Fetcher(cache_dir=tmp_path)
        try:
            result = fetcher.fetch(source)
            assert result.errors
            assert "RuntimeError" in result.errors[0]
        finally:
            fetcher.close()


def test_fetch_releases_incremental_triggers_fetch(tmp_path: Path) -> None:
    """incremental=True triggers git fetch on an existing clone dir."""
    source = WatchSource.from_dict(
        {
            "name": "incrproj",
            "repo": "owner/incrproj",
            "changelog_path": "CHANGELOG.md",
        }
    )
    safe = Fetcher._safe_clone_name(source.name)
    clone_dir = tmp_path / "git-clones" / safe
    clone_dir.mkdir(parents=True)
    (clone_dir / "CHANGELOG.md").write_text("## 2.0.0 - 2026-07-01\n### Added\n- Incremental\n", encoding="utf-8")

    # Patch _git_fetch to avoid calling the real git binary.
    with patch.object(Fetcher, "_git_fetch") as mock_fetch:
        fetcher = Fetcher(cache_dir=tmp_path)
        releases = fetcher.fetch_releases(source, incremental=True)
        mock_fetch.assert_called_once()
        assert len(releases) == 1
        assert releases[0].tag == "2.0.0"
        fetcher.close()


# ---------------------------------------------------------------------------
# CHANGELOG parsing tests
# ---------------------------------------------------------------------------


_SAMPLE_KEEP_A_CHANGELOG = """# Changelog

## [2.0.0] - 2026-06-15
### Added
- New telemetry pipeline
- MCP server hot-reload
### Fixed
- Memory leak in agent loop

## [1.5.0] - 2026-05-01
### Added
- Initial MCP support

## 1.0.0 - 2026-04-01
### Added
- First public release
"""


def test_parse_changelog_basic() -> None:
    """Parse a standard Keep-a-Changelog-style document."""
    from clawcodex_ext.community_radar.fetcher import Fetcher  # pylint: disable=redefined-outer-name

    source = WatchSource.from_dict({"name": "test", "repo": "owner/test"})
    releases = Fetcher._parse_changelog(_SAMPLE_KEEP_A_CHANGELOG, source=source)
    assert len(releases) == 3
    assert releases[0].tag == "2.0.0"
    assert releases[0].published_at == "2026-06-15T00:00:00Z"
    assert "telemetry" in releases[0].body
    assert releases[0].is_prerelease is False
    assert releases[1].tag == "1.5.0"
    assert releases[2].tag == "1.0.0"


def test_parse_changelog_without_brackets() -> None:
    """Version headings without [brackets] should also parse."""
    from clawcodex_ext.community_radar.fetcher import Fetcher

    source = WatchSource.from_dict({"name": "test", "repo": "owner/test"})
    text = "## 3.2.1 - 2026-07-01\n### Added\n- Feature A\n"
    releases = Fetcher._parse_changelog(text, source=source)
    assert len(releases) == 1
    assert releases[0].tag == "3.2.1"
    assert releases[0].published_at == "2026-07-01T00:00:00Z"


def test_parse_changelog_prerelease_detection() -> None:
    """-rc, -alpha, -beta suffixes should set is_prerelease=True."""
    from clawcodex_ext.community_radar.fetcher import Fetcher

    source = WatchSource.from_dict({"name": "test", "repo": "owner/test"})
    text = (
        "## 2.0.0-rc1 - 2026-06-01\n### Added\n- RC feature\n"
        "## 2.0.0-beta - 2026-05-15\n### Added\n- Beta feature\n"
        "## 1.9.0-alpha.1 - 2026-05-01\n### Added\n- Alpha feature\n"
    )
    releases = Fetcher._parse_changelog(text, source=source)
    assert len(releases) == 3
    assert releases[0].is_prerelease is True
    assert releases[1].is_prerelease is True
    assert releases[2].is_prerelease is True


def test_parse_changelog_missing_date() -> None:
    """Versions without dates should still parse (published_at=None)."""
    from clawcodex_ext.community_radar.fetcher import Fetcher

    source = WatchSource.from_dict({"name": "test", "repo": "owner/test"})
    text = "## 4.0.0\n### Added\n- No date here\n"
    releases = Fetcher._parse_changelog(text, source=source)
    assert len(releases) == 1
    assert releases[0].tag == "4.0.0"
    assert releases[0].published_at is None


def test_parse_changelog_no_version_headings() -> None:
    """A file with no ## version headings should return empty list."""
    from clawcodex_ext.community_radar.fetcher import Fetcher

    source = WatchSource.from_dict({"name": "test", "repo": "owner/test"})
    text = "# Project\nThis is a readme, not a changelog.\n"
    releases = Fetcher._parse_changelog(text, source=source)
    assert releases == []


def test_read_changelog_tries_multiple_paths(tmp_path: Path) -> None:
    """_read_changelog should try source.changelog_path first, then fallbacks."""
    from clawcodex_ext.community_radar.fetcher import Fetcher

    # Create a RELEASE.md in the clone dir as fallback
    (tmp_path / "RELEASE.md").write_text("# Release notes\n## 1.0.0\ncontent\n", encoding="utf-8")
    source = WatchSource.from_dict({"name": "test", "repo": "owner/test"})
    # source.changelog_path is None, so it will try defaults and find RELEASE.md
    result = Fetcher._read_changelog(tmp_path, source)
    assert result is not None
    assert "Release notes" in result


# ==============================================================================
# V2 tests — four-layer fallback, tag filter, rate limit, CHANGELOG cache
# ==============================================================================

# ── Helpers ───────────────────────────────────────────────────────────────────


def _mock_response(status_code: int = 200, json_data: Any = None, headers: dict[str, str] | None = None) -> MagicMock:
    """Build a minimal mock HTTP response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data or []
    resp.text = "" if status_code != 403 else "API rate limit exceeded"
    resp.headers = headers or {}
    return resp


def _make_client_mock(responses: list[MagicMock]) -> MagicMock:
    """Return a mock httpx client whose ``.request()`` returns each response in turn."""
    client = MagicMock()
    client.request.side_effect = responses
    return client


def _api_release_payload(
    tag: str = "v1.0.0",
    name: str = "1.0.0",
    body: str = "### Added\n- Feature X\n",
    published_at: str = "2026-06-01T12:00:00Z",
    prerelease: bool = False,
) -> dict[str, Any]:
    return {
        "tag_name": tag,
        "name": name,
        "body": body,
        "published_at": published_at,
        "html_url": f"https://github.com/owner/repo/releases/tag/{tag}",
        "prerelease": prerelease,
    }


def _tag_payload(name: str = "v1.0.0") -> dict[str, Any]:
    return {
        "name": name,
        "commit": {
            "sha": "abc123",
            "url": "https://api.github.com/repos/owner/repo/commits/abc123",
        },
        "zipball_url": f"https://api.github.com/repos/owner/repo/zipball/{name}",
        "tarball_url": f"https://api.github.com/repos/owner/repo/tarball/{name}",
    }


# ── _split_changelog_sections ─────────────────────────────────────────────────


def test_split_changelog_sections_basic() -> None:
    """Splits a standard changelog into {version: body} mapping."""
    text = "## [2.0.0] - 2026-06-15\n### Added\n- Feature A\n\n## [1.0.0] - 2026-05-01\n### Added\n- Feature B\n"
    sections = Fetcher._split_changelog_sections(text)
    assert len(sections) == 2
    assert "2.0.0" in sections
    assert "Feature A" in sections["2.0.0"]
    assert "1.0.0" in sections
    assert "Feature B" in sections["1.0.0"]


def test_split_changelog_sections_no_headings() -> None:
    """Returns empty dict when there are no ## headings."""
    sections = Fetcher._split_changelog_sections("# No version headings here\n")
    assert sections == {}


def test_split_changelog_sections_without_brackets() -> None:
    """Handles version headings without brackets."""
    text = "## 3.0.0 - 2026-07-01\n### Changed\n- Breaking change\n"
    sections = Fetcher._split_changelog_sections(text)
    assert "3.0.0" in sections
    assert "Breaking change" in sections["3.0.0"]


# ── _merge_changelog_raw ──────────────────────────────────────────────────────


def test_merge_changelog_raw_matches_by_tag() -> None:
    """raw_body is populated when the release tag matches a CHANGELOG section."""
    releases = [
        Release(
            tag="v2.0.0",
            name="2.0.0",
            body="API body",
            published_at="2026-06-01T12:00:00Z",
            url="https://gh/releases/v2.0.0",
        ),
        Release(
            tag="v1.0.0",
            name="1.0.0",
            body="API body",
            published_at="2026-05-01T12:00:00Z",
            url="https://gh/releases/v1.0.0",
        ),
    ]
    changelog = "## [v2.0.0] - 2026-06-01\n### Added\n- Feature X\n## [v1.0.0] - 2026-05-01\n### Added\n- Feature Y\n"
    result = Fetcher._merge_changelog_raw(releases, changelog)
    assert result[0].raw_body == "### Added\n- Feature X"
    assert result[1].raw_body == "### Added\n- Feature Y"


def test_merge_changelog_raw_no_match_keeps_raw_body_empty() -> None:
    """When no CHANGELOG section matches, raw_body stays empty."""
    releases = [
        Release(
            tag="v99.0.0",
            name="99.0.0",
            body="API body",
            published_at="2026-06-01T12:00:00Z",
            url="https://gh/releases/v99.0.0",
        ),
    ]
    changelog = "## [v1.0.0] - 2026-05-01\n### Added\n- Feature Y\n"
    result = Fetcher._merge_changelog_raw(releases, changelog)
    assert result[0].raw_body == ""


def test_merge_changelog_raw_empty_changelog() -> None:
    """Empty changelog text returns releases unchanged."""
    releases = [
        Release(tag="v1.0.0", name="1.0.0", body="body", published_at=None, url="u"),
    ]
    result = Fetcher._merge_changelog_raw(releases, "")
    assert result == releases


# ── _apply_tag_filter ─────────────────────────────────────────────────────────


def test_apply_tag_filter_prefix_match() -> None:
    """Only releases whose tag starts with the filter are kept."""
    source = WatchSource.from_dict({"name": "test", "repo": "owner/repo", "release_tag_filter": "v"})
    releases = [
        Release(tag="v1.0.0", name="1.0.0", body="", published_at=None, url=""),
        Release(tag="v2.0.0", name="2.0.0", body="", published_at=None, url=""),
        Release(tag="1.0.0", name="1.0.0", body="", published_at=None, url=""),
    ]
    filtered = Fetcher._apply_tag_filter(releases, source)
    assert len(filtered) == 2
    assert all(r.tag.startswith("v") for r in filtered)


def test_apply_tag_filter_no_filter_returns_all() -> None:
    """When release_tag_filter is None/empty, all releases pass through."""
    source = WatchSource.from_dict({"name": "test", "repo": "owner/repo"})
    releases = [
        Release(tag="v1.0.0", name="1.0.0", body="", published_at=None, url=""),
        Release(tag="abc", name="abc", body="", published_at=None, url=""),
    ]
    filtered = Fetcher._apply_tag_filter(releases, source)
    assert filtered == releases


def test_apply_tag_filter_excludes_all() -> None:
    """When filter matches nothing, empty list is returned."""
    source = WatchSource.from_dict({"name": "test", "repo": "owner/repo", "release_tag_filter": "zzz"})
    releases = [
        Release(tag="v1.0.0", name="1.0.0", body="", published_at=None, url=""),
    ]
    filtered = Fetcher._apply_tag_filter(releases, source)
    assert filtered == []


# ── _check_rate_limit ─────────────────────────────────────────────────────────


def test_check_rate_limit_warns_when_low() -> None:
    """Logs a warning when X-RateLimit-Remaining is below threshold."""
    resp = _mock_response(headers={"X-RateLimit-Remaining": str(RATE_LIMIT_WARN_THRESHOLD - 1)})
    with patch("clawcodex_ext.community_radar.fetcher._log.warning") as mock_warn:
        Fetcher._check_rate_limit(resp)
        mock_warn.assert_called_once()
        assert "low" in mock_warn.call_args[0][0].lower()


def test_check_rate_limit_silent_when_ok() -> None:
    """No warning when rate limit is healthy."""
    resp = _mock_response(headers={"X-RateLimit-Remaining": "1000"})
    with patch("clawcodex_ext.community_radar.fetcher._log.warning") as mock_warn:
        Fetcher._check_rate_limit(resp)
        mock_warn.assert_not_called()


def test_check_rate_limit_no_header_is_silent() -> None:
    """No warning when header is missing (e.g. non-GitHub response)."""
    resp = _mock_response()
    with patch("clawcodex_ext.community_radar.fetcher._log.warning") as mock_warn:
        Fetcher._check_rate_limit(resp)
        mock_warn.assert_not_called()


# ── Layer 1: _fetch_releases_api ─────────────────────────────────────────────


def test_layer1_returns_releases_on_200() -> None:
    """On HTTP 200, API releases are mapped to Release objects."""
    source = _source("test")
    payload = [
        _api_release_payload("v1.0.0", "1.0.0", "### Added\n- X\n", "2026-06-01T12:00:00Z"),
        _api_release_payload("v0.9.0", "0.9.0", "### Added\n- Y\n", "2026-05-01T12:00:00Z", prerelease=True),
    ]
    client = _make_client_mock([_mock_response(200, payload)])
    fetcher = Fetcher(cache_dir=Path("/tmp"), client=client)
    releases = fetcher._fetch_releases_api(source)
    assert len(releases) == 2
    assert releases[0].tag == "v1.0.0"
    assert releases[0].body == "### Added\n- X\n"
    assert releases[0].published_at == "2026-06-01T12:00:00Z"
    assert releases[1].is_prerelease is True
    fetcher.close()


def test_layer1_404_returns_empty() -> None:
    """HTTP 404 (repo not found) returns empty list."""
    source = _source("test")
    client = _make_client_mock([_mock_response(404)])
    fetcher = Fetcher(cache_dir=Path("/tmp"), client=client)
    releases = fetcher._fetch_releases_api(source)
    assert releases == []
    fetcher.close()


def test_layer1_401_returns_empty() -> None:
    """HTTP 401 (bad credentials) returns empty list to trigger fallback."""
    source = _source("test")
    client = _make_client_mock([_mock_response(401)])
    fetcher = Fetcher(cache_dir=Path("/tmp"), client=client)
    releases = fetcher._fetch_releases_api(source)
    assert releases == []
    fetcher.close()


def test_layer1_network_error_returns_empty() -> None:
    """Network errors are caught and return empty list."""
    source = _source("test")
    client = MagicMock()
    client.request.side_effect = ConnectionError("no network")
    fetcher = Fetcher(cache_dir=Path("/tmp"), client=client)
    releases = fetcher._fetch_releases_api(source)
    assert releases == []
    fetcher.close()


# ── Layer 2: _fetch_tags_api ──────────────────────────────────────────────────


def test_layer2_returns_tags_on_200() -> None:
    """On HTTP 200, tags are mapped to minimal Release objects."""
    source = _source("test")
    payload = [_tag_payload("v1.0.0"), _tag_payload("v2.0.0-rc1")]
    client = _make_client_mock([_mock_response(200, payload)])
    fetcher = Fetcher(cache_dir=Path("/tmp"), client=client)
    releases = fetcher._fetch_tags_api(source)
    assert len(releases) == 2
    assert releases[0].tag == "v1.0.0"
    assert releases[0].body == ""
    assert releases[0].published_at is None
    assert releases[1].is_prerelease is True  # -rc1
    fetcher.close()


def test_layer2_empty_tags_returns_empty() -> None:
    """Empty tag list returns empty."""
    source = _source("test")
    client = _make_client_mock([_mock_response(200, [])])
    fetcher = Fetcher(cache_dir=Path("/tmp"), client=client)
    releases = fetcher._fetch_tags_api(source)
    assert releases == []
    fetcher.close()


def test_layer2_403_returns_empty() -> None:
    """HTTP 403 returns empty list to trigger fallback."""
    source = _source("test")
    client = _make_client_mock([_mock_response(403)])
    fetcher = Fetcher(cache_dir=Path("/tmp"), client=client)
    releases = fetcher._fetch_tags_api(source)
    assert releases == []
    fetcher.close()


# ── Layer 3: _fetch_changelog_raw ─────────────────────────────────────────────


def test_layer3_returns_decoded_content(tmp_path: Path) -> None:
    """On HTTP 200 with base64 content, decoded text is returned and cached."""
    import base64

    source = _source("test")
    changelog_text = "## 1.0.0\n### Added\n- X\n"
    encoded = base64.b64encode(changelog_text.encode("utf-8")).decode("utf-8")
    payload = {"encoding": "base64", "content": encoded, "name": "CHANGELOG.md"}
    client = _make_client_mock([_mock_response(200, payload)])
    fetcher = Fetcher(cache_dir=tmp_path, client=client)
    result = fetcher._fetch_changelog_raw(source)
    assert result == changelog_text
    # Verify cache was written
    cached = _read_changelog_cache(tmp_path, source, ttl=CHANGELOG_CACHE_TTL_SECONDS)
    assert cached == changelog_text
    fetcher.close()


def test_layer3_uses_cache_on_hit(tmp_path: Path) -> None:
    """When a valid cache entry exists, no HTTP call is made."""
    source = _source("test")
    _write_changelog_cache(tmp_path, source, "cached content")
    client = _make_client_mock([])  # no responses — if HTTP is called it will fail
    fetcher = Fetcher(cache_dir=tmp_path, client=client)
    result = fetcher._fetch_changelog_raw(source)
    assert result == "cached content"
    fetcher.close()


def test_layer3_cache_expired_triggers_api(tmp_path: Path) -> None:
    """When the cache is expired or missing, the API is queried."""
    import base64

    source = _source("test")
    _write_changelog_cache(tmp_path, source, "stale")
    new_text = "## 2.0.0\n### Added\n- Fresh\n"
    encoded = base64.b64encode(new_text.encode("utf-8")).decode("utf-8")
    payload = {"encoding": "base64", "content": encoded}
    client = _make_client_mock([_mock_response(200, payload)])
    fetcher = Fetcher(cache_dir=tmp_path, client=client)
    # Force cache miss by making _read_changelog_cache return None
    with patch("clawcodex_ext.community_radar.fetcher._read_changelog_cache", return_value=None):
        result = fetcher._fetch_changelog_raw(source)
    assert result == new_text
    fetcher.close()


def test_layer3_404_tries_next_path(tmp_path: Path) -> None:
    """When CHANGELOG.md returns 404, the next file path is tried."""
    import base64

    source = _source("test")
    changelog_text = "## 1.0.0\ncontent\n"
    encoded = base64.b64encode(changelog_text.encode("utf-8")).decode("utf-8")
    # First path (CHANGELOG.md) returns 404, second (CHANGELOG) succeeds
    client = _make_client_mock(
        [
            _mock_response(404),  # CHANGELOG.md
            _mock_response(200, {"encoding": "base64", "content": encoded}),  # CHANGELOG
        ]
    )
    fetcher = Fetcher(cache_dir=tmp_path, client=client)
    result = fetcher._fetch_changelog_raw(source)
    assert result == changelog_text
    fetcher.close()


def test_layer3_all_paths_fail_returns_none(tmp_path: Path) -> None:
    """When every file path returns 404, None is returned."""
    source = _source("test")
    # All file paths return 404 (5 paths × 404 = 5 calls)
    responses = [_mock_response(404) for _ in range(6)]
    client = _make_client_mock(responses)
    fetcher = Fetcher(cache_dir=tmp_path, client=client)
    result = fetcher._fetch_changelog_raw(source)
    assert result is None
    fetcher.close()


def test_layer3_respects_custom_changelog_path(tmp_path: Path) -> None:
    """source.changelog_path is tried first."""
    import base64

    source = WatchSource.from_dict({"name": "test", "repo": "owner/repo", "changelog_path": "docs/CHANGES.md"})
    changelog_text = "## 5.0.0\n### Added\n- Custom path\n"
    encoded = base64.b64encode(changelog_text.encode("utf-8")).decode("utf-8")
    client = _make_client_mock(
        [
            _mock_response(200, {"encoding": "base64", "content": encoded}),
        ]
    )
    fetcher = Fetcher(cache_dir=tmp_path, client=client)
    result = fetcher._fetch_changelog_raw(source)
    assert result == changelog_text
    # Verify the custom path was used (first call)
    call_args = client.request.call_args_list[0]
    assert "docs/CHANGES.md" in str(call_args)
    fetcher.close()


# ── fetch_releases() four-layer fallback integration ──────────────────────────


def test_fetch_releases_layer1_succeeds_with_raw_body(tmp_path: Path) -> None:
    """When Layer 1 returns releases and Layer 3 has CHANGELOG, raw_body is merged."""
    import base64

    source = _source("test")
    api_payload = [
        _api_release_payload("v1.0.0", "1.0.0", "API body"),
        _api_release_payload("v0.9.0", "0.9.0", "API body"),
    ]
    changelog_text = (
        "## [v1.0.0] - 2026-06-01\n### Added\n- CHANGELOG detail\n## [v0.9.0] - 2026-05-01\n### Added\n- Old detail\n"
    )
    encoded = base64.b64encode(changelog_text.encode("utf-8")).decode("utf-8")
    # Layer 1 call + Layer 3 call (one path)
    client = _make_client_mock(
        [
            _mock_response(200, api_payload),  # L1
            _mock_response(200, {"encoding": "base64", "content": encoded}),  # L3
        ]
    )
    fetcher = Fetcher(cache_dir=tmp_path, client=client)
    releases = fetcher.fetch_releases(source)
    assert len(releases) == 2
    assert releases[0].tag == "v1.0.0"
    assert releases[0].body == "API body"
    assert releases[0].raw_body == "### Added\n- CHANGELOG detail"
    fetcher.close()


def test_fetch_releases_falls_back_to_layer2(tmp_path: Path) -> None:
    """When Layer 1 returns empty (no GitHub Releases), Layer 2 tags are used."""
    source = _source("test")
    payload = [_tag_payload("v1.0.0"), _tag_payload("v2.0.0")]
    client = _make_client_mock(
        [
            _mock_response(200, []),  # L1: empty
            _mock_response(200, payload),  # L2: tags
        ]
    )
    fetcher = Fetcher(cache_dir=tmp_path, client=client)
    releases = fetcher.fetch_releases(source)
    assert len(releases) == 2
    assert releases[0].tag == "v1.0.0"
    assert releases[0].body == ""
    fetcher.close()


def test_fetch_releases_falls_back_to_layer3(tmp_path: Path) -> None:
    """When L1 and L2 fail, L3 (CHANGELOG API) is used."""
    import base64

    source = _source("test")
    changelog_text = "## [1.0.0] - 2026-06-01\n### Added\n- From CHANGELOG\n"
    encoded = base64.b64encode(changelog_text.encode("utf-8")).decode("utf-8")
    # L1 + L2 fail, L3 succeeds
    client = _make_client_mock(
        [
            _mock_response(200, []),  # L1: empty
            _mock_response(200, []),  # L2: empty
            _mock_response(200, {"encoding": "base64", "content": encoded}),  # L3: success
        ]
    )
    fetcher = Fetcher(cache_dir=tmp_path, client=client)
    releases = fetcher.fetch_releases(source)
    assert len(releases) == 1
    assert releases[0].tag == "1.0.0"
    assert "From CHANGELOG" in releases[0].body
    fetcher.close()


def test_fetch_releases_falls_back_to_layer4(tmp_path: Path) -> None:
    """When L1/L2/L3 all fail, L4 (git clone) is used as last resort."""
    source = WatchSource.from_dict(
        {
            "name": "clonefallback",
            "repo": "owner/clonefallback",
            "changelog_path": "CHANGELOG.md",
        }
    )
    # Pre-create clone dir so _git_clone is skipped
    safe = Fetcher._safe_clone_name(source.name)
    clone_dir = tmp_path / "git-clones" / safe
    clone_dir.mkdir(parents=True)
    (clone_dir / "CHANGELOG.md").write_text(
        "## 3.0.0 - 2026-07-01\n### Added\n- Clone fallback feature\n", encoding="utf-8"
    )
    # L1, L2, L3 all fail (empty/error)
    client = _make_client_mock(
        [
            _mock_response(200, []),  # L1
            _mock_response(200, []),  # L2
            _mock_response(404),  # L3: CHANGELOG.md
            _mock_response(404),  # L3: CHANGELOG
            _mock_response(404),  # L3: RELEASE.md
            _mock_response(404),  # L3: RELEASE_NOTES.md
            _mock_response(404),  # L3: History.md
        ]
    )
    fetcher = Fetcher(cache_dir=tmp_path, client=client)
    releases = fetcher.fetch_releases(source)
    assert len(releases) == 1
    assert releases[0].tag == "3.0.0"
    assert "Clone fallback" in releases[0].body
    fetcher.close()


def test_fetch_releases_tag_filter_applied_in_all_layers(tmp_path: Path) -> None:
    """release_tag_filter filters results regardless of which layer wins."""
    source = WatchSource.from_dict({"name": "test", "repo": "owner/repo", "release_tag_filter": "v"})
    payload = [
        _api_release_payload("v1.0.0", "1.0.0"),
        _api_release_payload("1.0.0", "1.0.0"),  # won't match filter
        _api_release_payload("v2.0.0", "2.0.0"),
    ]
    client = _make_client_mock([_mock_response(200, payload)])
    fetcher = Fetcher(cache_dir=tmp_path, client=client)
    releases = fetcher.fetch_releases(source)
    assert len(releases) == 2
    assert all(r.tag.startswith("v") for r in releases)
    fetcher.close()


def test_fetch_releases_capped_at_max(tmp_path: Path) -> None:
    """Results are capped at MAX_RELEASES_PER_SOURCE regardless of API response size."""
    source = _source("test")
    # Return more releases than the cap
    payload = [_api_release_payload(f"v{i}.0.0", f"{i}.0.0") for i in range(50)]
    client = _make_client_mock([_mock_response(200, payload)])
    fetcher = Fetcher(cache_dir=tmp_path, client=client)
    releases = fetcher.fetch_releases(source)
    assert len(releases) == MAX_RELEASES_PER_SOURCE
    fetcher.close()


# ── CHANGELOG cache module-level helpers ──────────────────────────────────────


def test_changelog_cache_write_and_read(tmp_path: Path) -> None:
    """Cache roundtrip: write then read returns the same text."""
    source = _source("cached-proj")
    _write_changelog_cache(tmp_path, source, "test changelog content")
    result = _read_changelog_cache(tmp_path, source)
    assert result == "test changelog content"


def test_changelog_cache_miss(tmp_path: Path) -> None:
    """Reading a non-existent cache returns None."""
    source = _source("no-cache")
    result = _read_changelog_cache(tmp_path, source)
    assert result is None


def test_changelog_cache_expired(tmp_path: Path) -> None:
    """An expired cache entry returns None."""
    source = _source("expired")
    _write_changelog_cache(tmp_path, source, "stale")
    # Read with TTL=0 (immediately expired)
    result = _read_changelog_cache(tmp_path, source, ttl=0)
    assert result is None


# ── Release.raw_body serialisation roundtrip ──────────────────────────────────


def test_release_raw_body_roundtrip() -> None:
    """raw_body survives to_dict() → from_dict() roundtrip."""
    r = Release(
        tag="v1.0.0",
        name="1.0.0",
        body="API body",
        published_at="2026-06-01T12:00:00Z",
        url="https://gh/releases/v1.0.0",
        raw_body="### Added\n- CHANGELOG detail",
    )
    d = r.to_dict()
    assert d["raw_body"] == "### Added\n- CHANGELOG detail"
    r2 = Release.from_dict(d)
    assert r2.raw_body == "### Added\n- CHANGELOG detail"


def test_release_raw_body_defaults_to_empty() -> None:
    """When raw_body is absent from dict, it defaults to ''."""
    r = Release.from_dict({"tag": "v1.0.0", "name": "1.0.0", "body": "b", "url": "u"})
    assert r.raw_body == ""

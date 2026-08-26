# pylint: disable=use-implicit-booleaness-not-comparison
"""Tests for issue_sync module — cache, dedup, body rendering, candidate selection."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest import mock

from clawcodex_ext.community_radar.issue_platforms import (
    IssueClient,
    IssuePlatform,
    ResolvedTarget,
)
from clawcodex_ext.community_radar.issue_sync import (
    _ID_MARKER_RE,
    IssueSyncCache,
    IssueSyncResult,
    _bar_chart,
    _build_issue_body,
    _confirm_duplicate_override,
    _fetch_remote_feature_map,
    _select_candidates,
    list_candidates_interactive,
    sync_features_to_issues,
    sync_single_feature,
)
from clawcodex_ext.community_radar.models import (
    CommunityDigest,
    DigestStats,
    FeatureCategory,
    FeatureRecord,
    FeatureScore,
    FeatureType,
    ScoredFeature,
)


def _make_plat(name: str = "gitcode") -> IssuePlatform:
    return IssuePlatform(
        name=name,
        default_endpoint="https://api.example.com/api/v5",
        web_host="https://example.com",
        auth_mode="access_token",
        auth_param="access_token",
        accept_header="application/json",
        token_env_vars=(f"{name.upper()}_TOKEN",),
    )


def _make_target(name: str = "gitcode", owner: str = "o", repo: str = "r", token: str = "t") -> ResolvedTarget:
    return ResolvedTarget(platform=_make_plat(name), owner=owner, repo=repo, api_token=token)


def _make_record(feature_id: str = "abc123", source: str = "test/repo", title: str = "Test Feature") -> FeatureRecord:
    return FeatureRecord(
        id=feature_id,
        source=source,
        title=title,
        description="A test feature for unit tests.",
        category=FeatureCategory.AGENT_LOOP,
        feature_type=FeatureType.NEW,
        url="https://example.com/feature",
        related_projects=["proj-a", "proj-b"],
    )


def _make_score(record_id: str = "abc123") -> FeatureScore:
    return FeatureScore(
        record_id=record_id,
        overall=85.0,
        popularity=70.0,
        maturity=80.0,
        adaptation_cost=90.0,
        strategic_value=88.0,
        architecture_fit=82.0,
    )


def _make_scored_feature(feature_id: str = "abc123", title: str = "Test Feature") -> ScoredFeature:
    return ScoredFeature(record=_make_record(feature_id, title=title), score=_make_score(feature_id))


def _make_digest(**kwargs) -> CommunityDigest:
    defaults = dict(  # noqa: C408
        period="weekly",
        generated_at="2026-07-16T08:00:00Z",
        summary="Test digest.",
        period_start="2026-07-09T08:00:00Z",
        stats=DigestStats(total_versions=10, total_features=5),
        sources_used=["test/repo"],
    )
    defaults.update(kwargs)
    return CommunityDigest(**defaults)


def _make_interactive_digest_data() -> dict:
    return {
        "period": "weekly",
        "generated_at": "2026-07-16T08:00:00Z",
        "summary": "test",
        "highlights": [
            {
                "record": {
                    "id": "feat1",
                    "source": "test/repo",
                    "title": "Feature One",
                    "description": "desc",
                    "category": "agent_loop",
                    "feature_type": "new",
                    "url": "http://x",
                    "related_projects": [],
                    "tags": [],
                },
                "score": {
                    "record_id": "feat1",
                    "overall": 88.0,
                    "popularity": 70.0,
                    "maturity": 80.0,
                    "adaptation_cost": 90.0,
                    "strategic_value": 85.0,
                    "architecture_fit": 80.0,
                },
            },
            {
                "record": {
                    "id": "feat2",
                    "source": "other/repo",
                    "title": "Feature Two",
                    "description": "desc2",
                    "category": "tool_system",
                    "feature_type": "new",
                    "url": "http://y",
                    "related_projects": [],
                    "tags": [],
                },
                "score": {
                    "record_id": "feat2",
                    "overall": 72.0,
                    "popularity": 60.0,
                    "maturity": 65.0,
                    "adaptation_cost": 75.0,
                    "strategic_value": 70.0,
                    "architecture_fit": 68.0,
                },
            },
        ],
        "llm_importance": [
            {"feature_id": "feat1", "level": "MAJOR", "highlight": "big"},
            {"feature_id": "feat2", "level": "MINOR", "highlight": "small"},
        ],
        "stats": {"total_versions": 5, "total_features": 2},
        "sources_used": ["test/repo"],
        "errors": [],
    }


class TestIdMarkerRe:
    def test_extracts_feature_id(self) -> None:
        m = _ID_MARKER_RE.search("<!-- community-radar-id: abc123def456 -->")
        assert m is not None
        assert m.group(1) == "abc123def456"

    def test_allows_extra_whitespace(self) -> None:
        m = _ID_MARKER_RE.search("<!--   community-radar-id:   xyz789   -->")
        assert m is not None
        assert m.group(1) == "xyz789"

    def test_match_embedded_in_body(self) -> None:
        body = "Some text\n<!-- community-radar-id: fff111 -->\nMore text"
        m = _ID_MARKER_RE.search(body)
        assert m is not None
        assert m.group(1) == "fff111"


class TestBarChart:
    def test_zero_and_full(self) -> None:
        assert _bar_chart(0) == "░" * 20
        assert _bar_chart(100) == "█" * 20

    def test_half_and_custom_width(self) -> None:
        result = _bar_chart(50)
        assert result.count("█") == 10
        assert result.count("░") == 10
        result = _bar_chart(50, width=10)
        assert result == "█" * 5 + "░" * 5


class TestBuildIssueBody:
    def test_contains_title(self) -> None:
        sf = _make_scored_feature(title="Agent Protocol")
        digest = _make_digest()
        body = _build_issue_body(sf, digest, {})
        assert "Agent Protocol" in body

    def test_contains_feature_id_marker(self) -> None:
        sf = _make_scored_feature(feature_id="abc123")
        digest = _make_digest()
        body = _build_issue_body(sf, digest, {})
        assert "<!-- community-radar-id: abc123 -->" in body

    def test_contains_score_sections(self) -> None:
        sf = _make_scored_feature()
        digest = _make_digest()
        body = _build_issue_body(sf, digest, {})
        assert "评分明细" in body
        assert "决策清单" in body
        assert "详细分析" in body

    def test_contains_llm_info_when_provided(self) -> None:
        sf = _make_scored_feature()
        digest = _make_digest()
        llm_info = {"highlight": "A breakthrough feature.", "title_zh": "智能协议"}
        body = _build_issue_body(sf, digest, llm_info)
        assert "A breakthrough feature." in body
        assert "智能协议" in body

    def test_duplicate_warning_included(self) -> None:
        sf = _make_scored_feature()
        digest = _make_digest()
        body = _build_issue_body(sf, digest, {}, duplicate_warning="already exists as #42")
        assert "already exists as #42" in body

    def test_no_duplicate_warning_when_empty(self) -> None:
        sf = _make_scored_feature()
        digest = _make_digest()
        body = _build_issue_body(sf, digest, {})
        assert "⚠️ 重复提醒" not in body

    def test_bar_chart_included(self) -> None:
        sf = _make_scored_feature()
        digest = _make_digest()
        body = _build_issue_body(sf, digest, {})
        assert "█" in body
        assert "░" in body
        assert "流行度" in body


class TestSelectCandidates:
    def test_filters_major_only(self) -> None:
        major = _make_scored_feature("id1", "Major Feature")
        major.score.overall = 90.0
        minor = _make_scored_feature("id2", "Minor Feature")
        minor.score.overall = 60.0
        digest = _make_digest(highlights=[major, minor])
        llm = {
            "id1": {"level": "MAJOR", "highlight": "big deal"},
            "id2": {"level": "MINOR", "highlight": "small deal"},
        }
        result = _select_candidates(digest, llm, max_n=5)
        assert len(result) == 1
        assert result[0].record.id == "id1"

    def test_sorts_by_score_desc(self) -> None:
        a = _make_scored_feature("a", "A")
        a.score.overall = 70.0
        b = _make_scored_feature("b", "B")
        b.score.overall = 95.0
        c = _make_scored_feature("c", "C")
        c.score.overall = 80.0
        digest = _make_digest(highlights=[a, b, c])
        llm = {fid: {"level": "MAJOR"} for fid in ("a", "b", "c")}
        result = _select_candidates(digest, llm, max_n=5)
        assert [r.record.id for r in result] == ["b", "c", "a"]

    def test_respects_max_n(self) -> None:
        features = []
        llm = {}
        for i in range(5):
            fid = f"id{i}"
            sf = _make_scored_feature(fid, f"Feature {i}")
            sf.score.overall = 80.0 - i
            features.append(sf)
            llm[fid] = {"level": "MAJOR"}
        digest = _make_digest(highlights=features)
        result = _select_candidates(digest, llm, max_n=2)
        assert len(result) == 2

    def test_returns_empty_when_no_major(self) -> None:
        sf = _make_scored_feature("id1", "Only Minor")
        digest = _make_digest(highlights=[sf])
        llm = {"id1": {"level": "MINOR"}}
        result = _select_candidates(digest, llm, max_n=5)
        assert result == []

    def test_returns_empty_when_no_highlights(self) -> None:
        digest = _make_digest(highlights=[])
        result = _select_candidates(digest, {}, max_n=5)
        assert result == []


class TestIssueSyncCache:
    def test_new_cache_is_empty(self) -> None:
        cache = IssueSyncCache()
        assert cache.repos == {}

    def test_load_nonexistent_file(self) -> None:
        cache = IssueSyncCache.load(Path("/nonexistent/path/cache.json"))
        assert cache.repos == {}

    def test_save_and_load_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cache.json"
            cache = IssueSyncCache()
            target = _make_target()
            cache.put(target, "feat1", {"issue_number": 42, "state": "open"})
            cache.save(path)
            cache2 = IssueSyncCache.load(path)
            entry = cache2.get(target, "feat1")
            assert entry is not None
            assert entry["issue_number"] == 42

    def test_load_corrupted_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "cache.json"
            path.write_text("not valid json {{{")
            cache = IssueSyncCache.load(path)
            assert cache.repos == {}

    def test_exists_returns_true_for_open(self) -> None:
        cache = IssueSyncCache()
        target = _make_target()
        cache.put(target, "feat1", {"state": "open"})
        assert cache.exists(target, "feat1") is True

    def test_repo_key_isolation(self) -> None:
        cache = IssueSyncCache()
        t1 = _make_target(owner="o1", repo="r1")
        t2 = _make_target(owner="o2", repo="r2")
        cache.put(t1, "feat1", {"state": "open"})
        assert cache.exists(t1, "feat1") is True
        assert cache.exists(t2, "feat1") is False

    def test_sync_from_remote_adds_new_entries(self) -> None:
        cache = IssueSyncCache()
        target = _make_target()
        remote_map = {
            "feat1": {"feature_id": "feat1", "issue_number": 1, "state": "open"},
            "feat2": {"feature_id": "feat2", "issue_number": 2, "state": "open"},
        }
        added = cache.sync_from_remote(target, remote_map)
        assert added == 2
        assert cache.exists(target, "feat1") is True
        assert cache.exists(target, "feat2") is True

    def test_sync_from_remote_updates_existing(self) -> None:
        cache = IssueSyncCache()
        target = _make_target()
        cache.put(target, "feat1", {"issue_number": 1, "state": "open"})
        remote_map = {"feat1": {"feature_id": "feat1", "issue_number": 1, "state": "closed"}}
        added = cache.sync_from_remote(target, remote_map)
        assert added == 0
        assert cache.exists(target, "feat1") is False


class TestFetchRemoteFeatureMap:
    def test_extracts_markers_from_issues(self) -> None:
        plat = IssuePlatform(
            name="test",
            default_endpoint="https://api.example.com",
            web_host="https://example.com",
            auth_mode="access_token",
            auth_param="access_token",
            token_env_vars=(),
        )
        _target = ResolvedTarget(platform=plat, owner="o", repo="r", api_token="t")

        client = mock.MagicMock(spec=IssueClient)
        client.list_issues.return_value = [
            {
                "number": 1,
                "title": "Issue 1",
                "body": "Some text\n<!-- community-radar-id: feat001 -->\nMore",
                "html_url": "https://example.com/o/r/issues/1",
                "state": "open",
            },
            {
                "number": 2,
                "title": "Issue 2",
                "body": "No marker here.",
                "html_url": "https://example.com/o/r/issues/2",
                "state": "closed",
            },
            {
                "number": 3,
                "title": "Issue 3",
                "body": "<!-- community-radar-id: feat002 -->",
                "html_url": "https://example.com/o/r/issues/3",
                "state": "open",
            },
        ]

        mapping = _fetch_remote_feature_map(client)
        assert len(mapping) == 2
        assert "feat001" in mapping
        assert "feat002" in mapping
        assert mapping["feat001"]["issue_number"] == 1
        assert mapping["feat002"]["issue_number"] == 3

    def test_handles_client_error(self) -> None:
        client = mock.MagicMock(spec=IssueClient)
        client.list_issues.side_effect = Exception("network error")
        mapping = _fetch_remote_feature_map(client)
        assert mapping == {}


class TestConfirmDuplicateOverride:
    def test_yes_returns_true(self) -> None:
        with mock.patch("builtins.input", return_value="y"):
            assert _confirm_duplicate_override("f1", "Title", {"issue_url": "http://x"}) is True

    def test_no_returns_false(self) -> None:
        with mock.patch("builtins.input", return_value="n"):
            assert _confirm_duplicate_override("f1", "Title", {"issue_url": "http://x"}) is False


class TestIssueSyncResult:
    def test_defaults_empty(self) -> None:
        r = IssueSyncResult()
        assert r.created == []
        assert r.skipped == []
        assert r.warned == []
        assert r.errors == []


class TestSyncFeaturesToIssues:
    def test_no_target_returns_error(self) -> None:
        with mock.patch("clawcodex_ext.community_radar.issue_sync.resolve_target", return_value=None):
            result = sync_features_to_issues(
                digest=_make_digest(),
                llm_importance={},
                config=mock.MagicMock(),
                target=None,
            )
        assert len(result.errors) == 1
        assert "无法确定目标仓库" in result.errors[0]

    def test_no_token_returns_error(self) -> None:
        target = _make_target(token="")
        result = sync_features_to_issues(
            digest=_make_digest(),
            llm_importance={},
            config=mock.MagicMock(),
            target=target,
        )
        assert len(result.errors) == 1
        assert "API token" in result.errors[0]

    def test_no_major_candidates_skips(self) -> None:
        target = _make_target()
        digest = _make_digest(highlights=[])
        result = sync_features_to_issues(
            digest=digest,
            llm_importance={},
            config=mock.MagicMock(sync_issues_max_per_scan=2),
            target=target,
        )
        assert len(result.skipped) == 1

    def test_l1_cache_hit_skips(self) -> None:
        target = _make_target()
        sf = _make_scored_feature("feat1", "Test")
        sf.record.category = FeatureCategory.AGENT_LOOP
        digest = _make_digest(highlights=[sf])
        llm = {"feat1": {"level": "MAJOR"}}

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "issue_sync_cache.json"
            cache = IssueSyncCache()
            cache.put(
                target, "feat1", {"feature_id": "feat1", "issue_number": 1, "state": "open", "issue_url": "http://x"}
            )
            cache.save(cache_path)

            result = sync_features_to_issues(
                digest=digest,
                llm_importance=llm,
                config=mock.MagicMock(sync_issues_max_per_scan=2, sync_issues_labels=["community-radar"]),
                target=target,
                cache_dir=tmpdir,
            )
            assert len(result.skipped) >= 1
            created_ids = [c.get("feature_id") for c in result.created]
            assert "feat1" not in created_ids

    def test_creates_issue_on_success(self) -> None:
        target = _make_target()
        sf = _make_scored_feature("feat1", "Test Feature")
        sf.record.category = FeatureCategory.AGENT_LOOP
        digest = _make_digest(highlights=[sf])
        llm = {"feat1": {"level": "MAJOR"}}

        with tempfile.TemporaryDirectory() as tmpdir:  # noqa: SIM117
            with mock.patch("clawcodex_ext.community_radar.issue_sync.IssueClient") as mock_client_cls:
                mock_client = mock.MagicMock()
                mock_client.create_issue.return_value = {
                    "number": 42,
                    "html_url": "https://example.com/o/r/issues/42",
                }
                mock_client.list_issues.return_value = []
                mock_client_cls.return_value = mock_client

                result = sync_features_to_issues(
                    digest=digest,
                    llm_importance=llm,
                    config=mock.MagicMock(sync_issues_max_per_scan=2, sync_issues_labels=["community-radar"]),
                    target=target,
                    cache_dir=tmpdir,
                )
                assert len(result.created) == 1
                assert result.created[0]["issue_number"] == 42
                assert result.created[0]["feature_id"] == "feat1"

    def test_handles_create_failure(self) -> None:
        target = _make_target()
        sf = _make_scored_feature("feat1", "Test")
        sf.record.category = FeatureCategory.AGENT_LOOP
        digest = _make_digest(highlights=[sf])
        llm = {"feat1": {"level": "MAJOR"}}

        with tempfile.TemporaryDirectory() as tmpdir:  # noqa: SIM117
            with mock.patch("clawcodex_ext.community_radar.issue_sync.IssueClient") as mock_client_cls:
                mock_client = mock.MagicMock()
                mock_client.create_issue.return_value = None  # API failure
                mock_client.list_issues.return_value = []
                mock_client_cls.return_value = mock_client

                result = sync_features_to_issues(
                    digest=digest,
                    llm_importance=llm,
                    config=mock.MagicMock(sync_issues_max_per_scan=2, sync_issues_labels=["community-radar"]),
                    target=target,
                    cache_dir=tmpdir,
                )
                assert len(result.errors) >= 1

    def test_target_from_config(self) -> None:
        with mock.patch("clawcodex_ext.community_radar.issue_sync.resolve_target") as mock_resolve:
            mock_resolve.return_value = None  # Can't resolve → error
            config = mock.MagicMock()
            config.target_repo = "myowner/myrepo"
            config.api_token = ""
            result = sync_features_to_issues(
                digest=_make_digest(),
                llm_importance={},
                config=config,
                target=None,
            )
            mock_resolve.assert_called_once()
            assert len(result.errors) >= 1

    def test_skip_mode_skips_closed(self) -> None:
        target = _make_target()
        sf = _make_scored_feature("feat1", "Test")
        sf.record.category = FeatureCategory.AGENT_LOOP
        digest = _make_digest(highlights=[sf])
        llm = {"feat1": {"level": "MAJOR"}}

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "issue_sync_cache.json"
            cache = IssueSyncCache()
            cache.put(
                target,
                "feat1",
                {
                    "feature_id": "feat1",
                    "issue_number": 5,
                    "state": "closed",
                    "issue_url": "http://x",
                },
            )
            cache.save(cache_path)

            result = sync_features_to_issues(
                digest=digest,
                llm_importance=llm,
                config=mock.MagicMock(
                    sync_issues_max_per_scan=2,
                    sync_issues_labels=["community-radar"],
                ),
                target=target,
                cache_dir=tmpdir,
                closed_issue_mode="skip",
            )
            assert len(result.skipped) >= 1
            assert any("closed issue exists" in str(s.get("reason", "")) for s in result.skipped)
            assert len(result.created) == 0

    def test_retry_mode_recreates_closed(self) -> None:
        target = _make_target()
        sf = _make_scored_feature("feat1", "Test Feature")
        sf.record.category = FeatureCategory.AGENT_LOOP
        digest = _make_digest(highlights=[sf])
        llm = {"feat1": {"level": "MAJOR"}}

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "issue_sync_cache.json"
            cache = IssueSyncCache()
            cache.put(
                target,
                "feat1",
                {
                    "feature_id": "feat1",
                    "issue_number": 5,
                    "state": "closed",
                    "issue_url": "http://x",
                },
            )
            cache.save(cache_path)

            with mock.patch("clawcodex_ext.community_radar.issue_sync.IssueClient") as mock_client_cls:
                mock_client = mock.MagicMock()
                mock_client.create_issue.return_value = {
                    "number": 99,
                    "html_url": "https://example.com/o/r/issues/99",
                }
                mock_client.list_issues.return_value = []
                mock_client_cls.return_value = mock_client

                result = sync_features_to_issues(
                    digest=digest,
                    llm_importance=llm,
                    config=mock.MagicMock(
                        sync_issues_max_per_scan=2,
                        sync_issues_labels=["community-radar"],
                    ),
                    target=target,
                    cache_dir=tmpdir,
                    closed_issue_mode="retry",
                )
                assert len(result.created) == 1
                assert result.created[0]["feature_id"] == "feat1"

    def test_ask_mode_prompts_on_closed(self) -> None:
        target = _make_target()
        sf = _make_scored_feature("feat1", "Test Feature")
        sf.record.category = FeatureCategory.AGENT_LOOP
        digest = _make_digest(highlights=[sf])
        llm = {"feat1": {"level": "MAJOR"}}

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "issue_sync_cache.json"
            cache = IssueSyncCache()
            cache.put(
                target,
                "feat1",
                {
                    "feature_id": "feat1",
                    "issue_number": 5,
                    "state": "closed",
                    "issue_url": "http://x/issues/5",
                },
            )
            cache.save(cache_path)

            with mock.patch("clawcodex_ext.community_radar.issue_sync.IssueClient") as mock_client_cls:
                mock_client = mock.MagicMock()
                mock_client.create_issue.return_value = {
                    "number": 99,
                    "html_url": "https://example.com/o/r/issues/99",
                }
                mock_client.list_issues.return_value = []
                mock_client_cls.return_value = mock_client

                with mock.patch("builtins.input", return_value="y"):
                    with mock.patch("sys.stdin.isatty", return_value=True):
                        result = sync_features_to_issues(
                            digest=digest,
                            llm_importance=llm,
                            config=mock.MagicMock(
                                sync_issues_max_per_scan=2,
                                sync_issues_labels=["community-radar"],
                            ),
                            target=target,
                            cache_dir=tmpdir,
                            closed_issue_mode="ask",
                        )
                assert len(result.created) == 1
                assert result.created[0]["feature_id"] == "feat1"

    def test_ask_mode_tty_fallback_skips(self) -> None:
        target = _make_target()
        sf = _make_scored_feature("feat1", "Test Feature")
        sf.record.category = FeatureCategory.AGENT_LOOP
        digest = _make_digest(highlights=[sf])
        llm = {"feat1": {"level": "MAJOR"}}

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "issue_sync_cache.json"
            cache = IssueSyncCache()
            cache.put(
                target,
                "feat1",
                {
                    "feature_id": "feat1",
                    "issue_number": 5,
                    "state": "closed",
                    "issue_url": "http://x/issues/5",
                },
            )
            cache.save(cache_path)

            with mock.patch("sys.stdin.isatty", return_value=False):
                result = sync_features_to_issues(
                    digest=digest,
                    llm_importance=llm,
                    config=mock.MagicMock(
                        sync_issues_max_per_scan=2,
                        sync_issues_labels=["community-radar"],
                    ),
                    target=target,
                    cache_dir=tmpdir,
                    closed_issue_mode="ask",
                )
            assert len(result.created) == 0
            assert len(result.skipped) >= 1

    def test_respects_max_n_with_backfill_and_skips(self) -> None:
        target = _make_target()
        feat1 = _make_scored_feature("feat1", "Feature 1")
        feat1.record.category = FeatureCategory.AGENT_LOOP
        feat1.score.overall = 90.0
        feat2 = _make_scored_feature("feat2", "Feature 2")
        feat2.record.category = FeatureCategory.AGENT_LOOP
        feat2.score.overall = 85.0
        feat3 = _make_scored_feature("feat3", "Feature 3")
        feat3.record.category = FeatureCategory.AGENT_LOOP
        feat3.score.overall = 80.0

        digest = _make_digest(highlights=[feat1, feat2, feat3])
        llm = {f"feat{i}": {"level": "MAJOR"} for i in range(1, 4)}

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "issue_sync_cache.json"
            cache = IssueSyncCache()
            cache.put(
                target,
                "feat1",
                {
                    "feature_id": "feat1",
                    "issue_number": 1,
                    "state": "closed",
                    "issue_url": "http://x/1",
                },
            )
            cache.save(cache_path)

            with mock.patch("clawcodex_ext.community_radar.issue_sync.IssueClient") as mock_client_cls:
                mock_client = mock.MagicMock()
                mock_client.create_issue.return_value = {
                    "number": 99,
                    "html_url": "https://example.com/o/r/issues/99",
                }
                mock_client.list_issues.return_value = []
                mock_client_cls.return_value = mock_client

                result = sync_features_to_issues(
                    digest=digest,
                    llm_importance=llm,
                    config=mock.MagicMock(
                        sync_issues_max_per_scan=2,
                        sync_issues_labels=["community-radar"],
                    ),
                    target=target,
                    cache_dir=tmpdir,
                    closed_issue_mode="skip",
                )
                assert len(result.created) == 2
                created_ids = [c["feature_id"] for c in result.created]
                assert "feat1" not in created_ids
                assert "feat2" in created_ids
                assert "feat3" in created_ids


class TestSyncSingleFeature:
    def test_no_target_returns_error(self) -> None:
        with mock.patch("clawcodex_ext.community_radar.issue_sync.resolve_target", return_value=None):
            result = sync_single_feature(
                feature_id="feat1",
                config=mock.MagicMock(),
                target=None,
            )
        assert len(result.errors) >= 1

    def test_feature_not_found_in_digest(self) -> None:
        target = _make_target()
        digest = _make_digest(highlights=[])
        result = sync_single_feature(
            feature_id="nonexistent",
            config=mock.MagicMock(),
            digest=digest,
            target=target,
        )
        assert len(result.errors) >= 1
        assert "未找到" in result.errors[0]

    def test_l3_duplicate_cancelled(self) -> None:
        target = _make_target()
        sf = _make_scored_feature("feat1", "Test")
        sf.record.category = FeatureCategory.AGENT_LOOP
        digest = _make_digest(highlights=[sf])

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "issue_sync_cache.json"
            cache = IssueSyncCache()
            cache.put(
                target,
                "feat1",
                {
                    "feature_id": "feat1",
                    "issue_number": 1,
                    "state": "open",
                    "issue_url": "http://x",
                    "created_at": "2026-01-01",
                },
            )
            cache.save(cache_path)

            with mock.patch("clawcodex_ext.community_radar.issue_sync._confirm_duplicate_override", return_value=False):  # noqa: SIM117
                with mock.patch("clawcodex_ext.community_radar.issue_sync.IssueClient") as mock_cls:
                    mock_client = mock.MagicMock()
                    mock_client.list_issues.return_value = []
                    mock_cls.return_value = mock_client

                    result = sync_single_feature(
                        feature_id="feat1",
                        config=mock.MagicMock(sync_issues_labels=["community-radar"]),
                        digest=digest,
                        target=target,
                        cache_dir=tmpdir,
                    )
                    assert len(result.warned) >= 1
                    assert "user cancelled after duplicate warning" in str(result.warned[0].get("action", ""))

    def test_creates_issue_after_l3_approval(self) -> None:
        target = _make_target()
        sf = _make_scored_feature("feat1", "Test Feature")
        sf.record.category = FeatureCategory.AGENT_LOOP
        digest = _make_digest(highlights=[sf])

        with tempfile.TemporaryDirectory() as tmpdir:  # noqa: SIM117
            with mock.patch("clawcodex_ext.community_radar.issue_sync.IssueClient") as mock_cls:
                mock_client = mock.MagicMock()
                mock_client.create_issue.return_value = {
                    "number": 99,
                    "html_url": "https://example.com/o/r/issues/99",
                }
                mock_client.list_issues.return_value = []
                mock_cls.return_value = mock_client

                result = sync_single_feature(
                    feature_id="feat1",
                    config=mock.MagicMock(sync_issues_labels=["community-radar"]),
                    digest=digest,
                    llm_importance={"feat1": {"level": "MAJOR"}},
                    target=target,
                    cache_dir=tmpdir,
                )
                assert len(result.created) == 1
                assert result.created[0]["issue_number"] == 99

    def test_finds_feature_in_trending_when_not_in_highlights(self) -> None:
        target = _make_target()
        sf = _make_scored_feature("feat1", "Trending Feature")
        sf.record.category = FeatureCategory.AGENT_LOOP
        digest = _make_digest(highlights=[], trending=[sf])

        with tempfile.TemporaryDirectory() as tmpdir:  # noqa: SIM117
            with mock.patch("clawcodex_ext.community_radar.issue_sync.IssueClient") as mock_cls:
                mock_client = mock.MagicMock()
                mock_client.create_issue.return_value = {
                    "number": 7,
                    "html_url": "https://example.com/o/r/issues/7",
                }
                mock_client.list_issues.return_value = []
                mock_cls.return_value = mock_client

                result = sync_single_feature(
                    feature_id="feat1",
                    config=mock.MagicMock(sync_issues_labels=["community-radar"]),
                    digest=digest,
                    target=target,
                    cache_dir=tmpdir,
                )
                assert len(result.created) == 1


class TestListCandidatesInteractive:
    def test_no_output_dir(self) -> None:
        config = mock.MagicMock()
        config.output_dir = "/nonexistent/path/xyz"
        result = list_candidates_interactive(config=config)
        assert result is None

    def test_no_json_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = mock.MagicMock()
            config.output_dir = tmpdir
            result = list_candidates_interactive(config=config)
            assert result is None

    def test_renders_candidates_and_selects_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            digest_data = _make_interactive_digest_data()
            json_path = Path(tmpdir) / "community-digest-weekly-20260716T080000Z.json"
            json_path.write_text(json.dumps(digest_data))

            config = mock.MagicMock()
            config.output_dir = tmpdir

            with mock.patch("builtins.input", return_value="1"):
                result = list_candidates_interactive(config=config, cache_dir=tmpdir)
                assert result is not None
                assert len(result) == 1
                assert result[0]["feature_id"] == "feat1"

    def test_quit_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            digest_data = _make_interactive_digest_data()
            json_path = Path(tmpdir) / "digest.json"
            json_path.write_text(json.dumps(digest_data))

            config = mock.MagicMock()
            config.output_dir = tmpdir

            with mock.patch("builtins.input", return_value="q"):
                result = list_candidates_interactive(config=config, cache_dir=tmpdir)
                assert result is None

    def test_no_major_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            digest_data = _make_interactive_digest_data()
            json_path = Path(tmpdir) / "digest.json"
            json_path.write_text(json.dumps(digest_data))

            config = mock.MagicMock()
            config.output_dir = tmpdir

            with mock.patch("builtins.input", return_value="q"):
                result = list_candidates_interactive(config=config, cache_dir=tmpdir)
            assert result is None

    def test_handles_corrupted_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "digest.json"
            json_path.write_text("not valid {{{")

            config = mock.MagicMock()
            config.output_dir = tmpdir

            result = list_candidates_interactive(config=config, cache_dir=tmpdir)
            assert result is None

    def test_invalid_input_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            digest_data = _make_interactive_digest_data()
            json_path = Path(tmpdir) / "digest.json"
            json_path.write_text(json.dumps(digest_data))

            config = mock.MagicMock()
            config.output_dir = tmpdir

            with mock.patch("builtins.input", return_value="xyz_invalid"):
                result = list_candidates_interactive(config=config, cache_dir=tmpdir)
                assert result is None

    def test_eof_returns_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            digest_data = _make_interactive_digest_data()
            json_path = Path(tmpdir) / "digest.json"
            json_path.write_text(json.dumps(digest_data))

            config = mock.MagicMock()
            config.output_dir = tmpdir

            with mock.patch("builtins.input", side_effect=EOFError):
                result = list_candidates_interactive(config=config, cache_dir=tmpdir)
                assert result is None

# pylint: disable=use-implicit-booleaness-not-comparison
"""Tests for clawcodex_ext.community_radar.notifier (Phase 4)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from clawcodex_ext.community_radar.config import RadarConfig
from clawcodex_ext.community_radar.models import (
    CommunityDigest,
    DigestStats,
    FeatureCategory,
    FeatureRecord,
    FeatureScore,
    FeatureType,
    ScoredFeature,
    utc_now_iso,
)
from clawcodex_ext.community_radar.notifier import (
    NOTIFY_CONFIG_RELATIVE_PATH,
    DigestNotifier,
    _load_notify_config,
    build_digest_message,
)
from clawcodex_ext.community_radar.reporter import DigestWriteResult


def _record() -> FeatureRecord:
    return FeatureRecord(
        id="r1",
        source="aider",
        title="Auto lint fix",
        description="Adds automatic lint fixing",
        category=FeatureCategory.TOOL_SYSTEM,
        feature_type=FeatureType.NEW,
        released_at="2026-06-15T00:00:00Z",
    )


def _digest(
    *,
    trending: list[FeatureRecord] | None = None,
    breaking: list[FeatureRecord] | None = None,
    period: str = "weekly",
) -> CommunityDigest:
    # NB: ``trending if trending is not None else [_record()]`` — never
    # ``trending or [_record()]`` because that would silently swallow an
    # explicit empty list (which several tests rely on).
    trending = list(trending if trending is not None else [_record()])
    breaking = list(breaking or [])
    scored = [
        ScoredFeature(
            record=r,
            score=FeatureScore(
                record_id=r.id,
                overall=75.0,
                popularity=80.0,
                maturity=70.0,
                adaptation_cost=65.0,
                strategic_value=80.0,
                architecture_fit=70.0,
            ),
        )
        for r in trending
    ]
    return CommunityDigest(
        period=period,
        generated_at=utc_now_iso(),
        summary="本期重点：tool_system。",
        new_features=trending,
        trending=scored,
        breaking_changes=breaking,
        stats=DigestStats(
            total_versions=1,
            total_features=len(trending),
            by_category={"tool_system": max(len(trending), 1)},
        ),
        sources_used=["aider"],
    )


# ---------------------------------------------------------------------------
# _load_notify_config
# ---------------------------------------------------------------------------


def test_load_notify_config_missing_returns_empty(tmp_path: Path) -> None:
    cfg = _load_notify_config(tmp_path / "missing.yaml")
    assert cfg.channels == []


def test_load_notify_config_yaml(tmp_path: Path) -> None:
    cfg_path = tmp_path / "notify.yaml"
    cfg_path.write_text(
        "channels:\n  - name: feishu-team\n    type: feishu\n    webhook_url: https://open.feishu.cn/x\n",
        encoding="utf-8",
    )
    cfg = _load_notify_config(cfg_path)
    assert len(cfg.channels) == 1
    assert cfg.channels[0]["name"] == "feishu-team"
    assert cfg.channels[0]["type"] == "feishu"


def test_load_notify_config_json(tmp_path: Path) -> None:
    cfg_path = tmp_path / "notify.json"
    cfg_path.write_text(
        json.dumps({"channels": [{"name": "slack-dev", "type": "slack"}]}),
        encoding="utf-8",
    )
    cfg = _load_notify_config(cfg_path)
    assert len(cfg.channels) == 1
    assert cfg.channels[0]["type"] == "slack"


def test_load_notify_config_filters_invalid_entries(tmp_path: Path) -> None:
    cfg_path = tmp_path / "notify.yaml"
    cfg_path.write_text(
        "channels:\n  - not-a-dict\n  - name: ok\n    type: discord\n",
        encoding="utf-8",
    )
    cfg = _load_notify_config(cfg_path)
    assert len(cfg.channels) == 1
    assert cfg.channels[0]["name"] == "ok"


def test_load_notify_config_path_constant_is_relative() -> None:
    assert ".clawcodex" in NOTIFY_CONFIG_RELATIVE_PATH.parts


# ---------------------------------------------------------------------------
# DigestNotifier — disabled / no config
# ---------------------------------------------------------------------------


def test_notifier_returns_empty_when_notify_disabled() -> None:
    notifier = DigestNotifier(RadarConfig(notify=False))
    result = notifier.broadcast(_digest())
    assert result == {}


def test_notifier_returns_empty_when_no_channels(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CLAWCODEX_NOTIFY_CONFIG", str(tmp_path / "absent.yaml"))
    notifier = DigestNotifier(RadarConfig(notify=True))
    result = notifier.broadcast(_digest())
    assert result == {}


# ---------------------------------------------------------------------------
# DigestNotifier — channels module unavailable
# ---------------------------------------------------------------------------


def test_notifier_handles_missing_channels_module(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CLAWCODEX_NOTIFY_CONFIG", str(tmp_path / "nope.yaml"))
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "channels:\n  - name: x\n    type: feishu\n    webhook_url: u\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "clawcodex_ext.community_radar.notifier._load_channels_module",
        lambda: None,
    )
    notifier = DigestNotifier(
        RadarConfig(notify=True),
        notify_config_path=cfg_path,
    )
    result = notifier.broadcast(_digest())
    assert result == {"_error": False}


# ---------------------------------------------------------------------------
# DigestNotifier — manager_factory seam
# ---------------------------------------------------------------------------


class _FakeChannelMessage:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _FakeMessageLevel:
    SUCCESS = "success"
    INFO = "info"


class _FakeModels:
    ChannelMessage = _FakeChannelMessage
    MessageLevel = _FakeMessageLevel


class _FakeChannel:
    def __init__(self, config: Any) -> None:
        self.config = config


class _FakeBase:
    ChannelManager = type(  # type: ignore[assignment]
        "ChannelManager",
        (),
        {"__init__": lambda self: None, "register": lambda self, ch: None},
    )


class _FakeChannelManager:
    def __init__(self) -> None:
        self.broadcast_calls: list[Any] = []

    async def broadcast(self, message: Any) -> dict[str, bool]:
        self.broadcast_calls.append(message)
        return {"feishu-team": True}


def _fake_channels_module() -> dict[str, Any]:
    return {"base": _FakeBase, "models": _FakeModels}


def test_notifier_uses_manager_factory(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CLAWCODEX_NOTIFY_CONFIG", str(tmp_path / "nope.yaml"))
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "channels:\n  - name: feishu-team\n    type: feishu\n    webhook_url: u\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "clawcodex_ext.community_radar.notifier._load_channels_module",
        _fake_channels_module,
    )
    manager = _FakeChannelManager()
    notifier = DigestNotifier(
        RadarConfig(notify=True),
        notify_config_path=cfg_path,
        manager_factory=lambda: manager,
    )
    result = notifier.broadcast(_digest())
    assert result == {"feishu-team": True}
    assert manager.broadcast_calls, "broadcast was invoked"
    msg = manager.broadcast_calls[0]
    assert msg.kwargs["title"].startswith("ClawCodex 社区动态")
    assert msg.kwargs["level"] == _FakeMessageLevel.SUCCESS


def test_notifier_swallows_manager_factory_exception(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CLAWCODEX_NOTIFY_CONFIG", str(tmp_path / "nope.yaml"))
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "channels:\n  - name: feishu-team\n    type: feishu\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "clawcodex_ext.community_radar.notifier._load_channels_module",
        _fake_channels_module,
    )
    notifier = DigestNotifier(
        RadarConfig(notify=True),
        notify_config_path=cfg_path,
        manager_factory=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    # Should not raise; returns empty dict because manager was None.
    result = notifier.broadcast(_digest())
    assert result == {}


def test_notifier_handles_async_runtimeerror(tmp_path: Path, monkeypatch) -> None:
    """asyncio.get_event_loop() may raise in some threads; ensure we fall back."""

    class _LoopBlowUpManager(_FakeChannelManager):
        pass

    monkeypatch.setenv("CLAWCODEX_NOTIFY_CONFIG", str(tmp_path / "nope.yaml"))
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "channels:\n  - name: feishu-team\n    type: feishu\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "clawcodex_ext.community_radar.notifier._load_channels_module",
        _fake_channels_module,
    )
    manager = _LoopBlowUpManager()

    def _boom() -> None:
        raise RuntimeError("no event loop")

    monkeypatch.setattr(
        "clawcodex_ext.community_radar.notifier.asyncio.get_event_loop",
        _boom,
    )
    notifier = DigestNotifier(
        RadarConfig(notify=True),
        notify_config_path=cfg_path,
        manager_factory=lambda: manager,
    )
    result = notifier.broadcast(_digest())
    assert result == {"feishu-team": True}


# ---------------------------------------------------------------------------
# build_digest_message
# ---------------------------------------------------------------------------


def test_build_digest_message_requires_channels(monkeypatch) -> None:
    monkeypatch.setattr(
        "clawcodex_ext.community_radar.notifier._load_channels_module",
        lambda: None,
    )
    with pytest.raises(RuntimeError):
        build_digest_message(_digest())


def test_build_digest_message_uses_fake_models(monkeypatch) -> None:
    monkeypatch.setattr(
        "clawcodex_ext.community_radar.notifier._load_channels_module",
        _fake_channels_module,
    )
    digest = _digest(period="monthly")
    msg = build_digest_message(digest, top_n=3)
    assert isinstance(msg, _FakeChannelMessage)
    assert "月报" in msg.kwargs["title"]
    assert msg.kwargs["level"] == _FakeMessageLevel.SUCCESS
    assert "Auto lint fix" in msg.kwargs["text"]


def test_build_digest_message_includes_metadata(monkeypatch) -> None:
    monkeypatch.setattr(
        "clawcodex_ext.community_radar.notifier._load_channels_module",
        _fake_channels_module,
    )
    write_result = DigestWriteResult(
        markdown_path=Path("/tmp/report.md"),
        json_path=Path("/tmp/report.json"),
        proposals_path=Path("/tmp/report.proposals.json"),
    )
    msg = build_digest_message(_digest(), write_result)
    meta = msg.kwargs["metadata"]
    assert meta["digest_markdown"] == "/tmp/report.md"
    assert meta["digest_json"] == "/tmp/report.json"
    assert meta["proposals_json"] == "/tmp/report.proposals.json"
    assert meta["period"] == "weekly"


def test_build_digest_message_info_level_when_empty(monkeypatch) -> None:
    monkeypatch.setattr(
        "clawcodex_ext.community_radar.notifier._load_channels_module",
        _fake_channels_module,
    )
    digest = _digest(trending=[], breaking=[])
    msg = build_digest_message(digest)
    assert msg.kwargs["level"] == _FakeMessageLevel.INFO


def test_build_digest_message_truncates_long_body(monkeypatch) -> None:
    monkeypatch.setattr(
        "clawcodex_ext.community_radar.notifier._load_channels_module",
        _fake_channels_module,
    )
    record = _record()
    record.title = "X" * 50
    record.description = "Y" * 50
    digest = _digest(trending=[record])
    msg = build_digest_message(digest, top_n=1)
    # Hard cap is 30 000; we leave headroom at 8 000.
    assert len(msg.kwargs["text"]) <= 8000


def test_build_digest_message_breaking_changes(monkeypatch) -> None:
    monkeypatch.setattr(
        "clawcodex_ext.community_radar.notifier._load_channels_module",
        _fake_channels_module,
    )
    breaking = FeatureRecord(
        id="r2",
        source="langgraph",
        title="StateGraph refactor",
        description="breaking",
        category=FeatureCategory.AGENT_LOOP,
        feature_type=FeatureType.BREAKING,
    )
    digest = _digest(breaking=[breaking])
    msg = build_digest_message(digest)
    assert "破坏性变更" in msg.kwargs["text"]
    assert "StateGraph refactor" in msg.kwargs["text"]

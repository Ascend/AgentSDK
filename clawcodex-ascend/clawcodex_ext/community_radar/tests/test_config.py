"""Tests for clawcodex_ext.community_radar.config."""

from __future__ import annotations

from pathlib import Path

from clawcodex_ext.community_radar.config import (
    DEFAULT_ROADMAP_KEYWORDS,
    DEFAULT_WEIGHTS,
    RadarConfig,
    apply_env_overrides,
    default_config_path,
)


def test_defaults_phase3_opt_out() -> None:
    cfg = RadarConfig()
    # Phase 3 change: defaults flipped so a fresh install is functional.
    assert cfg.enabled is True
    assert cfg.notify is True
    assert cfg.use_llm is False  # pylint: disable=no-member
    assert cfg.max_features_per_report == 30


def test_normalized_weights_sum_to_one() -> None:
    cfg = RadarConfig()
    weights = cfg.normalized_weights()
    assert abs(sum(weights.values()) - 1.0) < 1e-9
    for key in DEFAULT_WEIGHTS:
        assert key in weights


def test_normalized_weights_recover_from_zero_total() -> None:
    cfg = RadarConfig(weights={k: 0 for k in DEFAULT_WEIGHTS})
    weights = cfg.normalized_weights()
    assert abs(sum(weights.values()) - 1.0) < 1e-9


def test_apply_env_overrides(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CLAWCODEX_RADAR_ENABLED", "1")
    monkeypatch.setenv("CLAWCODEX_RADAR_CRON", "0 9 * * 1")
    monkeypatch.setenv("CLAWCODEX_RADAR_MAX", "42")
    monkeypatch.setenv("CLAWCODEX_RADAR_NOTIFY", "true")
    monkeypatch.setenv("CLAWCODEX_RADAR_OUTPUT", str(tmp_path / "out"))
    monkeypatch.setenv("CLAWCODEX_RADAR_WEIGHT_POPULARITY", "0.30")
    cfg = RadarConfig()
    cfg = apply_env_overrides(cfg)
    assert cfg.enabled is True
    assert cfg.cron_schedule == "0 9 * * 1"
    assert cfg.max_features_per_report == 42
    assert cfg.notify is True
    assert cfg.output_dir == str(tmp_path / "out")
    assert cfg.weights["popularity"] == 0.30


def test_apply_env_overrides_can_opt_out(monkeypatch) -> None:
    """Phase 3: default is enabled; env var must be able to disable it."""
    monkeypatch.setenv("CLAWCODEX_RADAR_ENABLED", "0")
    monkeypatch.setenv("CLAWCODEX_RADAR_NOTIFY", "false")
    cfg = apply_env_overrides(RadarConfig())
    assert cfg.enabled is False
    assert cfg.notify is False


def test_from_dict_unknown_keys_ignored() -> None:
    cfg = RadarConfig.from_dict({"enabled": True, "ignored_key": "boom", "weights": {"popularity": "NaN"}})
    assert cfg.enabled is True
    # NaN string falls back to default for the popularity dim.
    assert isinstance(cfg.weights["popularity"], (int, float))


def test_default_config_path_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CLAWCODEX_HOME", str(tmp_path))
    assert default_config_path() == tmp_path / "community-radar" / "config.yaml"


def test_default_roadmap_keywords_not_empty() -> None:
    assert DEFAULT_ROADMAP_KEYWORDS
    assert all(isinstance(k, str) for k in DEFAULT_ROADMAP_KEYWORDS)

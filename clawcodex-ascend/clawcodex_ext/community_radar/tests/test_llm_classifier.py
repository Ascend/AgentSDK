# pylint: disable=use-implicit-booleaness-not-comparison
"""Tests for clawcodex_ext.community_radar.llm_classifier (Phase 2)."""

from __future__ import annotations

import json
from typing import Any

import pytest

from clawcodex_ext.community_radar.llm_classifier import (
    LLMConfig,
    _extract_json,
    build_classifier_hook,
    build_extractor_hook,
    build_summarizer_hook,
    llm_generated_marker,
    reset_litellm_module,
    set_litellm_module,
)
from clawcodex_ext.community_radar.models import (
    FeatureCategory,
    FeatureRecord,
    FeatureType,
)

# ---------------------------------------------------------------------------
# _extract_json
# ---------------------------------------------------------------------------


def test_extract_json_plain_object() -> None:
    assert _extract_json('{"category": "tool_system"}') == {"category": "tool_system"}


def test_extract_json_fenced_object() -> None:
    text = '```json\n{"category": "agent_runtime"}\n```'
    assert _extract_json(text) == {"category": "agent_runtime"}


def test_extract_json_fenced_no_lang() -> None:
    text = '```\n{"category": "memory"}\n```'
    assert _extract_json(text) == {"category": "memory"}


def test_extract_json_fenced_array() -> None:
    text = '```json\n[{"title": "x"}]\n```'
    assert _extract_json(text) == [{"title": "x"}]


def test_extract_json_with_prose_around() -> None:
    text = 'Here you go:\n{"a": 1}\nThanks!'
    assert _extract_json(text) == {"a": 1}


def test_extract_json_returns_none_on_garbage() -> None:
    assert _extract_json("not json at all") is None
    assert _extract_json("") is None
    assert _extract_json(None) is None  # type: ignore[arg-type]


def test_extract_json_recovers_trailing_garbage() -> None:
    # Closing brace is missing, but a } is somewhere in the string.
    text = '{"x": 1, "y": 2} more stuff'
    assert _extract_json(text) == {"x": 1, "y": 2}


# ---------------------------------------------------------------------------
# LLMConfig
# ---------------------------------------------------------------------------


def test_llmconfig_from_env_requires_model(monkeypatch) -> None:
    monkeypatch.delenv("CLAWCODEX_RADAR_LLM_MODEL", raising=False)
    monkeypatch.delenv("CLAWCODEX_RADAR_LLM", raising=False)
    with pytest.raises(ValueError):
        LLMConfig.from_env()


def test_llmconfig_from_env_reads_overrides(monkeypatch) -> None:
    monkeypatch.setenv("CLAWCODEX_RADAR_LLM_MODEL", "gpt-4o-mini")
    monkeypatch.setenv("CLAWCODEX_RADAR_LLM_API_KEY", "sk-test")
    monkeypatch.setenv("CLAWCODEX_RADAR_LLM_API_BASE", "https://api.example.com/v1")
    monkeypatch.setenv("CLAWCODEX_RADAR_LLM_TIMEOUT", "12.5")
    cfg = LLMConfig.from_env()
    assert cfg.model == "gpt-4o-mini"
    assert cfg.api_key == "sk-test"
    assert cfg.api_base == "https://api.example.com/v1"
    assert cfg.timeout_seconds == 12.5


def test_llmconfig_from_env_falls_back_to_openai_key(monkeypatch) -> None:
    monkeypatch.setenv("CLAWCODEX_RADAR_LLM_MODEL", "openai/gpt-4o")
    monkeypatch.delenv("CLAWCODEX_RADAR_LLM_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    cfg = LLMConfig.from_env()
    assert cfg.api_key == "sk-openai"


def test_llm_generated_marker_is_labeled() -> None:
    marker = llm_generated_marker()
    assert "LLM-assisted" in marker
    assert "T" in marker  # ISO timestamp separator


# ---------------------------------------------------------------------------
# build_classifier_hook
# ---------------------------------------------------------------------------


class _FakeLiteLLM:
    def __init__(self, response: Any) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []

    def completion(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self._response


def _assistant_message(text: str) -> dict[str, Any]:
    return {"choices": [{"message": {"content": text}}]}


def test_classifier_hook_parses_valid_response() -> None:
    fake = _FakeLiteLLM(_assistant_message(json.dumps({"category": "tool_system"})))
    set_litellm_module(fake)
    try:
        hook = build_classifier_hook(LLMConfig(model="x"))
        record = FeatureRecord(
            id="r1",
            source="aider",
            title="X",
            description="Y",
            category=FeatureCategory.UNKNOWN,
            feature_type=FeatureType.NEW,
        )
        assert hook(record) == FeatureCategory.TOOL_SYSTEM
        assert fake.calls, "completion was invoked"
        prompt = fake.calls[0]["messages"][0]["content"]
        assert "tool_system" in prompt  # category list included
    finally:
        reset_litellm_module()


def test_classifier_hook_handles_unknown_category() -> None:
    fake = _FakeLiteLLM(_assistant_message(json.dumps({"category": "nonsense"})))
    set_litellm_module(fake)
    try:
        hook = build_classifier_hook(LLMConfig(model="x"))
        record = FeatureRecord(
            id="r1",
            source="aider",
            title="X",
            description="Y",
            category=FeatureCategory.UNKNOWN,
            feature_type=FeatureType.NEW,
        )
        assert hook(record) == FeatureCategory.UNKNOWN
    finally:
        reset_litellm_module()


def test_classifier_hook_returns_unknown_on_bad_json() -> None:
    fake = _FakeLiteLLM(_assistant_message("not json"))
    set_litellm_module(fake)
    try:
        hook = build_classifier_hook(LLMConfig(model="x"))
        record = FeatureRecord(
            id="r1",
            source="aider",
            title="X",
            description="Y",
            category=FeatureCategory.UNKNOWN,
            feature_type=FeatureType.NEW,
        )
        assert hook(record) == FeatureCategory.UNKNOWN
    finally:
        reset_litellm_module()


def test_classifier_hook_swallows_completion_error() -> None:
    class _BoomLiteLLM:
        def completion(self, **kwargs: Any) -> Any:
            raise RuntimeError("rate limited")

    set_litellm_module(_BoomLiteLLM())
    try:
        hook = build_classifier_hook(LLMConfig(model="x"))
        record = FeatureRecord(
            id="r1",
            source="aider",
            title="X",
            description="Y",
            category=FeatureCategory.UNKNOWN,
            feature_type=FeatureType.NEW,
        )
        assert hook(record) == FeatureCategory.UNKNOWN
    finally:
        reset_litellm_module()


# ---------------------------------------------------------------------------
# build_extractor_hook
# ---------------------------------------------------------------------------


def test_extractor_hook_refines_records() -> None:
    payload = json.dumps(
        [
            {"title": "Refined A", "description": "new A"},
            {"title": "Refined B", "description": "new B"},
        ]
    )
    fake = _FakeLiteLLM(_assistant_message(f"```json\n{payload}\n```"))
    set_litellm_module(fake)
    try:
        hook = build_extractor_hook(LLMConfig(model="x"), max_keep=3)
        template = FeatureRecord(
            id="r1",
            source="aider",
            title="old",
            description="old desc",
            category=FeatureCategory.TOOL_SYSTEM,
            feature_type=FeatureType.NEW,
        )
        out = hook([template], body="release body")
        assert [r.title for r in out] == ["Refined A", "Refined B"]
        # Template metadata is preserved.
        assert all(r.source == "aider" for r in out)
        assert all(r.category == FeatureCategory.TOOL_SYSTEM for r in out)
    finally:
        reset_litellm_module()


def test_extractor_hook_passthrough_when_empty() -> None:
    fake = _FakeLiteLLM(_assistant_message("[]"))
    set_litellm_module(fake)
    try:
        hook = build_extractor_hook(LLMConfig(model="x"))
        assert hook([], body="") == []
        assert fake.calls == []  # never invoked for empty input
    finally:
        reset_litellm_module()


def test_extractor_hook_falls_back_on_bad_payload() -> None:
    fake = _FakeLiteLLM(_assistant_message("nope"))
    set_litellm_module(fake)
    try:
        hook = build_extractor_hook(LLMConfig(model="x"))
        template = FeatureRecord(
            id="r1",
            source="aider",
            title="keep",
            description="keep desc",
            category=FeatureCategory.TOOL_SYSTEM,
            feature_type=FeatureType.NEW,
        )
        out = hook([template], body="x")
        # On failure, the original records are preserved (not empty).
        assert out == [template]
    finally:
        reset_litellm_module()


# ---------------------------------------------------------------------------
# build_summarizer_hook
# ---------------------------------------------------------------------------


def test_summarizer_hook_returns_text() -> None:
    fake = _FakeLiteLLM(_assistant_message("本期重点：tool_system。"))
    set_litellm_module(fake)
    try:
        hook = build_summarizer_hook(LLMConfig(model="x"))
        record = FeatureRecord(
            id="r1",
            source="aider",
            title="X",
            description="Y",
            category=FeatureCategory.TOOL_SYSTEM,
            feature_type=FeatureType.NEW,
        )
        summary = hook([record], ["aider"], [])
        assert summary == "本期重点：tool_system。"
    finally:
        reset_litellm_module()


def test_summarizer_hook_returns_empty_on_failure() -> None:
    class _BoomLiteLLM:
        def completion(self, **kwargs: Any) -> Any:
            raise RuntimeError("network down")

    set_litellm_module(_BoomLiteLLM())
    try:
        hook = build_summarizer_hook(LLMConfig(model="x"))
        record = FeatureRecord(
            id="r1",
            source="aider",
            title="X",
            description="Y",
            category=FeatureCategory.TOOL_SYSTEM,
            feature_type=FeatureType.NEW,
        )
        assert hook([record], ["aider"], []) == ""
    finally:
        reset_litellm_module()

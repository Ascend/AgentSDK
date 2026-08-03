"""Tests for the F-72 capability constants module."""

from __future__ import annotations

from src.providers.native.capabilities import (
    CAP_AUDIO_INPUT,
    CAP_GROUNDING,
    CAP_LONG_CONTEXT,
    CAP_REASONING,
    CAP_SAFETY_SETTINGS,
    CAP_STREAMING_TOOLS,
    CAP_STRUCTURED_OUTPUT,
    CAP_TTS,
    CAP_VISION,
    CAPABILITY_DESCRIPTIONS,
)


def test_capability_constants_are_distinct_strings() -> None:
    """Each CAP_* constant must be a unique non-empty string."""
    values = {
        CAP_STRUCTURED_OUTPUT,
        CAP_STREAMING_TOOLS,
        CAP_VISION,
        CAP_SAFETY_SETTINGS,
        CAP_GROUNDING,
        CAP_TTS,
        CAP_AUDIO_INPUT,
        CAP_LONG_CONTEXT,
        CAP_REASONING,
    }
    assert len(values) == 9
    for v in values:
        assert isinstance(v, str)
        assert v


def test_capability_descriptions_cover_all_constants() -> None:
    """Every CAP_* constant must have a human-readable description."""
    for cap in (
        CAP_STRUCTURED_OUTPUT,
        CAP_STREAMING_TOOLS,
        CAP_VISION,
        CAP_SAFETY_SETTINGS,
        CAP_GROUNDING,
        CAP_TTS,
        CAP_AUDIO_INPUT,
        CAP_LONG_CONTEXT,
        CAP_REASONING,
    ):
        assert cap in CAPABILITY_DESCRIPTIONS
        assert isinstance(CAPABILITY_DESCRIPTIONS[cap], str)
        assert CAPABILITY_DESCRIPTIONS[cap]


def test_capability_strings_use_snake_case() -> None:
    """Capability identifiers should be snake_case to match the
    F-72 plan's naming convention (``structured_output``,
    ``streaming_tools``, ...).
    """
    for cap in (
        CAP_STRUCTURED_OUTPUT,
        CAP_STREAMING_TOOLS,
        CAP_VISION,
        CAP_SAFETY_SETTINGS,
        CAP_GROUNDING,
        CAP_TTS,
        CAP_AUDIO_INPUT,
        CAP_LONG_CONTEXT,
        CAP_REASONING,
    ):
        assert cap == cap.lower(), f"{cap} is not lowercase"
        assert " " not in cap, f"{cap} contains whitespace"

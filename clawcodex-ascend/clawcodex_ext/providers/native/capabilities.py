"""Platform-specific capability markers for native LLM providers.

Each native adapter advertises the set of platform-exclusive features it
supports via the ``capabilities`` class attribute on
:class:`~clawcodex_ext.providers.native.base.NativeProvider`. Downstream code can
then query :meth:`NativeProvider.check_capabilities` to decide whether a
given request needs to fall back to a provider that supports the
required feature (or surface a clear error to the user).

The constants are plain string identifiers — they describe a *capability
family*, not a specific provider API. New capabilities should be added
here so that all adapters and consumers reference the same names.
"""

from __future__ import annotations

# Structured output via JSON Schema (OpenAI ``response_format`` /
# Gemini ``response_schema`` / Grok ``response_format``).
CAP_STRUCTURED_OUTPUT = "structured_output"

# Streaming function-calling — emit ``tool_calls`` deltas mid-stream
# rather than at the end of the response. OpenAI's modern SDK supports
# this; some compat backends do not.
CAP_STREAMING_TOOLS = "streaming_tools"

# Native image / vision understanding (``image_url`` on OpenAI / Grok,
# ``inline_data`` on Gemini).
CAP_VISION = "vision"

# Platform safety filters (Gemini ``SafetySetting``).
CAP_SAFETY_SETTINGS = "safety_settings"

# Grounded generation with web/PDF search (Gemini ``google_search`` /
# ``google_search_retrieval`` tools).
CAP_GROUNDING = "grounding"

# Text-to-speech synthesis (Gemini TTS, OpenAI ``audio.speech``).
CAP_TTS = "tts"

# Audio input — speech-to-text or audio understanding
# (Gemini ``audio`` parts, OpenAI ``input_audio``).
CAP_AUDIO_INPUT = "audio_input"

# Long-context million-token window (matches the upstream
# ``strip_1m_context_suffix`` opt-in).
CAP_LONG_CONTEXT = "long_context"

# Explicit reasoning / thinking traces (DeepSeek-R1, o1-series, etc.).
CAP_REASONING = "reasoning"

# Image generation — text-to-image / image-to-image (DALL-E, Stable
# Diffusion, Flux, Agnes Image, etc.). Providers that exclusively serve
# image generation (no chat interface) advertise this capability.
CAP_IMAGE_GENERATION = "image_generation"

# Video generation — text-to-video / image-to-video (Runway, Pika, Sora,
# Agnes Video, etc.). Typically follows an async task pattern (create →
# poll → retrieve).
CAP_VIDEO_GENERATION = "video_generation"


#: Mapping of capability → human-readable description, for help text and
#: the visualizer's provider capability panel.
CAPABILITY_DESCRIPTIONS: dict[str, str] = {
    CAP_STRUCTURED_OUTPUT: "JSON-schema constrained decoding",
    CAP_STREAMING_TOOLS: "Tool calls emitted mid-stream",
    CAP_VISION: "Image understanding",
    CAP_SAFETY_SETTINGS: "Provider safety filter configuration",
    CAP_GROUNDING: "Web/PDF retrieval-augmented generation",
    CAP_TTS: "Text-to-speech synthesis",
    CAP_AUDIO_INPUT: "Audio input understanding",
    CAP_LONG_CONTEXT: "Million-token context window",
    CAP_REASONING: "Explicit reasoning/thinking traces",
    CAP_IMAGE_GENERATION: "Image generation (text-to-image / image-to-image)",
    CAP_VIDEO_GENERATION: "Video generation (text-to-video / image-to-video)",
}


__all__ = [
    "CAP_STRUCTURED_OUTPUT",
    "CAP_STREAMING_TOOLS",
    "CAP_VISION",
    "CAP_SAFETY_SETTINGS",
    "CAP_GROUNDING",
    "CAP_TTS",
    "CAP_AUDIO_INPUT",
    "CAP_LONG_CONTEXT",
    "CAP_REASONING",
    "CAP_IMAGE_GENERATION",
    "CAP_VIDEO_GENERATION",
    "CAPABILITY_DESCRIPTIONS",
]

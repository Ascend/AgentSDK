# pylint: disable=relative-beyond-top-level
"""LLM-assisted classification for SR-5.1 (Phase 2).

Implements the LLM-assisted classification hook:

> 2. LLM assist: when rule matching fails or confidence is low, call the LLM
>    to extract the feature category from the release body

The LLM is **opt-in** — callers must enable it via ``RadarConfig.use_llm``
and provide at least one ``llm_model`` string (any value accepted by
`litellm.completion` works: ``"gpt-4o-mini"``, ``"claude-3-5-haiku-20241022"``,
``"openai/gpt-4o"`` etc.). When the hook is invoked without a usable
backend it degrades to a no-op so the pipeline never crashes.

Three pluggable hooks are exposed:

* :func:`build_classifier_hook` — returns a function compatible with
  ``FeatureClassifier(llm_hook=...)`` that maps a ``FeatureRecord``
  to a ``FeatureCategory``. Only invoked when the rule-based
  classifier produced ``FeatureCategory.UNKNOWN`` so the LLM is not
  called for every record.
* :func:`build_extractor_hook` — returns a function compatible with
  ``FeatureExtractor(llm_hook=...)`` that refines the rule-based
  candidate list with LLM-extracted structure.
* :func:`build_summarizer_hook` — returns a function that produces a
  one-paragraph digest summary from the candidate list.

The hooks intentionally accept a ``client`` argument so tests can
substitute a fake; the default client lazily imports ``litellm`` so
importing this module never hard-depends on a runtime model SDK.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from .models import (
    FeatureCategory,
    FeatureRecord,
    FeatureType,
    utc_now_iso,
)

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy litellm import
# ---------------------------------------------------------------------------


_LITELLM: Any | None = None
_LITELLM_IMPORT_ERROR: Exception | None = None


def _get_litellm() -> Any:
    """Return the cached litellm module (lazy-imported).

    Caches the import error so subsequent attempts do not re-pay the
    ImportError cost. Tests that need to stub the module call
    :func:`set_litellm_module` first.
    """
    global _LITELLM, _LITELLM_IMPORT_ERROR
    if _LITELLM is not None:
        return _LITELLM
    if _LITELLM_IMPORT_ERROR is not None:
        raise _LITELLM_IMPORT_ERROR
    try:
        import litellm  # type: ignore
    except Exception as exc:
        _LITELLM_IMPORT_ERROR = exc
        raise RuntimeError(
            "litellm is required for SR-5.1 LLM hooks. "
            "Install with `pip install litellm` or disable "
            "`use_llm` in RadarConfig."
        ) from exc
    _LITELLM = litellm
    return litellm


def set_litellm_module(module: Any) -> None:
    """Test seam: install a fake litellm module."""
    global _LITELLM, _LITELLM_IMPORT_ERROR
    _LITELLM = module
    _LITELLM_IMPORT_ERROR = None


def reset_litellm_module() -> None:
    """Drop the cached litellm module so the next call re-imports."""
    global _LITELLM, _LITELLM_IMPORT_ERROR
    _LITELLM = None
    _LITELLM_IMPORT_ERROR = None


# ---------------------------------------------------------------------------
# JSON helpers — accept either plain JSON or JSON fenced in ```json … ```
# ---------------------------------------------------------------------------


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(?P<body>\{.*?\}|\[.*?\])\s*```", re.DOTALL)


def _extract_json(text: str) -> Any:
    """Best-effort JSON extraction from an LLM response."""
    if not text:
        return None
    fence = _JSON_FENCE_RE.search(text)
    candidate = fence.group("body") if fence else text.strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        # Some models add a trailing comma or prose around the JSON.
        # Try slicing the first { … } block.
        start = candidate.find("{")
        end = candidate.rfind("}")
        if 0 <= start < end:
            try:
                return json.loads(candidate[start : end + 1])
            except json.JSONDecodeError:
                pass
        return None


# ---------------------------------------------------------------------------
# Low-level completion helper
# ---------------------------------------------------------------------------


@dataclass
class LLMConfig:
    """Runtime knobs for LLM-assisted SR-5.1 hooks."""

    model: str
    api_key: str | None = None
    api_base: str | None = None
    timeout_seconds: float = 30.0
    temperature: float = 0.2
    max_tokens: int = 512
    # Optional override for test injection.
    client: Any | None = None

    @classmethod
    def from_env(cls, *, model: str | None = None) -> LLMConfig:
        """Build from ``CLAWCODEX_RADAR_LLM_*`` env vars."""
        m = model or os.environ.get("CLAWCODEX_RADAR_LLM_MODEL") or os.environ.get("CLAWCODEX_RADAR_LLM") or ""
        if not m:
            raise ValueError(
                "LLMConfig requires a model name. Set CLAWCODEX_RADAR_LLM_MODEL or pass model=... explicitly."
            )
        api_key = (
            os.environ.get("CLAWCODEX_RADAR_LLM_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("ANTHROPIC_API_KEY")
        )
        api_base = os.environ.get("CLAWCODEX_RADAR_LLM_API_BASE")
        try:
            timeout_seconds = float(os.environ.get("CLAWCODEX_RADAR_LLM_TIMEOUT", "30.0"))
        except ValueError:
            _log.warning("invalid CLAWCODEX_RADAR_LLM_TIMEOUT; falling back to 30.0")
            timeout_seconds = 30.0
        return cls(
            model=m,
            api_key=api_key,
            api_base=api_base,
            timeout_seconds=timeout_seconds,
        )


def _complete(prompt: str, config: LLMConfig) -> str:
    """Run a single completion; returns the assistant text or empty string."""
    client = config.client
    if client is None:
        litellm = _get_litellm()
        client = litellm
    kwargs: dict[str, Any] = {
        "model": config.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "timeout": config.timeout_seconds,
    }
    if config.api_key:
        kwargs["api_key"] = config.api_key
    if config.api_base:
        kwargs["api_base"] = config.api_base
    try:
        response = client.completion(**kwargs)
    except Exception as exc:  # noqa: BLE001
        _log.warning("LLM completion failed: %s", exc)
        return ""
    try:
        return response["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        return ""


def _safe_format(template: str, **kwargs: object) -> str:
    """Fill ``{key}`` placeholders without interpreting braces inside values.

    Unlike ``str.format``, ``{``/``}`` inside substituted values (release
    notes routinely contain JSON snippets or ``${VAR}``) is left untouched.
    """
    text = template.replace("{{", "\x00").replace("}}", "\x01")
    for key, value in kwargs.items():
        text = text.replace("{" + key + "}", str(value))
    return text.replace("\x00", "{").replace("\x01", "}")


# ---------------------------------------------------------------------------
# Classifier hook (refine UNKNOWN → concrete category)
# ---------------------------------------------------------------------------


_CATEGORY_NAMES: tuple[str, ...] = tuple(c.value for c in FeatureCategory)
_TYPE_NAMES: tuple[str, ...] = tuple(t.value for t in FeatureType)


_CLASSIFIER_PROMPT = (
    "You are classifying a single community feature record from an open-source "
    "agent project. Pick the single best ClawCodex taxonomy category and feature type.\n"
    "Allowed categories: {cats}.\nAllowed types: {types}.\n"
    'Respond with JSON only, no prose: {{"category": "<one>", "feature_type": "<one>"}}.\n\n'
    "Title: {title}\nDescription: {description}\nSource: {source}\n"
)


def build_classifier_hook(
    config: LLMConfig | None = None,
) -> Callable[[FeatureRecord], FeatureCategory]:
    """Return a classifier hook suitable for ``FeatureClassifier(llm_hook=...)``.

    The hook is invoked only when the rule-based classifier produces
    ``FeatureCategory.UNKNOWN``; the LLM therefore sees a curated subset.
    On any failure (timeout, bad JSON, unexpected category) the hook
    returns ``UNKNOWN`` so the pipeline never breaks the digest.
    """
    cfg = config or LLMConfig.from_env()

    def hook(record: FeatureRecord) -> FeatureCategory:
        prompt = _safe_format(
            _CLASSIFIER_PROMPT,
            cats=", ".join(_CATEGORY_NAMES),
            types=", ".join(_TYPE_NAMES),
            title=(record.title or "")[:200],
            description=(record.description or "")[:500],
            source=record.source or "",
        )
        text = _complete(prompt, cfg)
        data = _extract_json(text)
        if not isinstance(data, dict):
            return FeatureCategory.UNKNOWN
        cat_value = str(data.get("category") or "")
        try:
            return FeatureCategory(cat_value)
        except ValueError:
            return FeatureCategory.UNKNOWN

    return hook


# ---------------------------------------------------------------------------
# Extractor hook (refine rule-extracted candidates)
# ---------------------------------------------------------------------------


_EXTRACTOR_PROMPT = (
    "You are refining a list of feature candidates extracted from a release note. "
    "Re-rank them by importance, drop pure bug-fix entries, and return a JSON array. "
    'Each element: {{"title": "<short>", "description": "<one sentence>"}}.\n'
    "If a candidate is not actually a feature, omit it. Keep at most {max_keep} items.\n\n"
    "Source: {source}\nRelease tag: {tag}\nRelease body:\n{body}\n\n"
    "Candidates (one per line):\n{candidates}\n"
)


def build_extractor_hook(
    config: LLMConfig | None = None,
    *,
    max_keep: int = 20,
) -> Callable[[list[FeatureRecord], str], list[FeatureRecord]]:
    """Return an extractor hook for ``FeatureExtractor(llm_hook=...)``.

    The hook only runs when :attr:`RadarConfig.use_llm` is True and at
    least one candidate was extracted by the rules; if the LLM fails or
    returns unparseable JSON the rule-based list is returned untouched.
    """
    cfg = config or LLMConfig.from_env()

    def hook(records: list[FeatureRecord], body: str) -> list[FeatureRecord]:
        if not records:
            return records
        candidates = "\n".join(
            f"- {(r.title or '').strip()} | {(r.description or '').strip()}" for r in records[: max_keep * 2]
        )
        prompt = _safe_format(
            _EXTRACTOR_PROMPT,
            max_keep=max_keep,
            source=records[0].source if records else "",
            tag=records[0].released_at or "",
            body=(body or "")[:1500],
            candidates=candidates,
        )
        text = _complete(prompt, cfg)
        data = _extract_json(text)
        if not isinstance(data, list):
            return records
        refined: list[FeatureRecord] = []
        for idx, item in enumerate(data[:max_keep]):
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            if not title:
                continue
            description = str(item.get("description") or "").strip()
            template = records[0]
            refined.append(
                FeatureRecord(
                    id=f"{template.id}-llm-{idx}",
                    source=template.source,
                    title=title,
                    description=description or title,
                    category=template.category,
                    feature_type=template.feature_type,
                    released_at=template.released_at,
                    url=template.url,
                    related_projects=list(template.related_projects),
                    tags=list(template.tags),
                    raw_body=template.raw_body,
                )
            )
        return refined or records

    return hook


# ---------------------------------------------------------------------------
# Summarizer hook (digest one-liner)
# ---------------------------------------------------------------------------


_SUMMARIZER_PROMPT = (
    "Write a one-paragraph Chinese-language summary for a community digest covering "
    "{n_features} candidate features across {n_projects} projects. Highlight the "
    "top 3 categories and call out any breaking changes. Plain prose, no bullet "
    "points, no JSON, ≤ 200 Chinese characters.\n\n"
    "Projects: {projects}\n"
    "Category counts: {categories}\n"
    "Breaking changes: {breaking}\n"
)


def build_summarizer_hook(
    config: LLMConfig | None = None,
) -> Callable[[Iterable[FeatureRecord], Iterable[str], Iterable[FeatureRecord]], str]:
    """Return a callable that produces a Chinese one-paragraph digest summary.

    Accepts ``(features, projects, breaking_changes)`` and returns the
    summary text. On any failure the hook returns an empty string so the
    pipeline can fall back to its deterministic stub.
    """
    cfg = config or LLMConfig.from_env()

    def hook(
        features: Iterable[FeatureRecord],
        projects: Iterable[str],
        breaking_changes: Iterable[FeatureRecord],
    ) -> str:
        features = list(features)
        projects = list(projects)
        breaking = list(breaking_changes)
        from collections import Counter

        counts = Counter(f.category.value for f in features)
        prompt = _safe_format(
            _SUMMARIZER_PROMPT,
            n_features=len(features),
            n_projects=len(projects),
            projects=", ".join(projects[:8]) or "(none)",
            categories=", ".join(f"{k}={v}" for k, v in counts.most_common(5)) or "(none)",
            breaking="\n".join(f"- {b.title}" for b in breaking[:5]) or "(none)",
        )
        return _complete(prompt, cfg).strip()

    return hook


# ---------------------------------------------------------------------------
# Convenience: stamp a generation timestamp on LLM-touched digests
# ---------------------------------------------------------------------------


def llm_generated_marker() -> str:
    """Return ``" (LLM-assisted)"`` so digests can self-identify."""
    return f" (LLM-assisted at {utc_now_iso()})"


__all__ = [
    "LLMConfig",
    "build_classifier_hook",
    "build_extractor_hook",
    "build_summarizer_hook",
    "llm_generated_marker",
    "reset_litellm_module",
    "set_litellm_module",
]

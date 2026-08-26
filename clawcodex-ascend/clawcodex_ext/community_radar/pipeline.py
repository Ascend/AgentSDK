# pylint: disable=relative-beyond-top-level
"""End-to-end pipeline orchestrator for SR-5.1.

Glues :class:`SourceRegistry`, :class:`Fetcher`, :class:`FeatureExtractor`,
:class:`FeatureClassifier`, :class:`FeatureDeduplicator`,
:class:`FeatureScorer`, and :class:`CommunityReporter` into a single
``run_scan()`` call. CLI / Cron call this — they never touch the
individual modules directly.

Pipeline stages:

    registry → fetcher → extractor → classifier → dedup → scorer →
    reporter → dual-write

Every stage is failure-isolated: an exception in one source does not
abort the scan. Errors are captured in :attr:`CommunityDigest.errors`
so the digest always renders (with a "fetch errors" section).
"""

from __future__ import annotations

import json
import logging
import re
import threading
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .classifier import FeatureClassifier
from .config import RadarConfig, apply_env_overrides
from .cron_integration import _load_config_safely, ensure_cron_installed, load_registry_safely
from .deduplicator import FeatureDeduplicator
from .extractor import FeatureExtractor
from .fetcher import Fetcher
from .i18n import get_text
from .models import (
    CommunityDigest,
    FeatureRecord,
    FetchResult,
    ScoredFeature,
    WatchSource,
)
from .notifier import DigestNotifier
from .registry import SourceRegistry
from .reporter import CommunityReporter, DigestWriteResult, copy_to_persistent
from .scorer import FeatureScorer

_log = logging.getLogger(__name__)

# ── LLM importance classification ──────────────────────────────────────────


# Empirical: 36 k chars of JSON output covers ≈160 features with Chinese
# translations.  A batch of 150 features keeps the LLM response safely
# under the 16 384 max_tokens limit so truncation is rare.
_LLM_BATCH_SIZE = 150


def _build_classify_prompt(batch: list[dict[str, Any]]) -> str:
    """Build a classification prompt for a single batch.

    Always uses the zh prompt so MAJOR/MINOR decisions are identical
    regardless of the report language, and title_zh/desc_zh translations
    are always generated in the same LLM call.
    """
    features_json = json.dumps(batch, ensure_ascii=False, indent=2)
    prompt = get_text("llm_importance_prompt", "zh", n=len(batch))
    return prompt.replace("{features_json}", features_json)


def _llm_classify_importance(
    scored: list[ScoredFeature],
) -> dict[str, dict[str, str]]:
    """Call the LLM (via LiteLLM) to classify features as MAJOR/MINOR.

    Always uses the zh prompt so MAJOR/MINOR decisions are identical
    regardless of report language, and title_zh/desc_zh translations
    are generated in the same call.

    Features are split into batches to avoid token-limit truncation.
    Batch 1 runs synchronously first (to probe the working model + cache
    it), then batches 2..N are dispatched in parallel.
    """
    if not scored:
        return {}

    all_features: list[dict[str, Any]] = []
    for item in scored:
        all_features.append(
            {
                "id": item.record.id,
                "title": item.record.title,
                "description": item.record.description[:200],
                "category": item.record.category.value,
                "score": round(item.score.overall, 1),
                "source": item.record.source,
                "related_projects": item.record.related_projects,
            }
        )

    n_total = len(scored)

    # Build all batch prompts upfront
    batch_jobs: list[tuple[int, int, str]] = []  # (start_idx, count, prompt)
    for batch_start in range(0, n_total, _LLM_BATCH_SIZE):
        batch = all_features[batch_start : batch_start + _LLM_BATCH_SIZE]
        prompt = _build_classify_prompt(batch)
        batch_jobs.append((batch_start, len(batch), prompt))

    merged: dict[str, dict[str, str]] = {}

    # ── Batch 1: synchronous probe + real work ──
    if batch_jobs:
        b1_start, b1_len, b1_prompt = batch_jobs[0]
        parsed = _call_and_parse_batch(b1_prompt)
        if parsed:
            _log.warning(
                "LLM batch %d–%d: parsed %d/%d features (probe)", b1_start + 1, b1_start + b1_len, len(parsed), b1_len
            )
            merged.update(parsed)

    rest = batch_jobs[1:]
    if not rest:
        return merged

    # ── Batches 2..N: parallel via thread pool ──
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures: dict[Any, tuple[int, int]] = {}
        for batch_start, batch_len, prompt in rest:
            future = executor.submit(_call_and_parse_batch, prompt)
            futures[future] = (batch_start, batch_len)

        for future in as_completed(futures):
            batch_start, batch_len = futures[future]
            try:
                parsed = future.result()
                if parsed:
                    _log.warning(
                        "LLM batch %d–%d: parsed %d/%d features",
                        batch_start + 1,
                        batch_start + batch_len,
                        len(parsed),
                        batch_len,
                    )
                    merged.update(parsed)
                else:
                    _log.warning(
                        "LLM batch %d–%d: empty result",
                        batch_start + 1,
                        batch_start + batch_len,
                    )
            except Exception as exc:  # noqa: BLE001
                _log.warning(
                    "LLM batch %d–%d failed: %s; continuing",
                    batch_start + 1,
                    batch_start + batch_len,
                    exc,
                )

    _log.warning(
        "LLM classification: %d/%d features classified across %d batches", len(merged), n_total, len(batch_jobs)
    )
    return merged


# ── LLM classification cache ────────────────────────────────────────────────
# Avoids non-deterministic LLM output causing different highlights between
# consecutive scans with different --language values.  The cache is keyed by
# a content-fingerprint (sorted feature IDs + scores) with a 1-hour TTL.


def _feature_fingerprint(scored: list[ScoredFeature]) -> str:
    """Stable hash over the features being classified."""
    import hashlib as _hl

    parts: list[str] = []
    for sf in sorted(scored, key=lambda s: s.record.id):
        parts.append(f"{sf.record.id}:{sf.score.overall:.1f}")
    return _hl.sha256("|".join(parts).encode()).hexdigest()[:16]


def _llm_cache_path(cache_dir: str | None) -> Path | None:
    """Return the cache file path or None when cache_dir is unset."""
    if not cache_dir:
        return None
    return Path(cache_dir) / "llm_classification_cache.json"


def _llm_classify_importance_cached(
    scored: list[ScoredFeature],
    cache_dir: str | None = None,
) -> dict[str, dict[str, str]]:
    """Call the LLM, using a 1-hour cache to ensure repeatable results."""
    cache_path = _llm_cache_path(cache_dir)
    fingerprint = _feature_fingerprint(scored)

    # ── Try cache ──
    if cache_path is not None and cache_path.exists():
        try:
            raw = cache_path.read_text(encoding="utf-8")
            cache = json.loads(raw)
            entry = cache.get(fingerprint)
            if isinstance(entry, dict):
                ts = entry.get("_ts", 0)
                age_s = datetime.now(timezone.utc).timestamp() - ts
                if age_s < 3600:  # 1-hour TTL
                    result = _validate_cache_result(entry.get("data", {}), scored)
                    if result is not None:
                        _log.info("LLM classification cache hit (age=%.0fs)", age_s)
                        return result
        except Exception:  # noqa: BLE001  # nosec B110
            pass

    # ── Call LLM ──
    result = _llm_classify_importance(scored)

    # ── Write cache ──
    if cache_path is not None and result:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            existing: dict[str, Any] = {}
            if cache_path.exists():
                try:
                    existing = json.loads(cache_path.read_text(encoding="utf-8"))
                except Exception:  # noqa: BLE001  # nosec B110
                    pass
            if not isinstance(existing, dict):
                existing = {}
            existing[fingerprint] = {
                "_ts": datetime.now(timezone.utc).timestamp(),
                "data": result,
            }
            # Prune entries older than 24 hours
            cutoff = datetime.now(timezone.utc).timestamp() - 86400
            existing = {k: v for k, v in existing.items() if isinstance(v, dict) and v.get("_ts", 0) > cutoff}
            cache_path.write_text(
                json.dumps(existing, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:  # noqa: BLE001
            _log.debug("failed to write LLM classification cache", exc_info=True)

    return result


def _validate_cache_result(
    cached: dict[str, Any],
    scored: list[ScoredFeature],
) -> dict[str, dict[str, str]] | None:
    """Check that the cached result covers all the features we need."""
    needed = {sf.record.id for sf in scored}
    if not needed:
        return None
    # At least 90% of needed features must be in the cache
    got = set(cached.keys())
    coverage = len(got & needed) / len(needed)
    if coverage < 0.9:
        return None
    # Return only the entries for features we still care about
    return {fid: {str(k): str(v) for k, v in entry.items()} for fid, entry in cached.items() if fid in needed}


# ═════════════════════════════════════════════════════════════════════════════


def _call_and_parse_batch(prompt: str) -> dict[str, dict[str, str]] | None:
    """Call LLM + parse response for a single batch.  Top-level helper so
    ThreadPoolExecutor can pickle it.
    """
    result = _call_litellm(prompt)
    if result is None:
        return None
    return _parse_llm_importance_response(result)


def _resolve_api_key_for_model(model: str) -> tuple[str | None, str | None]:
    """Resolve an API key for *model* by checking:
    1. ClawCodex provider config (``~/.clawcodex/config.json``)
    2. Standard environment variables (``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``)

    Returns ``(api_key, provider_name)`` — both are ``None`` when no key
    is configured so the caller can degrade gracefully.
    """
    import os as _os

    # Parse provider prefix from litellm model name: "openai/gpt-4o-mini" → "openai"
    provider = model.split("/")[0] if "/" in model else model

    # 1. Try ClawCodex credential store
    # pylint: disable=too-many-nested-blocks
    try:
        from src.config import load_config as _load_clawcodex_config
    except Exception:  # noqa: BLE001  # nosec B110
        pass
    else:
        try:
            cfg = _load_clawcodex_config()
            providers: dict = cfg.get("providers", {})
            if isinstance(providers, dict):
                # 1a. Exact match on the model's provider prefix
                if provider in providers:
                    entry = providers[provider]
                    if isinstance(entry, dict) and entry.get("api_key"):
                        return str(entry["api_key"]), provider
                # 1b. Case-insensitive match
                for key, value in providers.items():
                    if isinstance(value, dict) and key.lower() == provider.lower():
                        api_key = value.get("api_key", "")
                        if api_key:
                            return str(api_key), key
                # 1c. Fallback: only when provider is absent from config entirely,
                #     use the first configured api_key we find (any provider).
                #     Do NOT fall back when the provider exists but has no key —
                #     that would inject a wrong provider's key (e.g. deepseek →
                #     OPENAI_API_KEY) and produce spurious auth errors.
                if provider not in providers and not any(k.lower() == provider.lower() for k in providers):
                    for key, value in providers.items():
                        if isinstance(value, dict) and value.get("api_key"):
                            return str(value["api_key"]), key
        except Exception:  # noqa: BLE001  # nosec B110
            pass

    # 2. Standard env vars — only check the one matching this provider
    env_var = _PROVIDER_ENV_MAP.get(provider)
    if env_var:
        val = _os.environ.get(env_var)
        if val:
            return val, provider

    return None, None


_PROVIDER_ENV_MAP: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}

# Guards _cached_llm_model so concurrent scans cannot interleave sets/clears.
_llm_model_lock = threading.Lock()

_cached_llm_model: str | None = None
"""After the first successful LLM call the working model is cached here
so subsequent batches skip the provider-fallback loop entirely."""


def _call_litellm(prompt: str) -> str | None:
    """Call LiteLLM with a simple completion prompt.

    The first call iterates the model list (openai → deepseek → anthropic)
    to find a working provider.  Once found the model is cached so every
    later call goes straight to it, avoiding repeated auth failures and
    their stderr noise.
    """
    global _cached_llm_model

    try:
        import litellm  # type: ignore
    except Exception:  # noqa: BLE001
        _log.debug("litellm not available; skipping LLM importance classification")
        return None

    litellm.suppress_debug_info = True
    logging.getLogger("litellm").setLevel(logging.WARNING)

    import os as _os

    # Ordered by preference — first model whose provider has a key wins.
    _MODELS = [
        "openai/gpt-4o-mini",
        "openai/gpt-4o",
        "deepseek/deepseek-chat",
        "anthropic/claude-sonnet-4-6",
    ]

    # When we already know the working model, try it first. On failure
    # (_try_litellm_call clears the cache), fall through to the probe loop
    # so the call doesn't silently return None.
    if _cached_llm_model is not None:
        result = _try_litellm_call(_cached_llm_model, _PROVIDER_ENV_MAP, _os, prompt)
        if result is not None:
            return result

    for model in _MODELS:
        result = _try_litellm_call(model, _PROVIDER_ENV_MAP, _os, prompt)
        if result is not None:
            with _llm_model_lock:
                _cached_llm_model = model
            return result

    _log.warning("all litellm models failed for importance classification")
    return None


def _try_litellm_call(
    model: str,
    provider_env_map: dict[str, str],
    _os: Any,
    prompt: str,
) -> str | None:
    """Attempt a single litellm call.  Returns the text or None."""
    import litellm  # type: ignore

    provider = model.split("/")[0] if "/" in model else model
    api_key, _found_provider = _resolve_api_key_for_model(model)
    if api_key:
        env_var = provider_env_map.get(provider, "OPENAI_API_KEY")
        _os.environ.setdefault(env_var, api_key)
    try:
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=16384,
            temperature=0.0,
        )
        _log.warning("LLM success with model=%s", model)
        raw = response.choices[0].message.content
        _log.warning("LLM raw response length=%d", len(raw) if raw else 0)
        return raw  # type: ignore[no-any-return]
    except Exception:  # noqa: BLE001
        global _cached_llm_model
        # Only log the first failure per model; subsequent cached-path
        # failures still get logged so we don't mask a key-revocation.
        if _cached_llm_model is None:
            _log.debug("LLM model=%s failed (no credentials or auth error)", model)
        else:
            _log.warning("LLM cached model=%s failed; will re-probe on next call", model)
            with _llm_model_lock:
                _cached_llm_model = None
        return None


def _parse_truncated_json_objects(text: str) -> list[dict[str, Any]]:
    """Extract individual JSON objects from a possibly-truncated JSON array.

    When the LLM response is cut off by token limits, the outer ``[...]``
    will be incomplete.  This scanner walks the text character by character,
    tracking brace depth, and yields each top-level object when its closing
    ``}`` is found.  Objects that span more than one nesting level (e.g. a
    string containing ``{``) are handled correctly by the depth counter.
    """
    objects: list[dict[str, Any]] = []
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                candidate = text[start : i + 1]
                try:
                    obj = json.loads(candidate)
                    if isinstance(obj, dict):
                        objects.append(obj)
                except json.JSONDecodeError:
                    pass
                start = -1
    return objects


def _parse_llm_importance_response(raw: str) -> dict[str, dict[str, str]]:
    """Parse the LLM JSON response into a ``feature_id → {level, highlight}`` map.

    Tries multiple parsing strategies to be robust against common LLM output
    quirks (markdown code fences, trailing commas, extra text).
    """
    text = raw.strip()

    # Strategy 1: direct JSON parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return _normalize_llm_result(parsed)
    except json.JSONDecodeError:
        pass

    # Strategy 2: extract JSON array from markdown code fences
    fence_match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if fence_match:
        try:
            parsed = json.loads(fence_match.group(1))
            if isinstance(parsed, list):
                return _normalize_llm_result(parsed)
        except json.JSONDecodeError:
            pass

    # Strategy 3: find outermost [...] in the text
    bracket_match = re.search(r"\[.*\]", text, re.DOTALL)
    if bracket_match:
        try:
            parsed = json.loads(bracket_match.group(0))
            if isinstance(parsed, list):
                return _normalize_llm_result(parsed)
        except json.JSONDecodeError:
            pass

    # Strategy 4: handle truncated JSON array (common when max_tokens is
    #     reached before all features are serialised)
    objects = _parse_truncated_json_objects(text)
    if objects:
        return _normalize_llm_result(objects)

    # Strategy 5: line-by-line — try to parse each non-empty line as an
    # individual JSON object, then reuse the shared normaliser so all
    # strategies keep the same fields (level/highlight/title_zh/desc_zh).
    parsed: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith(("```", "#")):
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict) and "id" in obj:
                parsed.append(obj)
        except json.JSONDecodeError:
            continue
    return _normalize_llm_result(parsed)


def _normalize_llm_result(
    parsed: list[dict[str, Any]],
) -> dict[str, dict[str, str]]:
    """Convert a list of LLM response objects to a ``feature_id → {level, highlight, title_zh, desc_zh}`` map."""
    result: dict[str, dict[str, str]] = {}
    for obj in parsed:
        if not isinstance(obj, dict) or "id" not in obj:
            continue
        fid = str(obj["id"])
        level = str(obj.get("level", "MINOR")).upper()
        if level not in ("MAJOR", "MINOR"):
            level = "MINOR"
        highlight = str(obj.get("highlight", ""))
        title_zh = str(obj.get("title_zh", ""))
        desc_zh = str(obj.get("desc_zh", ""))
        result[fid] = {
            "level": level,
            "highlight": highlight,
            "title_zh": title_zh,
            "desc_zh": desc_zh,
        }
    return result


@dataclass
class ScanResult:
    digest: CommunityDigest
    write_result: DigestWriteResult | None
    records: list[FeatureRecord]
    notifications: dict[str, bool] | None = None
    cron_status: dict[str, Any] | None = None
    issue_sync: Any | None = None  # IssueSyncResult (lazy import avoids circular dep)


class CommunityRadarPipeline:
    """High-level entrypoint used by the CLI and the Cron trigger."""

    def __init__(
        self,
        *,
        config: RadarConfig | None = None,
        registry: SourceRegistry | None = None,
        fetcher: Fetcher | None = None,
        extractor: FeatureExtractor | None = None,
        classifier: FeatureClassifier | None = None,
        deduplicator: FeatureDeduplicator | None = None,
        scorer: FeatureScorer | None = None,
        reporter: CommunityReporter | None = None,
        notifier: DigestNotifier | None = None,
        ensure_cron: bool = True,
    ) -> None:
        self.config = config or RadarConfig()
        self.registry = registry
        self.extractor = extractor or FeatureExtractor()

        # Build source_name → domain map from the registry so the
        # classifier can reject cross-domain keyword matches.
        _domain_map: dict[str, str] = {}
        if self.registry is not None:
            for source in self.registry.list():
                if source.domain:
                    _domain_map[source.name] = source.domain
        if classifier is not None:
            self.classifier = classifier
        else:
            self.classifier = FeatureClassifier(
                roadmap_keywords=self.config.roadmap_keywords,
                source_domain_map=_domain_map,
            )
        self._owns_classifier = classifier is None
        self.deduplicator = deduplicator or FeatureDeduplicator()
        self.scorer = scorer or FeatureScorer(self.config)
        self.reporter = reporter or CommunityReporter(self.config)
        self.notifier = notifier or DigestNotifier(self.config)
        self._ensure_cron = ensure_cron
        self._owns_fetcher = fetcher is None
        self.fetcher = fetcher

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_scan(
        self,
        *,
        sources: Iterable[WatchSource] | None = None,
        period: str = "weekly",
        write: bool = True,
        output_dir: Path | str | None = None,
        persistent_copy: bool = True,
        notify: bool | None = None,
        auto_install_cron: bool | None = None,
        compare: bool = False,
        incremental: bool = False,
        issue_sync_target: Any = None,  # ResolvedTarget (lazy import)
        issue_sync_cli_repo: str | None = None,
        issue_sync_cli_platform: str | None = None,
        issue_sync_closed_issue_mode: str | None = None,
    ) -> ScanResult:
        """Run the full pipeline and (optionally) persist a digest.

        ``sources`` overrides the registry when supplied; callers that
        already pre-loaded their own list of :class:`WatchSource`
        records can skip the registry entirely.

        ``notify`` and ``auto_install_cron`` override the corresponding
        ``RadarConfig`` flags for this call only. ``None`` means
        "inherit from config".
        """
        sources_provided = sources is not None
        sources = list(sources) if sources is not None else self._load_sources()
        if not sources:
            _log.warning("community radar scan: no sources configured")

        # Auto-detect domains for sources still marked "general" so the
        # classifier's domain-blocking logic can work correctly.
        if self.registry is not None:
            import os as _os

            token = _os.environ.get("GITHUB_TOKEN")
            detected = self.registry.auto_detect_domains(self.config.cache_dir, github_token=token)
            if detected:
                _log.info("auto-detected domain for %d source(s)", detected)
                if self._owns_classifier:
                    _domain_map = {s.name: s.domain for s in self.registry.list() if s.domain}
                    self.classifier = FeatureClassifier(
                        roadmap_keywords=self.config.roadmap_keywords,
                        source_domain_map=_domain_map,
                    )
                if not sources_provided:
                    sources = self._load_sources()

        cron_status: dict[str, Any] | None = None
        if auto_install_cron is None:
            auto_install_cron = self._ensure_cron
        if auto_install_cron:
            try:
                summary = ensure_cron_installed()
                cron_status = {
                    "task_id": summary.task_id,
                    "installed": summary.installed,
                    "schedule": summary.schedule,
                    "message": summary.message,
                }
            except Exception as exc:  # noqa: BLE001
                _log.warning("ensure_cron_installed failed: %s", exc)
                cron_status = {"error": str(exc)}

        notify_flag = self.config.notify if notify is None else bool(notify)
        notifications: dict[str, bool] | None = None

        # ── Time-range for period-based scanning ──
        if period == "monthly":
            since = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
        elif period == "weekly":
            since = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            since = None  # "full" — no time filter, pull everything
        _log.info("Scan period=%s, since=%s", period, since)

        try:
            fetch_results = self._fetch_all(sources, incremental=incremental, since=since)
            records = self._extract(fetch_results)
            records = self.classifier.classify_many(records)
            records = self.deduplicator.deduplicate(records)
            scored = [ScoredFeature(record=record, score=self.scorer.score(record)) for record in records]

            # ── Filter excluded feature types ──
            exclude_types_lower = {t.lower() for t in self.config.exclude_feature_types}
            filtered_out: list[FeatureRecord] = []
            kept_records: list[FeatureRecord] = []
            for record in records:
                if record.feature_type.value.lower() in exclude_types_lower:
                    filtered_out.append(record)
                else:
                    kept_records.append(record)
            kept_ids = {r.id for r in kept_records}
            scored = [s for s in scored if s.record.id in kept_ids]
            # Also filter the records list used for stats and breaking-changes
            records = kept_records

            # ── LLM importance classification ──
            # Always uses the zh prompt so MAJOR/MINOR decisions are identical
            # regardless of report language, and title_zh/desc_zh are generated
            # in the same call.
            # Results are cached (keyed by feature-id fingerprint) so that
            # consecutive scans with different --language produce the same
            # highlights — only the display language differs, not the content.
            llm_importance: dict[str, dict[str, str]] = {}
            if scored and self.config.use_llm:
                _llm_limit = max(
                    200,
                    3 * self.config.max_features_per_report,
                )
                _candidates = sorted(scored, key=lambda s: s.score.overall, reverse=True)
                _top_for_llm = _candidates[:_llm_limit]
                _log.info(
                    "LLM classification starting for top-%d/%d features",
                    len(_top_for_llm),
                    len(scored),
                )
                llm_importance = _llm_classify_importance_cached(
                    _top_for_llm,
                    self.config.cache_dir,
                )
                _log.info("LLM classification complete: %d features classified", len(llm_importance))

            versions_total = sum(len(fr.releases) for fr in fetch_results)
            errors: list[str] = []
            for fr in fetch_results:
                errors.extend(f"[{fr.source}] {msg}" for msg in fr.errors)

            digest = self.reporter.build_digest(
                period=period,
                features=records,
                scored=scored,
                sources_used=[s.name for s in sources],
                errors=errors,
                versions_total=versions_total,
                filtered_count=len(filtered_out),
                lang=self.config.language,
                llm_importance=llm_importance,
            )
            digest.period_start = since or ""

            write_result: DigestWriteResult | None = None
            if write:
                target_dir = Path(output_dir or self.config.output_dir)
                write_result = self.reporter.write(digest, target_dir, compare=compare)
                if persistent_copy:
                    copy_to_persistent(write_result.markdown_path)
                    copy_to_persistent(write_result.json_path)
                    if write_result.proposals_path is not None:
                        copy_to_persistent(write_result.proposals_path)

            if notify_flag and write_result is not None:
                try:
                    notifications = self.notifier.broadcast(digest, write_result)
                except Exception as exc:  # noqa: BLE001
                    _log.warning("notification broadcast failed: %s", exc)
                    notifications = {"error": str(exc)}  # type: ignore[assignment]

            # ── GitCode / GitHub / Gitee issue sync ──
            issue_sync_result = None
            if self.config.sync_issues:
                try:
                    from .issue_sync import sync_features_to_issues

                    issue_sync_result = sync_features_to_issues(
                        digest=digest,
                        llm_importance=llm_importance,
                        config=self.config,
                        max_n=self.config.sync_issues_max_per_scan,
                        target=issue_sync_target,
                        cli_repo=issue_sync_cli_repo,
                        cli_platform=issue_sync_cli_platform,
                        cache_dir=self.config.cache_dir,
                        closed_issue_mode=issue_sync_closed_issue_mode,
                    )
                    if issue_sync_result.created:
                        _log.info(
                            "issue sync: created %d issues",
                            len(issue_sync_result.created),
                        )
                    if issue_sync_result.errors:
                        _log.warning(
                            "issue sync errors: %s",
                            "; ".join(issue_sync_result.errors),
                        )
                except Exception as exc:  # noqa: BLE001
                    _log.warning("issue sync failed: %s", exc)
        finally:
            # Always close an owned fetcher so HTTP clients don't leak
            # even when the pipeline short-circuits on empty input.
            if self._owns_fetcher and self.fetcher is not None:
                self.fetcher.close()
                self.fetcher = None

        return ScanResult(
            digest=digest,
            write_result=write_result,
            records=records,
            notifications=notifications,
            cron_status=cron_status,
            issue_sync=issue_sync_result,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_sources(self) -> list[WatchSource]:
        if self.registry is None:
            _log.debug("no registry configured; returning empty source list")
            return []
        if not self.registry._sources:  # type: ignore[attr-defined]
            try:
                self.registry.load()
            except Exception as exc:  # noqa: BLE001
                _log.warning("registry load failed: %s", exc)
        return self.registry.list()

    def _fetch_all(
        self,
        sources: list[WatchSource],
        *,
        incremental: bool = False,
        since: str | None = None,
    ) -> list:
        if not sources:
            return []
        if self.fetcher is None:
            self.fetcher = Fetcher(cache_dir=self.config.cache_dir)

        if len(sources) <= 1:
            return self.fetcher.fetch_all(sources, incremental=incremental, since=since)

        # Parallel fetch: GitHub API latency dominates, so thread-per-source
        # saturates network without extra complexity.

        results: list[Any] = [None] * len(sources)
        with ThreadPoolExecutor(max_workers=min(len(sources), 6)) as executor:
            futures: dict[Any, int] = {}
            for i, source in enumerate(sources):
                future = executor.submit(self.fetcher.fetch, source, incremental=incremental, since=since)
                futures[future] = i
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as exc:  # noqa: BLE001
                    _log.warning("fetch failed for %s: %s", sources[idx].name, exc)
                    results[idx] = FetchResult(
                        source=sources[idx].name,
                        releases=[],
                        errors=[f"[{sources[idx].name}] fetch error: {exc}"],
                    )
        return results  # type: ignore[return-value]

    def _extract(self, fetch_results: list) -> list[FeatureRecord]:
        records: list[FeatureRecord] = []
        for fetch_result in fetch_results:
            records.extend(self.extractor.extract_many(fetch_result.releases, fetch_result.source))
        return records


# ---------------------------------------------------------------------------
# Convenience: one-call entry used by the Cron task and the CLI.
# ---------------------------------------------------------------------------


def run_community_scan(
    *,
    config: RadarConfig | None = None,
    registry: SourceRegistry | None = None,
    output_dir: Path | str | None = None,
    period: str = "weekly",
    sources: Iterable[WatchSource] | None = None,
    incremental: bool = False,
) -> ScanResult:
    """Run a single scan and return the result.

    Both ``clawcodex-dev community-radar scan`` and the durable Cron
    task (``run_community_scan``) call this. It is intentionally
    side-effect-free aside from the dual-write performed by the
    pipeline, so tests can call it against a temp directory.

    When *config* and *registry* are not provided (the typical cron
    path), this function auto-loads them from disk so that
    ``sync_issues``, ``target_repo``, notifications, and other
    file-based settings take effect.
    """
    if config is None:
        config = apply_env_overrides(_load_config_safely())
    if registry is None:
        registry = load_registry_safely()

    # Resolve issue-sync target early when sync_issues is enabled so
    # pipeline.run_scan() receives a validated target (same behaviour
    # as the CLI path).  Failures are logged as warnings rather than
    # raised — the scan still completes, and sync_features_to_issues()
    # has its own internal resolve_target() fallback.
    issue_sync_target: Any = None
    issue_sync_cli_repo: str | None = None
    issue_sync_cli_platform: str | None = None
    if config.sync_issues:
        from .issue_platforms import resolve_target  # lazy — avoids circular dep

        target = resolve_target(
            config_target_repo=config.target_repo,
            config_api_token=config.api_token,
        )
        if target is None:
            _log.warning(
                "sync_issues is enabled but no target repo could be resolved. "
                "Set target_repo in ~/.clawcodex/community-radar/config.yaml "
                "or configure a git remote."
            )
        elif not target.api_token:
            _log.warning(
                "sync_issues is enabled but no API token found for %s. Set %s environment variables.",
                target.platform.name,
                ", ".join(target.platform.token_env_vars),
            )
        else:
            issue_sync_target = target

    pipeline = CommunityRadarPipeline(
        config=config,
        registry=registry,
    )
    return pipeline.run_scan(
        sources=sources,
        period=period,
        write=True,
        output_dir=output_dir,
        persistent_copy=True,
        incremental=incremental,
        issue_sync_target=issue_sync_target,
        issue_sync_cli_repo=issue_sync_cli_repo,
        issue_sync_cli_platform=issue_sync_cli_platform,
    )

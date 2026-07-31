"""Fast stale-while-revalidate cache for provider model catalogs."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

from clawcodex_ext.utils.file_lock import exclusive_file_lock  # pylint: disable=no-name-in-module


MODEL_CATALOG_TTL_SECONDS = 300.0
MODEL_CATALOG_RETRY_SECONDS = 30.0
MODEL_CATALOG_SYNC_WAIT_SECONDS = 6.5
_GLOBAL_CACHE: ModelCatalogCache | None = None
_GLOBAL_CACHE_PATH: Path | None = None
_GLOBAL_CACHE_LOCK = threading.Lock()


@dataclass(frozen=True)
class ModelCatalogSnapshot:
    """Immutable view of a provider model catalog and refresh state."""

    models: list[str]
    source: str
    refreshing: bool
    error: str | None = None


class ModelCatalogCache:
    """Return cached/fallback models immediately and refresh out of band."""

    def __init__(
        self,
        *,
        cache_path: Path,
        ttl_seconds: float = MODEL_CATALOG_TTL_SECONDS,
        retry_seconds: float = MODEL_CATALOG_RETRY_SECONDS,
        sync_wait_seconds: float = MODEL_CATALOG_SYNC_WAIT_SECONDS,
        clock: Callable[[], float] = time.time,
        schedule: Callable[[Callable[[], None]], None] | None = None,
    ) -> None:
        self.cache_path = Path(cache_path)
        self.ttl_seconds = ttl_seconds
        self.retry_seconds = retry_seconds
        self.sync_wait_seconds = sync_wait_seconds
        self._clock = clock
        self._schedule = schedule or self._schedule_thread
        self._entries: dict[str, tuple[list[str], float]] = {}
        self._refreshing: set[str] = set()
        self._refresh_events: dict[str, threading.Event] = {}
        self._errors: dict[str, tuple[str, float]] = {}
        self._lock = threading.Lock()
        self._load()

    def get_catalog(self, provider_name: str, provider: Any) -> ModelCatalogSnapshot:
        """Return cached or fallback models and schedule stale refreshes.

        Args:
            provider_name: Stable provider identifier.
            provider: Runtime provider used for fallback and live discovery.

        Returns:
            The immediately available catalog and its refresh state.
        """

        now = self._clock()
        with self._lock:
            entry = self._entries.get(provider_name)
            if entry is not None and now - entry[1] <= self.ttl_seconds:
                return ModelCatalogSnapshot(list(entry[0]), "cache", False)
            error_info = self._errors.get(provider_name)
            error = error_info[0] if error_info is not None else None
            retry_allowed = error_info is None or now - error_info[1] >= self.retry_seconds
            refreshing = provider_name in self._refreshing
            if not refreshing and retry_allowed:
                self._refreshing.add(provider_name)
                self._refresh_events[provider_name] = threading.Event()

        if not refreshing and retry_allowed:
            try:
                self._schedule(lambda: self._refresh(provider_name, provider))
                refreshing = True
            except Exception as exc:
                error = str(exc)
                refreshing = False
                with self._lock:
                    self._refreshing.discard(provider_name)
                    event = self._refresh_events.pop(provider_name, None)
                    if event is not None:
                        event.set()
                    self._errors[provider_name] = (error, self._clock())
                    self._save_locked()

        if entry is not None:
            return ModelCatalogSnapshot(list(entry[0]), "stale-cache", refreshing, error)
        try:
            fallback = list(provider.get_available_models() or [])
        except Exception as exc:
            fallback = []
            error = error or str(exc)
        return ModelCatalogSnapshot(
            list(dict.fromkeys(str(model) for model in fallback)),
            "fallback",
            refreshing,
            error,
        )

    def refresh_catalog(self, provider_name: str, provider: Any) -> ModelCatalogSnapshot:
        """Refresh a stale catalog synchronously for one-shot CLI processes."""
        now = self._clock()
        with self._lock:
            entry = self._entries.get(provider_name)
            if entry is not None and now - entry[1] <= self.ttl_seconds:
                return ModelCatalogSnapshot(list(entry[0]), "cache", False)
            error_info = self._errors.get(provider_name)
            retry_allowed = error_info is None or now - error_info[1] >= self.retry_seconds
            if provider_name in self._refreshing:
                event = self._refresh_events[provider_name]
                should_refresh = False
            elif not retry_allowed:
                event = None
                should_refresh = False
            else:
                self._refreshing.add(provider_name)
                event = threading.Event()
                self._refresh_events[provider_name] = event
                should_refresh = True
        if should_refresh:
            try:
                self._schedule_thread(lambda: self._refresh(provider_name, provider))
            except Exception as exc:
                with self._lock:
                    self._refreshing.discard(provider_name)
                    self._refresh_events.pop(provider_name, None)
                    self._errors[provider_name] = (str(exc), self._clock())
                    self._save_locked()
                event.set()
        if event is not None:
            event.wait(timeout=self.sync_wait_seconds)
        return self.get_catalog(provider_name, provider)

    def _refresh(self, provider_name: str, provider: Any) -> None:
        try:
            discovered = list(provider.discover_available_models() or [])
            models = list(dict.fromkeys(str(model) for model in discovered if model))
            if not models:
                raise RuntimeError("model discovery returned no models")
            with self._lock:
                self._entries[provider_name] = (models, self._clock())
                self._errors.pop(provider_name, None)
                self._save_locked()
        except Exception as exc:
            with self._lock:
                self._errors[provider_name] = (str(exc), self._clock())
                self._save_locked()
        finally:
            with self._lock:
                self._refreshing.discard(provider_name)
                event = self._refresh_events.pop(provider_name, None)
                if event is not None:
                    event.set()

    def _load(self) -> None:
        self._entries, self._errors = self._read_disk_state()

    def _read_disk_state(
        self,
    ) -> tuple[dict[str, tuple[list[str], float]], dict[str, tuple[str, float]]]:
        entries: dict[str, tuple[list[str], float]] = {}
        errors: dict[str, tuple[str, float]] = {}
        with suppress(Exception):
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
            providers = payload.get("providers", {})
            for key, item in providers.items():
                cache_key = str(key)
                models = item.get("models", [])
                updated_at = float(item.get("updated_at", 0))
                if isinstance(models, list) and updated_at > 0:
                    entries[cache_key] = ([str(model) for model in models], updated_at)
                error = item.get("error")
                failed_at = float(item.get("failed_at", 0))
                if isinstance(error, str) and error and failed_at > 0:
                    errors[cache_key] = (error, failed_at)
        return entries, errors

    def _save_locked(self) -> None:
        temporary = self.cache_path.with_name(f"{self.cache_path.name}.{os.getpid()}.tmp")
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            lock_path = self.cache_path.with_name(f"{self.cache_path.name}.lock")
            with exclusive_file_lock(lock_path):
                disk_entries, disk_errors = self._read_disk_state()
                merged_entries = _merge_newest(disk_entries, self._entries)
                merged_errors = _merge_newest(disk_errors, self._errors)
                for key, (_, updated_at) in merged_entries.items():
                    error_info = merged_errors.get(key)
                    if error_info is not None and updated_at >= error_info[1]:
                        merged_errors.pop(key, None)
                keys = merged_entries.keys() | merged_errors.keys()
                payload = {
                    "version": 1,
                    "providers": {
                        key: _serialize_entry(
                            merged_entries.get(key),
                            merged_errors.get(key),
                        )
                        for key in keys
                    },
                }
                temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
                temporary.replace(self.cache_path)
                self._entries = merged_entries
                self._errors = merged_errors
        except Exception:
            try:
                temporary.unlink(missing_ok=True)
            except Exception:  # nosec B110
                pass

    @staticmethod
    def _schedule_thread(target: Callable[[], None]) -> None:
        threading.Thread(target=target, name="model-catalog-refresh", daemon=True).start()


def default_model_catalog_cache_path() -> Path:
    """Return the process-configured persistent model-catalog cache path."""

    root = Path(os.environ.get("CLAWCODEX_HOME") or Path.home() / ".clawcodex")
    return root / "cache" / "model-catalogs.json"


def get_model_catalog(provider_name: str, provider: Any) -> ModelCatalogSnapshot:
    """Return the shared cached catalog without blocking for discovery."""

    cache = _get_global_cache()
    return cache.get_catalog(_catalog_cache_key(provider_name, provider), provider)


def refresh_model_catalog(provider_name: str, provider: Any) -> ModelCatalogSnapshot:
    """Refresh and return the shared catalog for a one-shot caller."""

    cache = _get_global_cache()
    return cache.refresh_catalog(_catalog_cache_key(provider_name, provider), provider)


def prewarm_model_catalog(provider_name: str, provider: Any) -> None:
    """Schedule a background refresh for a provider catalog."""

    get_model_catalog(provider_name, provider)


def reset_model_catalog_cache() -> None:
    """Reset the process-global cache instance for isolation and tests."""

    global _GLOBAL_CACHE, _GLOBAL_CACHE_PATH
    with _GLOBAL_CACHE_LOCK:
        _GLOBAL_CACHE = None
        _GLOBAL_CACHE_PATH = None


def _get_global_cache() -> ModelCatalogCache:
    global _GLOBAL_CACHE, _GLOBAL_CACHE_PATH
    path = default_model_catalog_cache_path()
    with _GLOBAL_CACHE_LOCK:
        if _GLOBAL_CACHE is None or _GLOBAL_CACHE_PATH != path:
            _GLOBAL_CACHE = ModelCatalogCache(cache_path=path)
            _GLOBAL_CACHE_PATH = path
        return _GLOBAL_CACHE


def _catalog_cache_key(provider_name: str, provider: Any) -> str:
    base_url = str(getattr(provider, "base_url", "") or "default")
    scope_getter = getattr(provider, "model_catalog_cache_scope", None)
    try:
        raw_scope = scope_getter() if callable(scope_getter) else None
    except Exception:
        raw_scope = None
    if not raw_scope:
        raw_scope = getattr(provider, "api_key", "") or "anonymous"
    scope = hashlib.sha256(str(raw_scope).encode("utf-8")).hexdigest()
    return f"{provider_name}|{base_url}|{scope}"


def _merge_newest(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    merged = dict(first)
    for key, value in second.items():
        previous = merged.get(key)
        if previous is None or value[1] >= previous[1]:
            merged[key] = value
    return merged


def _serialize_entry(
    entry: tuple[list[str], float] | None,
    error_info: tuple[str, float] | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "models": entry[0] if entry is not None else [],
        "updated_at": entry[1] if entry is not None else 0,
    }
    if error_info is not None:
        payload.update({"error": error_info[0], "failed_at": error_info[1]})
    return payload


__all__ = [
    "MODEL_CATALOG_TTL_SECONDS",
    "MODEL_CATALOG_RETRY_SECONDS",
    "MODEL_CATALOG_SYNC_WAIT_SECONDS",
    "ModelCatalogCache",
    "ModelCatalogSnapshot",
    "default_model_catalog_cache_path",
    "get_model_catalog",
    "prewarm_model_catalog",
    "refresh_model_catalog",
    "reset_model_catalog_cache",
]

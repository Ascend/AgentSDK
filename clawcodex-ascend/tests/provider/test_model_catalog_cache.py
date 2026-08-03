from __future__ import annotations

import threading
import time
from types import SimpleNamespace

from clawcodex_ext.providers.model_catalog_cache import (
    ModelCatalogCache,
    _catalog_cache_key,
)


def test_catalog_returns_fallback_immediately_then_publishes_background_refresh(tmp_path) -> None:
    scheduled = []
    provider = SimpleNamespace(
        get_available_models=lambda: ["fallback-model"],
        discover_available_models=lambda: ["live-model"],
    )
    cache = ModelCatalogCache(
        cache_path=tmp_path / "models.json",
        schedule=scheduled.append,
    )

    first = cache.get_catalog("example", provider)

    assert first.models == ["fallback-model"]
    assert first.source == "fallback"
    assert first.refreshing is True
    assert len(scheduled) == 1

    scheduled.pop()()
    refreshed = cache.get_catalog("example", provider)

    assert refreshed.models == ["live-model"]
    assert refreshed.source == "cache"
    assert refreshed.refreshing is False


def test_catalog_persists_last_known_good_and_keeps_it_when_refresh_fails(tmp_path) -> None:
    now = [1_000.0]
    scheduled = []
    state = {"fail": False}

    def discover():
        if state["fail"]:
            raise RuntimeError("gateway unavailable")
        return ["live-model"]

    provider = SimpleNamespace(
        get_available_models=lambda: ["fallback-model"],
        discover_available_models=discover,
    )
    cache_path = tmp_path / "models.json"
    cache = ModelCatalogCache(
        cache_path=cache_path,
        ttl_seconds=60,
        clock=lambda: now[0],
        schedule=scheduled.append,
    )
    cache.get_catalog("example", provider)
    scheduled.pop()()

    reloaded = ModelCatalogCache(
        cache_path=cache_path,
        ttl_seconds=60,
        clock=lambda: now[0],
        schedule=scheduled.append,
    )
    assert reloaded.get_catalog("example", provider).models == ["live-model"]
    assert scheduled == []

    now[0] += 61
    state["fail"] = True
    stale = reloaded.get_catalog("example", provider)
    assert stale.models == ["live-model"]
    assert stale.source == "stale-cache"
    scheduled.pop()()

    after_failure = reloaded.get_catalog("example", provider)
    assert after_failure.models == ["live-model"]
    assert after_failure.refreshing is False
    assert "gateway unavailable" in (after_failure.error or "")
    assert scheduled == []


def test_catalog_deduplicates_concurrent_background_refreshes(tmp_path) -> None:
    scheduled = []
    provider = SimpleNamespace(
        get_available_models=lambda: ["fallback-model"],
        discover_available_models=lambda: ["live-model"],
    )
    cache = ModelCatalogCache(
        cache_path=tmp_path / "models.json",
        schedule=scheduled.append,
    )

    cache.get_catalog("example", provider)
    cache.get_catalog("example", provider)

    assert len(scheduled) == 1


def test_catalog_still_refreshes_when_local_fallback_is_unavailable(tmp_path) -> None:
    scheduled = []
    provider = SimpleNamespace(
        get_available_models=lambda: (_ for _ in ()).throw(RuntimeError("no local catalog")),
        discover_available_models=lambda: ["live-model"],
    )
    cache = ModelCatalogCache(
        cache_path=tmp_path / "models.json",
        schedule=scheduled.append,
    )

    first = cache.get_catalog("example", provider)

    assert first.models == []
    assert "no local catalog" in (first.error or "")
    assert first.refreshing is True
    scheduled.pop()()
    assert cache.get_catalog("example", provider).models == ["live-model"]


def test_empty_discovery_keeps_fallback_and_throttles_retry(tmp_path) -> None:
    scheduled = []
    calls = []
    cache = ModelCatalogCache(
        cache_path=tmp_path / "models.json",
        schedule=scheduled.append,
    )
    provider = SimpleNamespace(
        get_available_models=lambda: ["fallback-model"],
        discover_available_models=lambda: calls.append(1) or [],
    )

    first = cache.get_catalog("example", provider)
    scheduled.pop()()
    second = cache.get_catalog("example", provider)

    assert first.models == ["fallback-model"]
    assert second.models == ["fallback-model"]
    assert second.refreshing is False
    assert second.error == "model discovery returned no models"
    assert calls == [1]
    assert scheduled == []


def test_schedule_failure_rolls_back_refreshing_state(tmp_path) -> None:
    def fail_schedule(target) -> None:
        raise RuntimeError("thread unavailable")

    cache = ModelCatalogCache(
        cache_path=tmp_path / "models.json",
        schedule=fail_schedule,
    )
    provider = SimpleNamespace(
        get_available_models=lambda: ["fallback-model"],
        discover_available_models=lambda: ["live-model"],
    )

    snapshot = cache.get_catalog("example", provider)

    assert snapshot.models == ["fallback-model"]
    assert snapshot.refreshing is False
    assert snapshot.error == "thread unavailable"
    reloaded = ModelCatalogCache(cache_path=tmp_path / "models.json")
    assert reloaded.get_catalog("example", provider).refreshing is False
    assert reloaded.get_catalog("example", provider).error == "thread unavailable"


def test_synchronous_refresh_wait_is_bounded_for_slow_provider(tmp_path) -> None:
    started = threading.Event()
    release = threading.Event()

    def discover() -> list[str]:
        started.set()
        release.wait(timeout=1)
        return ["live-model"]

    cache = ModelCatalogCache(
        cache_path=tmp_path / "models.json",
        sync_wait_seconds=0.05,
    )
    provider = SimpleNamespace(
        get_available_models=lambda: ["fallback-model"],
        discover_available_models=discover,
    )

    before = time.perf_counter()
    snapshot = cache.refresh_catalog("example", provider)
    elapsed = time.perf_counter() - before

    assert started.wait(timeout=0.2)
    assert elapsed < 0.2
    assert snapshot.models == ["fallback-model"]
    assert snapshot.refreshing is True
    release.set()


def test_synchronous_refresh_waits_for_existing_background_refresh(tmp_path) -> None:
    started = threading.Event()
    release = threading.Event()

    def discover() -> list[str]:
        started.set()
        release.wait(timeout=1)
        return ["live-model"]

    cache = ModelCatalogCache(
        cache_path=tmp_path / "models.json",
        sync_wait_seconds=0.5,
    )
    provider = SimpleNamespace(
        get_available_models=lambda: ["fallback-model"],
        discover_available_models=discover,
    )
    cache.get_catalog("example", provider)
    assert started.wait(timeout=0.2)
    release.set()

    snapshot = cache.refresh_catalog("example", provider)

    assert snapshot.models == ["live-model"]
    assert snapshot.refreshing is False


def test_concurrent_cache_instances_merge_last_known_good_entries(tmp_path) -> None:
    cache_path = tmp_path / "models.json"
    first_tasks = []
    second_tasks = []
    first = ModelCatalogCache(cache_path=cache_path, schedule=first_tasks.append)
    second = ModelCatalogCache(cache_path=cache_path, schedule=second_tasks.append)
    provider_a = SimpleNamespace(
        get_available_models=lambda: ["fallback-a"],
        discover_available_models=lambda: ["live-a"],
    )
    provider_b = SimpleNamespace(
        get_available_models=lambda: ["fallback-b"],
        discover_available_models=lambda: ["live-b"],
    )

    first.get_catalog("provider-a", provider_a)
    second.get_catalog("provider-b", provider_b)
    first_tasks.pop()()
    second_tasks.pop()()

    reloaded = ModelCatalogCache(cache_path=cache_path, schedule=lambda target: None)
    assert reloaded.get_catalog("provider-a", provider_a).models == ["live-a"]
    assert reloaded.get_catalog("provider-b", provider_b).models == ["live-b"]


def test_failure_backoff_is_persisted_between_processes(tmp_path) -> None:
    now = [1_000.0]
    cache_path = tmp_path / "models.json"
    tasks = []
    provider = SimpleNamespace(
        get_available_models=lambda: ["fallback-model"],
        discover_available_models=lambda: (_ for _ in ()).throw(RuntimeError("offline")),
    )
    first = ModelCatalogCache(
        cache_path=cache_path,
        clock=lambda: now[0],
        schedule=tasks.append,
    )
    first.get_catalog("provider", provider)
    tasks.pop()()

    second_tasks = []
    reloaded = ModelCatalogCache(
        cache_path=cache_path,
        clock=lambda: now[0],
        schedule=second_tasks.append,
    )
    snapshot = reloaded.get_catalog("provider", provider)

    assert snapshot.error == "offline"
    assert snapshot.refreshing is False
    assert second_tasks == []


def test_cache_key_isolates_credentials_without_storing_them() -> None:
    first = SimpleNamespace(base_url="https://api.example", api_key="secret-a")
    second = SimpleNamespace(base_url="https://api.example", api_key="secret-b")

    first_key = _catalog_cache_key("provider", first)
    second_key = _catalog_cache_key("provider", second)

    assert first_key != second_key
    assert "secret-a" not in first_key
    assert "secret-b" not in second_key


def test_cache_key_prefers_stable_provider_account_scope() -> None:
    first = SimpleNamespace(
        base_url="https://api.example",
        api_key="rotated-token-a",
        model_catalog_cache_scope=lambda: "account-1",
    )
    rotated = SimpleNamespace(
        base_url="https://api.example",
        api_key="rotated-token-b",
        model_catalog_cache_scope=lambda: "account-1",
    )
    other_account = SimpleNamespace(
        base_url="https://api.example",
        api_key="other-token",
        model_catalog_cache_scope=lambda: "account-2",
    )

    assert _catalog_cache_key("provider", first) == _catalog_cache_key("provider", rotated)
    assert _catalog_cache_key("provider", first) != _catalog_cache_key("provider", other_account)

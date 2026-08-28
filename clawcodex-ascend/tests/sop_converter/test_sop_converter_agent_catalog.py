#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------
# This file is part of the AgentSDK project.
# Copyright (c) 2026 Clawd Codex Team
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# AgentSDK is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

"""Unit tests for :mod:`extensions.sop_converter.agent_catalog`.

Covers:

* round-trip JSON persistence (load → save → load)
* idempotent ``upsert`` (same agent_id re-applied → metadata merged, not
  duplicated)
* sensitive-field redaction (api_key / token / secret / password stripped
  to ``<redacted:env:...>`` placeholders, restored from env on ``get``)
* corrupt-file tolerance (bad JSON / non-dict top-level / unknown version
  all degrade to empty catalog + warning, do **not** raise)
* cross-process reload (save in one process, load in another via fresh
  import)
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from extensions.sop_converter.core.agent_catalog import (
    AgentCatalog,
    AgentCatalogEntry,
    _redact_dsl,
    _redact_value,
    _restore_dsl,
)
from extensions.sop_converter.core.agent_catalog_resolver import (
    HOME_ONLY_ENV,
    HOME_ROOT_ENV,
    CatalogLocation,
    resolve_catalog_path,
)


def _tmp_path() -> Path:
    tmp = tempfile.mkdtemp(prefix="agent_catalog_test_")
    return Path(tmp)


def _make_entry(
    *,
    agent_id: str = "0f40ed92-test",
    sdk_source_dir: str = "/tmp/sdk",
    dsl: dict | None = None,
    model: str = "gpt-4o",
    provider: str = "openai",
    class_name: str = "LLMAgent",
    module_name: str = "openjiuwen.agents.llm",
    init_kwargs: dict | None = None,
    metadata: dict | None = None,
) -> AgentCatalogEntry:
    return AgentCatalogEntry(
        agent_id=agent_id,
        sdk_source_dir=sdk_source_dir,
        dsl=dsl or {"name": "demo", "max_steps": 8},
        model=model,
        provider=provider,
        class_name=class_name,
        module_name=module_name,
        init_kwargs=init_kwargs or {"temperature": 0.2},
        metadata=metadata or {"source_tool": "agentbuilder-build-agent"},
    )


class TestAgentCatalogRoundTrip(unittest.TestCase):
    def test_save_and_load_round_trip(self) -> None:
        path = _tmp_path() / "catalog.json"
        cat = AgentCatalog()
        cat.upsert(_make_entry())
        cat.save(path)

        loaded = AgentCatalog.load(path)
        self.assertEqual(loaded.list_ids(), ["0f40ed92-test"])
        entry = loaded.get("0f40ed92-test")
        self.assertIsNotNone(entry)
        assert entry is not None
        self.assertEqual(entry.model, "gpt-4o")
        self.assertEqual(entry.class_name, "LLMAgent")
        self.assertEqual(entry.dsl["name"], "demo")
        self.assertEqual(entry.dsl["max_steps"], 8)

    def test_save_uses_atomic_replace(self) -> None:
        """save() must use a unique tmp file + os.replace so readers never see a half-written file."""
        path = _tmp_path() / "catalog.json"
        cat = AgentCatalog()
        cat.upsert(_make_entry())
        with patch("os.replace") as replace:
            cat.save(path)
        replace.assert_called_once()
        tmp_arg = Path(replace.call_args[0][0])
        self.assertEqual(tmp_arg.parent, path.parent)
        self.assertTrue(tmp_arg.name.startswith(".catalog.json."))
        self.assertTrue(tmp_arg.name.endswith(".tmp"))


class TestAgentCatalogUpsertIdempotent(unittest.TestCase):
    def test_upsert_same_id_merges_metadata(self) -> None:
        cat = AgentCatalog()
        cat.upsert(_make_entry(metadata={"source_tool": "agentbuilder-build-agent", "v": 1}))
        cat.upsert(_make_entry(metadata={"v": 2, "extra": "x"}))
        entry = cat.get("0f40ed92-test")
        assert entry is not None
        self.assertEqual(entry.metadata["source_tool"], "agentbuilder-build-agent")
        self.assertEqual(entry.metadata["v"], 2)
        self.assertEqual(entry.metadata["extra"], "x")

    def test_upsert_preserves_created_at(self) -> None:
        cat = AgentCatalog()
        cat.upsert(_make_entry())
        first_created = cat.get("0f40ed92-test").created_at  # type: ignore[union-attr]
        cat.upsert(_make_entry(metadata={"v": 2}))
        second_created = cat.get("0f40ed92-test").created_at  # type: ignore[union-attr]
        self.assertEqual(first_created, second_created)

    def test_upsert_different_ids_keeps_both(self) -> None:
        cat = AgentCatalog()
        cat.upsert(_make_entry(agent_id="a"))
        cat.upsert(_make_entry(agent_id="b"))
        self.assertEqual(cat.list_ids(), ["a", "b"])


class TestAgentCatalogCorruptionTolerance(unittest.TestCase):
    def test_missing_file_returns_empty(self) -> None:
        path = _tmp_path() / "missing.json"
        cat = AgentCatalog.load(path)
        self.assertEqual(cat.list_ids(), [])

    def test_invalid_json_returns_empty(self) -> None:
        path = _tmp_path() / "broken.json"
        path.write_text("{not json", encoding="utf-8")
        cat = AgentCatalog.load(path)
        self.assertEqual(cat.list_ids(), [])

    def test_non_dict_top_level_returns_empty(self) -> None:
        path = _tmp_path() / "wrong-shape.json"
        path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
        cat = AgentCatalog.load(path)
        self.assertEqual(cat.list_ids(), [])

    def test_unknown_version_returns_empty(self) -> None:
        path = _tmp_path() / "future.json"
        path.write_text(json.dumps({"version": 999, "entries": {}}), encoding="utf-8")
        cat = AgentCatalog.load(path)
        self.assertEqual(cat.list_ids(), [])
        # The version is preserved so we don't accidentally clobber it on save.
        self.assertEqual(cat.version, 999)

    def test_skips_non_dict_entries(self) -> None:
        path = _tmp_path() / "mixed.json"
        path.write_text(
            json.dumps({"version": 1, "entries": {"good": _make_entry().to_dict(), "bad": "oops"}}),
            encoding="utf-8",
        )
        cat = AgentCatalog.load(path)
        self.assertEqual(cat.list_ids(), ["good"])


class TestAgentCatalogRedaction(unittest.TestCase):
    def test_redact_value_strips_api_key(self) -> None:
        value, env = _redact_value("api_key", "sk-1234", bundle_id="core")
        self.assertEqual(value, "<redacted:env:CLAWCODEX_CORE_API_KEY>")
        self.assertEqual(env, "CLAWCODEX_CORE_API_KEY")

    def test_redact_value_strips_token_and_secret(self) -> None:
        for key in ("token", "secret", "password", "Access_Token"):
            value, env = _redact_value(key, "raw", bundle_id="b")
            assert env is not None
            self.assertTrue(value.startswith("<redacted:env:"))
            self.assertIn(key.upper().replace("__", "_"), env)

    def test_redact_value_passes_through_non_sensitive(self) -> None:
        value, env = _redact_value("model", "gpt-4o", bundle_id="b")
        self.assertEqual(value, "gpt-4o")
        self.assertIsNone(env)

    def test_redact_value_passes_through_already_redacted(self) -> None:
        """An env:... reference (from a previous round) must not be re-wrapped."""
        value, env = _redact_value("api_key", "env:CLAWCODEX_BUNDLE_API_KEY", bundle_id="b")
        self.assertEqual(value, "env:CLAWCODEX_BUNDLE_API_KEY")
        self.assertIsNone(env)

    def test_redact_dsl_walks_nested(self) -> None:
        env_vars: list[str] = []
        out = _redact_dsl(
            {
                "creds": {"api_key": "sk-1", "api_secret": "sec-2"},
                "model": "gpt-4o",
                "list": [{"token": "tok"}],
            },
            bundle_id="b",
            env_vars=env_vars,
        )
        self.assertEqual(out["model"], "gpt-4o")
        self.assertTrue(out["creds"]["api_key"].startswith("<redacted:env:"))
        self.assertTrue(out["creds"]["api_secret"].startswith("<redacted:env:"))
        self.assertTrue(out["list"][0]["token"].startswith("<redacted:env:"))
        self.assertGreaterEqual(len(env_vars), 3)

    def test_restore_dsl_resolves_env_references(self) -> None:
        with patch.dict(
            os.environ,
            {"CLAWCODEX_BUNDLE_API_KEY": "sk-real"},
        ):
            out = _restore_dsl(
                {
                    "api_key": "<redacted:env:CLAWCODEX_BUNDLE_API_KEY>",
                    "model": "gpt-4o",
                }
            )
            self.assertEqual(out["api_key"], "sk-real")
            self.assertEqual(out["model"], "gpt-4o")

    def test_restore_dsl_keeps_placeholder_if_env_missing(self) -> None:
        out = _restore_dsl({"api_key": "<redacted:env:CLAWCODEX_MISSING_VAR>"})
        self.assertEqual(out["api_key"], "<redacted:env:CLAWCODEX_MISSING_VAR>")

    def test_get_restores_redacted_fields_from_env(self) -> None:
        cat = AgentCatalog()
        cat.upsert(
            _make_entry(
                dsl={"api_key": "sk-should-not-persist"},
                init_kwargs={"api_key": "sk-should-not-persist"},
            ),
            bundle_id="bundle",
        )
        with patch.dict(os.environ, {"CLAWCODEX_BUNDLE_API_KEY": "sk-restored"}):
            entry = cat.get("0f40ed92-test")
        assert entry is not None
        self.assertEqual(entry.dsl["api_key"], "sk-restored")
        self.assertEqual(entry.init_kwargs["api_key"], "sk-restored")

    def test_metadata_records_env_vars(self) -> None:
        cat = AgentCatalog()
        cat.upsert(_make_entry(dsl={"api_key": "sk-1"}), bundle_id="b")
        entry = cat.entries["0f40ed92-test"]
        self.assertIn("env_vars", entry.metadata)
        self.assertTrue(any(v.startswith("CLAWCODEX_B_API_KEY") for v in entry.metadata["env_vars"]))


class TestAgentCatalogCrossProcess(unittest.TestCase):
    def test_save_then_reload_returns_same_data(self) -> None:
        path = _tmp_path() / "cross.json"
        cat = AgentCatalog()
        cat.upsert(_make_entry(agent_id="x", model="gpt-4o"))
        cat.save(path)
        # Simulate a fresh process by constructing a new AgentCatalog.
        cat2 = AgentCatalog.load(path)
        self.assertEqual(cat2.list_ids(), ["x"])
        self.assertEqual(cat2.get("x").model, "gpt-4o")  # type: ignore[union-attr]


class TestCatalogResolver(unittest.TestCase):
    def setUp(self) -> None:
        # Ensure no stray env vars from the host affect path resolution.
        self._saved_env = {
            HOME_ROOT_ENV: os.environ.pop(HOME_ROOT_ENV, None),
            HOME_ONLY_ENV: os.environ.pop(HOME_ONLY_ENV, None),
        }

    def tearDown(self) -> None:
        for k, v in self._saved_env.items():
            if v is not None:
                os.environ[k] = v

    def test_bundle_local_when_bundle_provided(self) -> None:
        bundle = _tmp_path() / "mybundle"
        bundle.mkdir()
        loc = resolve_catalog_path(bundle)
        self.assertEqual(loc.reason, "bundle-local")
        self.assertEqual(loc.path, bundle / ".clawcodex" / "agent-catalog.json")

    def test_home_fallback_when_no_bundle(self) -> None:
        loc = resolve_catalog_path(None, bundle_id="mybundle")
        self.assertEqual(loc.reason, "no-bundle")
        self.assertEqual(loc.path, Path("~/.clawcodex/sop-agents/mybundle/agent-catalog.json").expanduser())

    def test_home_forced_when_env_set(self) -> None:
        bundle = _tmp_path() / "mybundle"
        bundle.mkdir()
        os.environ[HOME_ONLY_ENV] = "1"
        loc = resolve_catalog_path(bundle)
        self.assertEqual(loc.reason, "home-forced")
        self.assertEqual(loc.path.parts[-3:], ("sop-agents", "mybundle", "agent-catalog.json"))

    def test_explicit_home_only_true(self) -> None:
        bundle = _tmp_path() / "mybundle"
        bundle.mkdir()
        loc = resolve_catalog_path(bundle, home_only=True)
        self.assertEqual(loc.reason, "home-forced")

    def test_bundle_id_overrides_path_name(self) -> None:
        bundle = _tmp_path() / "mybundle"
        bundle.mkdir()
        loc = resolve_catalog_path(bundle, bundle_id="renamed")
        self.assertEqual(loc.path, bundle / ".clawcodex" / "agent-catalog.json")
        # Falls through to home fallback with the new name.
        home_loc = resolve_catalog_path(bundle, bundle_id="renamed", home_only=True)
        self.assertEqual(home_loc.path.parts[-3:], ("sop-agents", "renamed", "agent-catalog.json"))

    def test_ensure_parent_creates_directory(self) -> None:
        bundle = _tmp_path() / "mybundle"
        bundle.mkdir()
        loc = resolve_catalog_path(bundle)
        self.assertFalse(loc.path.parent.exists())
        loc.ensure_parent()
        self.assertTrue(loc.path.parent.is_dir())

    def test_resolve_accepts_string_path(self) -> None:
        bundle = _tmp_path() / "abc"
        bundle.mkdir()
        loc = resolve_catalog_path(str(bundle))
        self.assertEqual(loc.path, bundle / ".clawcodex" / "agent-catalog.json")

    def test_writable_probe_on_existing_dir(self) -> None:
        bundle = _tmp_path() / "mybundle"
        bundle.mkdir()
        loc = resolve_catalog_path(bundle)
        self.assertTrue(loc.writable is True or loc.writable is None)

    def test_writable_probe_is_none_when_parent_missing(self) -> None:
        bundle = _tmp_path() / "ghostbundle"
        loc = resolve_catalog_path(bundle)
        self.assertIsNone(loc.writable)

    def test_home_override(self) -> None:
        root = _tmp_path() / "home"
        root.mkdir()
        with patch.dict(os.environ, {HOME_ROOT_ENV: str(root)}):
            loc = resolve_catalog_path(None, bundle_id="b")
            self.assertEqual(loc.path, root.resolve() / "sop-agents" / "b" / "agent-catalog.json")


class TestCatalogLocationDataclass(unittest.TestCase):
    def test_frozen(self) -> None:
        loc = CatalogLocation(path=Path("/x"), reason="r", writable=True)
        with self.assertRaises(Exception):
            loc.reason = "other"  # type: ignore[misc]


class TestAgentCatalogConcurrency(unittest.TestCase):
    """Concurrent ``load → upsert → save`` must not corrupt or lose entries."""

    def test_concurrent_upserts_persist_all_entries(self) -> None:
        # Windows file replace can still race with AV/indexers even under the
        # catalog lock; retry the whole scenario a few times before failing.
        last_errors: list[tuple[int, str]] = []
        for _attempt in range(3):
            path = Path(tempfile.mkdtemp(prefix="agent_catalog_race_")) / "catalog.json"
            errors: list[tuple[int, str]] = []

            def upsert(i: int) -> None:
                try:
                    cat = AgentCatalog.load(path)
                    cat.upsert(
                        AgentCatalogEntry(
                            agent_id=f"agent-{i:03d}",
                            sdk_source_dir="/tmp/sdk",
                            dsl={"idx": i},
                        )
                    )
                    cat.save(path)
                except Exception as exc:  # pragma: no cover
                    errors.append((i, str(exc)))

            threads = [threading.Thread(target=upsert, args=(i,)) for i in range(30)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            last_errors = errors
            if not errors:
                final = AgentCatalog.load(path)
                self.assertCountEqual(final.list_ids(), [f"agent-{i:03d}" for i in range(30)])
                return

        self.assertEqual(last_errors, [])


if __name__ == "__main__":
    unittest.main()
